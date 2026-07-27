/* olmoe_forward orchestrator + scratch (de)alloc. Wires the per-kind ops
 * from engine_embed/norm/attn/mlp plus the RoPE/SDPA/SiLU kernels in the
 * OLMoE forward order. See docs/engine_module.md. */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "olmoe/engine/engine.h"
#include "olmoe/engine/engine_internal.h"
#include "kernels/cpu_rope.h"
#include "kernels/cpu_sdpa.h"
#include "kernels/cpu_silu.h"

static olmoe_act_t *alloc_act_buffer(size_t n)
{
    if (n == 0) {
        return NULL;
    }
    size_t bytes = olmoe_engine_safe_array_size(n, sizeof(olmoe_act_t));
    if (bytes == 0) {
        return NULL;
    }
    return (olmoe_act_t *)malloc(bytes);
}

olmoe_status_t olmoe_scratch_init(olmoe_scratch_t *s, size_t seq_len)
{
    if (!s) {
        return OLMOE_ERR_NULL;
    }
    memset(s, 0, sizeof *s);

    /* Per-buffer element counts; each = seq_len * extent. */
    size_t h_cnt   = olmoe_engine_safe_array_size(seq_len, OLMOE_HIDDEN);
    size_t v_cnt   = olmoe_engine_safe_array_size(seq_len, OLMOE_VOCAB);
    size_t r_cnt   = olmoe_engine_safe_array_size(seq_len, OLMOE_N_EXPERTS);
    size_t k_cnt   = olmoe_engine_safe_array_size(seq_len,
                                                  OLMOE_N_EXPERTS_PER_TOK);
    size_t i_cnt   = olmoe_engine_safe_array_size(seq_len, OLMOE_INTER);
    if (!h_cnt || !v_cnt || !r_cnt || !k_cnt || !i_cnt) {
        return OLMOE_ERR_SHAPE;
    }

    s->hidden_in     = alloc_act_buffer(h_cnt);
    s->hidden_out    = alloc_act_buffer(h_cnt);
    s->q             = alloc_act_buffer(h_cnt);
    s->k             = alloc_act_buffer(h_cnt);
    s->v             = alloc_act_buffer(h_cnt);
    s->ctx           = alloc_act_buffer(h_cnt);
    s->router_logits = alloc_act_buffer(r_cnt);
    s->topk_w        = alloc_act_buffer(k_cnt);
    s->expert_in     = alloc_act_buffer(i_cnt);
    s->expert_out    = alloc_act_buffer(h_cnt);
    s->logits        = alloc_act_buffer(v_cnt);
    s->topk_idx      = (int *)malloc(k_cnt * sizeof(int));
    if (!s->hidden_in || !s->hidden_out || !s->q || !s->k || !s->v ||
        !s->ctx || !s->router_logits || !s->topk_w || !s->expert_in ||
        !s->expert_out || !s->logits || !s->topk_idx) {
        olmoe_scratch_free(s);
        return OLMOE_ERR_ALLOC;
    }
    s->seq_len = seq_len;
    return OLMOE_OK;
}

void olmoe_scratch_free(olmoe_scratch_t *s)
{
    if (!s) {
        return;
    }
    free(s->hidden_in);
    free(s->hidden_out);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->ctx);
    free(s->router_logits);
    free(s->topk_idx);
    free(s->topk_w);
    free(s->expert_in);
    free(s->expert_out);
    free(s->logits);
    memset(s, 0, sizeof *s);
}

/* Element-wise residual add: out[i] += x[i] for n lanes. Reused by both the
 * attention and MoE block residuals so the integrator names the buffer
 * carrying the live residual stream in exactly one place. */
static void add_residual(olmoe_act_t *out, const olmoe_act_t *x, size_t n)
{
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) out[i] += x[i];
}

/* Attention block: input_ln -> q/k/v_proj -> q/k_norm -> RoPE q/k -> causal
 * SDPA -> o_proj, then residual x added into `out`. s->ctx is reused as
 * both the post-input_ln normed buffer (consumed by the projections) and
 * then the SDPA output buffer (overwritten once the projections are done).*/
static void attention_block(const olmoe_layer_t *L, const olmoe_act_t *x,
                            size_t seq, olmoe_scratch_t *s, olmoe_act_t *out)
{
    olmoe_input_ln_forward(L->input_layernorm, x, seq, s->ctx);
    olmoe_q_proj_forward(&L->self_attn, s->ctx, seq, s->q);
    olmoe_k_proj_forward(&L->self_attn, s->ctx, seq, s->k);
    olmoe_v_proj_forward(&L->self_attn, s->ctx, seq, s->v);
    olmoe_q_norm_forward(L->self_attn.q_norm, s->q, seq);
    olmoe_k_norm_forward(L->self_attn.k_norm, s->k, seq);
    cpu_rope(s->q, seq, OLMOE_NUM_HEADS, OLMOE_HEAD_DIM, OLMOE_ROPE_THETA);
    cpu_rope(s->k, seq, OLMOE_NUM_HEADS, OLMOE_HEAD_DIM, OLMOE_ROPE_THETA);
    float scale = 1.0f / sqrtf((float)OLMOE_HEAD_DIM);
    cpu_sdpa(s->ctx, s->q, s->k, s->v, seq, OLMOE_NUM_HEADS,
             OLMOE_HEAD_DIM, scale);
    olmoe_o_proj_forward(&L->self_attn, s->ctx, seq, out);
    add_residual(out, x, seq * (size_t)OLMOE_HIDDEN);
}

/* Run one selected expert on one token, then accumulate its scaled
 * down-projection into the MoE accumulator row. The per-token [inter]x3 and
 * [hidden] scratch buffers are owned by moe_block and reused across all
 * (token, expert) pairs of the block to avoid per-call allocation. */
static void expert_accumulate(const olmoe_expert_t *e, const olmoe_act_t *tok,
                              olmoe_act_t *acc_row, float w,
                              float *gate, float *up, float *down, float *act)
{
    olmoe_expert_gate_forward(e, tok, 1, gate);
    olmoe_expert_up_forward(e, tok, 1, up);
    for (size_t j = 0; j < (size_t)OLMOE_INTER; ++j)
        act[j] = cpu_silu(gate[j]) * up[j];
    olmoe_expert_down_forward(e, act, 1, down);
    for (size_t h = 0; h < (size_t)OLMOE_HIDDEN; ++h)
        acc_row[h] += w * down[h];
}

/* MoE block: post_ln(x) -> mlp_gate -> top-K expert dispatch with SiLU
 * gating and weighted down-projection accumulated into s->expert_out, then
 * residual added into `out`. s->ctx holds the post_ln output (normed2). */
static void moe_block(const olmoe_layer_t *L, const olmoe_act_t *x,
                      size_t seq, olmoe_scratch_t *s, olmoe_act_t *out)
{
    olmoe_post_ln_forward(L->post_attention_layernorm, x, seq, s->ctx);
    olmoe_mlp_gate_forward(L->mlp_gate, s->ctx, seq, s->topk_idx, s->topk_w);
    memset(s->expert_out, 0, seq * (size_t)OLMOE_HIDDEN * sizeof(olmoe_act_t));
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < seq; ++i) {
        float gate[OLMOE_INTER], up[OLMOE_INTER], act[OLMOE_INTER];
        float down[OLMOE_HIDDEN];
        const olmoe_act_t *tok = s->ctx + i * (size_t)OLMOE_HIDDEN;
        olmoe_act_t *acc = s->expert_out + i * (size_t)OLMOE_HIDDEN;
        for (size_t r = 0; r < (size_t)OLMOE_N_EXPERTS_PER_TOK; ++r) {
            int e = s->topk_idx[i * OLMOE_N_EXPERTS_PER_TOK + r];
            float w = s->topk_w[i * OLMOE_N_EXPERTS_PER_TOK + r];
            expert_accumulate(&L->experts[e], tok, acc, w, gate, up, down, act);
        }
    }
    add_residual(out, s->expert_out, seq * (size_t)OLMOE_HIDDEN);
}

/* One transformer layer: attention block (residual into out), then MoE
 * block (residual into the same out). `out` ends the layer holding the
 * live residual stream for the next layer; `x` is this layer's input. */
static void run_layer(const olmoe_layer_t *L, const olmoe_act_t *x,
                      size_t seq, olmoe_scratch_t *s, olmoe_act_t *out)
{
    attention_block(L, x, seq, s, out);
    moe_block(L, out, seq, s, out);
}

olmoe_status_t olmoe_forward(const olmoe_model_t *m, const int *token_ids,
                             size_t seq_len, olmoe_scratch_t *scratch,
                             olmoe_act_t *logits_out)
{
    if (!m || !token_ids || !scratch || !logits_out) {
        return OLMOE_ERR_NULL;
    }
    if (scratch->seq_len < seq_len) {
        return OLMOE_ERR_SHAPE;
    }
    if (seq_len == 0) {
        return OLMOE_OK;
    }
    olmoe_embed_forward(m, token_ids, seq_len, scratch->hidden_in);
    olmoe_act_t *h_in = scratch->hidden_in;
    olmoe_act_t *h_out = scratch->hidden_out;
    /* Loop over m->n_layers (not the baked OLMOE_N_LAYERS) so a synthetic
     * low-RAM model may set n_layers smaller for end-to-end validation. */
    for (size_t l = 0; l < m->n_layers; ++l) {
        run_layer(&m->layers[l], h_in, seq_len, scratch, h_out);
        olmoe_act_t *tmp = h_in; h_in = h_out; h_out = tmp;
    }
    olmoe_final_norm_forward(m->norm, h_in, seq_len, h_out);
    olmoe_lm_head_forward(m, h_out, seq_len, logits_out);
    return OLMOE_OK;
}
