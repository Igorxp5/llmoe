/* olmoe_forward orchestrator + scratch (de)alloc. Stubs only; see
 * docs/engine_module.md. The real impl will walk the layers in order and
 * call the per-kind ops from engine_embed/norm/attn/mlp. */

#include <stdlib.h>
#include <string.h>

#include "olmoe/engine/engine.h"
#include "olmoe/engine/engine_internal.h"

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
    /* TODO: wire embed -> [per-layer: input_ln, q/k/v_proj, q/k_norm, rope,
     * sdpa, o_proj, residual, post_ln, mlp_gate, expert dispatch, residual]
     * -> final_norm -> lm_head -> logits_out. */
    return OLMOE_OK;
}