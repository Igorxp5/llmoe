/* olmoe_forward orchestrator + scratch (de)alloc. Wires the per-kind ops
 * from engine_embed/norm/attn/mlp plus the RoPE/SDPA/SiLU kernels in the
 * OLMoE forward order. See docs/engine_module.md. */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "olmoe/engine/engine.h"
#include "olmoe/engine/engine_internal.h"
#include "kernels/kernels.h"
#include "kernels/cpu_rope.h"
#include "kernels/cpu_sdpa.h"
#include "kernels/cpu_silu.h"

static inline olmoe_act_t *alloc_act_buffer(size_t n)
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

olmoe_status_t olmoe_scratch_init(olmoe_scratch_t * restrict s,
                                   size_t seq_len, size_t max_cache_len)
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
    size_t i_cnt   = olmoe_engine_safe_array_size(seq_len,
                                                   OLMOE_INTER);
    if (!h_cnt || !v_cnt || !r_cnt || !k_cnt || !i_cnt) {
        return OLMOE_ERR_SHAPE;
    }
    /* SDPA scores: OLMOE_NUM_HEADS per-head slices of the largest token
     * span any forward call can hit (prefill seq_len, decode cache_len). */
    size_t s_cnt = olmoe_engine_safe_array_size(
                       seq_len > max_cache_len ? seq_len : max_cache_len,
                       OLMOE_NUM_HEADS);
    if (!s_cnt) {
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
    s->scores        = alloc_act_buffer(s_cnt);
    if (!s->hidden_in || !s->hidden_out || !s->q || !s->k || !s->v ||
        !s->ctx || !s->router_logits || !s->topk_w || !s->expert_in ||
        !s->expert_out || !s->logits || !s->topk_idx || !s->scores) {
        olmoe_scratch_free(s);
        return OLMOE_ERR_ALLOC;
    }
    s->seq_len = seq_len;

    /* Optional KV cache — allocated only when max_cache_len > 0. */
    if (max_cache_len > 0) {
        size_t cache_cnt = olmoe_engine_safe_array_size(
            (size_t)OLMOE_N_LAYERS * max_cache_len, OLMOE_HIDDEN);
        if (!cache_cnt) {
            olmoe_scratch_free(s);
            return OLMOE_ERR_SHAPE;
        }
        s->cache_k = alloc_act_buffer(cache_cnt);
        s->cache_v = alloc_act_buffer(cache_cnt);
        if (!s->cache_k || !s->cache_v) {
            olmoe_scratch_free(s);
            return OLMOE_ERR_ALLOC;
        }
        s->max_cache_len = max_cache_len;
    }
    return OLMOE_OK;
}

void olmoe_scratch_free(olmoe_scratch_t * restrict s)
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
    free(s->scores);
    free(s->cache_k);
    free(s->cache_v);
    memset(s, 0, sizeof *s);
}

/* Element-wise residual add: out[i] += x[i] for n lanes. Reused by both the
 * attention and MoE block residuals so the integrator names the buffer
 * carrying the live residual stream in exactly one place. */
static inline void add_residual(olmoe_act_t * restrict out,
                         const olmoe_act_t * restrict x, size_t n)
{
    /* Mask every 16-lane block rather than splitting a full-width fast path
     * from a masked tail: GCC miscompiles the branchy form (a `lanes == 16`
     * __builtin_expect branch inside the OpenMP stepped loop produced
     * wrong results for n >= 100, first-bad 25a8615, found via git bisect
     * against benchmark.expected.txt). Branchless masking is correct for all
     * n and every thread count. */
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i += 16) {
        size_t lanes = n - i; if (lanes > 16) lanes = 16;
        __mmask16 mask = (__mmask16)((1u << lanes) - 1);
        __m512 xv = _mm512_mask_loadu_ps(_mm512_setzero_ps(), mask, x + i);
        __m512 ov = _mm512_mask_loadu_ps(_mm512_setzero_ps(), mask, out + i);
        _mm512_mask_storeu_ps(out + i, mask, _mm512_add_ps(ov, xv));
    }
}

/* Attention block with KV-cache support.  When s->cache_len > 0 the
 * function skips the full causal SDPA and instead runs incremental
 * attention for the `seq` new tokens against the cached K/V history.
 *
 * K/V for the current tokens are stored into the cache *before* the
 * SDPA step so the incremental kernel sees its own position too. */
static void attention_block(const olmoe_layer_t * restrict L,
                            const olmoe_act_t * restrict x,
                            size_t seq, size_t pos, size_t l,
                            olmoe_scratch_t * restrict s,
                            olmoe_act_t * restrict out)
{
    olmoe_input_ln_forward(L->input_layernorm, x, seq, s->ctx);
    olmoe_q_proj_forward(&L->self_attn, s->ctx, seq, s->q);
    olmoe_k_proj_forward(&L->self_attn, s->ctx, seq, s->k);
    olmoe_v_proj_forward(&L->self_attn, s->ctx, seq, s->v);
    olmoe_q_norm_forward(L->self_attn.q_norm, s->q, seq);
    olmoe_k_norm_forward(L->self_attn.k_norm, s->k, seq);
    cpu_rope(s->q, seq, pos, OLMOE_NUM_HEADS, OLMOE_HEAD_DIM, OLMOE_ROPE_THETA);
    cpu_rope(s->k, seq, pos, OLMOE_NUM_HEADS, OLMOE_HEAD_DIM, OLMOE_ROPE_THETA);

    /* Store K/V into the per-layer KV cache (if available). */
    if (s->cache_k) {
        size_t layer_offset = l * s->max_cache_len * (size_t)OLMOE_HIDDEN;
        size_t slot_offset = pos * (size_t)OLMOE_HIDDEN;
        size_t n = seq * (size_t)OLMOE_HIDDEN;
        memcpy(s->cache_k + layer_offset + slot_offset, s->k,
               n * sizeof(olmoe_act_t));
        memcpy(s->cache_v + layer_offset + slot_offset, s->v,
               n * sizeof(olmoe_act_t));
    }

    float scale = 1.0f / sqrtf((float)OLMOE_HEAD_DIM);
    if (__builtin_expect(s->cache_len > 0, 1)) {
        /* Incremental decode: compute attention only for the new tokens. */
        size_t layer_offset = l * s->max_cache_len * (size_t)OLMOE_HIDDEN;
        cpu_sdpa_incremental(s->ctx, s->q,
                             s->cache_k + layer_offset,
                             s->cache_v + layer_offset,
                             seq, pos, OLMOE_NUM_HEADS,
                             OLMOE_HEAD_DIM, scale,
                             s->scores, pos + seq);
    } else {
        /* First call (prefill, no history): full causal SDPA. */
        cpu_sdpa(s->ctx, s->q, s->k, s->v, seq, OLMOE_NUM_HEADS,
                 OLMOE_HEAD_DIM, scale, s->scores, seq);
    }
    olmoe_o_proj_forward(&L->self_attn, s->ctx, seq, out);
    add_residual(out, x, seq * (size_t)OLMOE_HIDDEN);
}

/* Run one selected expert on one token, then accumulate its scaled
 * down-projection into the MoE accumulator row. The per-token [inter]x3 and
 * [hidden] scratch buffers are owned by moe_block and reused across all
 * (token, expert) pairs of the block to avoid per-call allocation. */
static inline void expert_accumulate(const olmoe_expert_t * restrict e,
                              const olmoe_act_t * restrict tok,
                              olmoe_act_t * restrict acc_row, float w,
                              float * restrict gate, float * restrict up,
                              float * restrict down, float * restrict act)
{
    olmoe_expert_gate_forward(e, tok, 1, gate);
    olmoe_expert_up_forward(e, tok, 1, up);
    #pragma omp parallel for simd schedule(static)
    for (size_t j = 0; j < (size_t)OLMOE_INTER; ++j)
        act[j] = cpu_silu(gate[j]) * up[j];
    olmoe_expert_down_forward(e, act, 1, down);
    const __m512 vw = _mm512_set1_ps(w);
    #pragma omp parallel for simd schedule(static)
    for (size_t h = 0; h < (size_t)OLMOE_HIDDEN; h += 16)
        _mm512_storeu_ps(acc_row + h,
                         _mm512_fmadd_ps(vw, _mm512_loadu_ps(down + h),
                                         _mm512_loadu_ps(acc_row + h)));
}

/* Resolve the (token, expert-slot) routing pair and fold that expert's
 * scaled contribution into `acc_row`. `acc_row` MUST be private to the
 * calling iteration: concurrent iterations may fold the same token's slots
 * into it, and expert_accumulate's final loop is a non-atomic RMW. */
static inline void expert_accumulate_slot(const olmoe_layer_t * restrict L,
                                          const olmoe_scratch_t * restrict s,
                                          size_t i, size_t r,
                                          olmoe_act_t * restrict acc_row)
{
    int e = s->topk_idx[i * OLMOE_N_EXPERTS_PER_TOK + r];
    float w = s->topk_w[i * OLMOE_N_EXPERTS_PER_TOK + r];
    float gate[OLMOE_INTER], up[OLMOE_INTER], act[OLMOE_INTER];
    float down[OLMOE_HIDDEN];
    expert_accumulate(&L->experts[e], s->ctx + i * (size_t)OLMOE_HIDDEN,
                      acc_row, w, gate, up, down, act);
}

/* Decode (seq == 1): the token's 8 expert slots run on separate threads,
 * each folding into its own row of `acc_slots` (disjoint writers, so the
 * expert_accumulate RMW is race-free), then the rows are combined into
 * s->expert_out in ascending slot order so the sum is bit-reproducible
 * run-to-run. collapse(2) here was a data race — see
 * docs/moe_expert_accumulate_race.md. */
static void moe_block_decode(const olmoe_layer_t * restrict L,
                             const olmoe_scratch_t * restrict s)
{
    olmoe_act_t acc_slots[OLMOE_N_EXPERTS_PER_TOK][OLMOE_HIDDEN] = {{0}};
    #pragma omp parallel for schedule(static)
    for (size_t r = 0; r < (size_t)OLMOE_N_EXPERTS_PER_TOK; ++r)
        expert_accumulate_slot(L, s, 0, r, acc_slots[r]);
    for (size_t r = 0; r < (size_t)OLMOE_N_EXPERTS_PER_TOK; ++r)
        for (size_t h = 0; h < (size_t)OLMOE_HIDDEN; h += 16)
            _mm512_storeu_ps(s->expert_out + h,
                _mm512_add_ps(_mm512_loadu_ps(s->expert_out + h),
                              _mm512_loadu_ps(acc_slots[r] + h)));
}

/* Prefill (seq > 1): one token per iteration; the token's 8 expert slots
 * fold serially into its own row, which is disjoint across threads. */
static void moe_block_prefill(const olmoe_layer_t * restrict L,
                              const olmoe_scratch_t * restrict s,
                              size_t seq)
{
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < seq; ++i)
        for (size_t r = 0; r < (size_t)OLMOE_N_EXPERTS_PER_TOK; ++r)
            expert_accumulate_slot(L, s, i, r,
                                   s->expert_out + i * (size_t)OLMOE_HIDDEN);
}

/* MoE block: post_ln(x) -> mlp_gate -> top-K expert dispatch with SiLU
 * gating and weighted down-projection accumulated into s->expert_out, then
 * residual added into `out`. s->ctx holds the post_ln output (normed2). */
/* NOTE: `x` and `out` are intentionally NOT restrict-qualified — run_layer
 * passes the same live residual buffer as both (see call site below), so the
 * two genuinely alias; `restrict` would be a lie. */
static void moe_block(const olmoe_layer_t * restrict L, const olmoe_act_t *x,
                      size_t seq, olmoe_scratch_t * restrict s,
                      olmoe_act_t *out)
{
    olmoe_post_ln_forward(L->post_attention_layernorm, x, seq, s->ctx);
    olmoe_mlp_gate_forward(L->mlp_gate, s->ctx, seq, s->topk_idx, s->topk_w);
    memset(s->expert_out, 0, seq * (size_t)OLMOE_HIDDEN * sizeof(olmoe_act_t));
    if (__builtin_expect(seq == 1, 1))
        moe_block_decode(L, s);
    else
        moe_block_prefill(L, s, seq);
    add_residual(out, s->expert_out, seq * (size_t)OLMOE_HIDDEN);
}

/* One transformer layer: attention block (residual into out), then MoE
 * block (residual into the same out). `out` ends the layer holding the
 * live residual stream for the next layer; `x` is this layer's input. */
static void run_layer(const olmoe_layer_t * restrict L,
                      const olmoe_act_t * restrict x,
                      size_t seq, size_t pos, size_t l,
                      olmoe_scratch_t * restrict s,
                      olmoe_act_t * restrict out)
{
    attention_block(L, x, seq, pos, l, s, out);
    moe_block(L, out, seq, s, out);
}

olmoe_status_t olmoe_forward(const olmoe_model_t * restrict m,
                             const int * restrict token_ids,
                             size_t seq_len, size_t pos,
                             olmoe_scratch_t * restrict scratch,
                             olmoe_act_t * restrict logits_out)
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
    /* Check KV-cache capacity (if cache is active). */
    if (scratch->max_cache_len > 0 &&
        pos + seq_len > scratch->max_cache_len) {
        return OLMOE_ERR_SHAPE;
    }
    olmoe_embed_forward(m, token_ids, seq_len, scratch->hidden_in);
    olmoe_act_t *h_in = scratch->hidden_in;
    olmoe_act_t *h_out = scratch->hidden_out;
    for (size_t l = 0; l < (size_t)OLMOE_N_LAYERS; ++l) {
        run_layer(&m->layers[l], h_in, seq_len, pos, l, scratch, h_out);
        olmoe_act_t *tmp = h_in; h_in = h_out; h_out = tmp;
    }
    olmoe_final_norm_forward(m->norm, h_in, seq_len, h_out);
    olmoe_lm_head_forward(m, h_out, seq_len, logits_out);
    /* Update cached token count for the next incremental call. */
    if (scratch->max_cache_len > 0) {
        size_t end = pos + seq_len;
        if (end > scratch->cache_len) {
            scratch->cache_len = end;
        }
    }
    return OLMOE_OK;
}
