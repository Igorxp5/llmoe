#ifndef OLMOE_ENGINE_H
#define OLMOE_ENGINE_H

#include <stddef.h>

#include "olmoe/layers/layers.h"

/* Activation dtype. Stored and computed in FP32 during the stub phase;
 * swapping to BF16 later only needs a one-line change here, not a rewrite
 * of every call site. Weights remain BF16 (`olmoe_bf16_t`); conversion is
 * deferred to the per-op impls (TODO when real matmul lands). */
typedef float olmoe_act_t;

/* Engine-wide status. Stubs only ever return OLMOE_OK or OLMOE_ERR_NULL. */
typedef enum {
    OLMOE_OK         = 0,
    OLMOE_ERR_NULL   = 1,
    OLMOE_ERR_SHAPE  = 2,
    OLMOE_ERR_ALLOC  = 3
} olmoe_status_t;

/* OLMoE-1B-7B topology constant the loader does not need (it is not a
 * tensor kind) but the MoE router does: tokens are routed to the top-K of
 * the 64 experts. Lives here rather than in model_layout.inc because the
 * layout generator only bakes quantities derivable from the safetensors
 * index. */
#define OLMOE_N_EXPERTS_PER_TOK 8

/* Caller-owned activation buffers for a single forward pass. The engine
 * never mallocs activations: the caller allocates once via
 * olmoe_scratch_init and reuses across calls. Capex for bearing the typing
 * is one indirection; payoff is no per-call allocator churn.
 *
 * Every member is capacity `seq_len` (the value passed to init) along the
 * sequence dimension; the hidden/inter/vocab extents are the baked
 * OLMOE_* constants. */
typedef struct {
    olmoe_act_t *hidden_in;     /* [seq, OLMOE_HIDDEN]                */
    olmoe_act_t *hidden_out;    /* [seq, OLMOE_HIDDEN]                */
    olmoe_act_t *q;             /* [seq, OLMOE_HIDDEN]                */
    olmoe_act_t *k;             /* [seq, OLMOE_HIDDEN]                */
    olmoe_act_t *v;             /* [seq, OLMOE_HIDDEN]                */
    olmoe_act_t *ctx;           /* [seq, OLMOE_HIDDEN]  attn output    */
    olmoe_act_t *router_logits; /* [seq, OLMOE_N_EXPERTS]             */
    int         *topk_idx;      /* [seq, OLMOE_N_EXPERTS_PER_TOK]     */
    olmoe_act_t *topk_w;        /* [seq, OLMOE_N_EXPERTS_PER_TOK]     */
    olmoe_act_t *expert_in;     /* [seq, OLMOE_INTER]  chosen-expert in */
    olmoe_act_t *expert_out;    /* [seq, OLMOE_HIDDEN] mlp accumulate  */
    olmoe_act_t *logits;        /* [seq, OLMOE_VOCAB]  caller-visible  */
    size_t       seq_len;       /* capacity sized by init             */
} olmoe_scratch_t;

/* Size a scratch for `seq_len` tokens and fill all pointer fields.
 * Returns OLMOE_ERR_NULL if `s` is NULL, OLMOE_ERR_ALLOC on malloc failure.
 *
 * Example:
 *     olmoe_scratch_t s;
 *     if (olmoe_scratch_init(&s, 64) != OLMOE_OK) return 1;
 *     ... olmoe_forward(m, ids, 64, &s, s.logits) ...
 *     olmoe_scratch_free(&s);
 */
olmoe_status_t olmoe_scratch_init(olmoe_scratch_t *s, size_t seq_len);

/* Release every buffer in `s` and zero the struct. NULL-safe. */
void olmoe_scratch_free(olmoe_scratch_t *s);

/* ── One forward function per tensor kind (olmoe_slot_kind_t) ──────────────
 * Each op reads BF16 weights from a loaded `olmoe_model_t` and reads/writes
 * caller-owned `olmoe_act_t` buffers. These are pure compute: they trust the
 * integrator (olmoe_forward) to have validated args and seq_len, and return
 * void. Real compute lands in a later PR. */

/* OLMOE_KIND_EMBED: token-id lookup into embed_tokens->[vocab, hidden]. */
void olmoe_embed_forward(const olmoe_model_t *m,
                         const int *token_ids, size_t seq_len,
                         olmoe_act_t *hidden_out);

/* OLMOE_KIND_LM_HEAD: final classifier matmul -> logits[seq, vocab]. */
void olmoe_lm_head_forward(const olmoe_model_t *m,
                           const olmoe_act_t *x, size_t seq_len,
                           olmoe_act_t *logits_out);

/* OLMOE_KIND_NORM: final RMSNorm before the LM head. */
void olmoe_final_norm_forward(const olmoe_bf16_t *w,
                               const olmoe_act_t *x, size_t seq_len,
                               olmoe_act_t *out);

/* OLMOE_KIND_INPUT_LN: RMSNorm at the start of each layer. */
void olmoe_input_ln_forward(const olmoe_bf16_t *w,
                            const olmoe_act_t *x, size_t seq_len,
                            olmoe_act_t *out);

/* OLMOE_KIND_POST_LN: RMSNorm after attention, before MLP. */
void olmoe_post_ln_forward(const olmoe_bf16_t *w,
                           const olmoe_act_t *x, size_t seq_len,
                           olmoe_act_t *out);

/* OLMOE_KIND_Q_PROJ: x @ q_proj^T -> [seq, hidden]. */
void olmoe_q_proj_forward(const olmoe_self_attn_t *a,
                          const olmoe_act_t *x, size_t seq_len,
                          olmoe_act_t *out);

/* OLMOE_KIND_K_PROJ: x @ k_proj^T -> [seq, hidden]. */
void olmoe_k_proj_forward(const olmoe_self_attn_t *a,
                          const olmoe_act_t *x, size_t seq_len,
                          olmoe_act_t *out);

/* OLMOE_KIND_V_PROJ: x @ v_proj^T -> [seq, hidden]. */
void olmoe_v_proj_forward(const olmoe_self_attn_t *a,
                          const olmoe_act_t *x, size_t seq_len,
                          olmoe_act_t *out);

/* OLMOE_KIND_O_PROJ: attn output @ o_proj^T -> [seq, hidden]. */
void olmoe_o_proj_forward(const olmoe_self_attn_t *a,
                          const olmoe_act_t *x, size_t seq_len,
                          olmoe_act_t *out);

/* OLMOE_KIND_Q_NORM: RMSNorm over the q vector (applied in-place). */
void olmoe_q_norm_forward(const olmoe_bf16_t *w,
                          olmoe_act_t *q, size_t seq_len);

/* OLMOE_KIND_K_NORM: RMSNorm over the k vector (applied in-place). */
void olmoe_k_norm_forward(const olmoe_bf16_t *w,
                          olmoe_act_t *k, size_t seq_len);

/* OLMOE_KIND_MLP_GATE: router logits -> top-K experts per token.
 * Fills topk_idx[seq*K] and topk_w[seq*K]. */
void olmoe_mlp_gate_forward(const olmoe_bf16_t *w,
                            const olmoe_act_t *x, size_t seq_len,
                            int *topk_idx, olmoe_act_t *topk_w);

/* OLMOE_KIND_EXPERT_GATE: x @ expert.gate_proj^T -> [n_tokens, inter]. */
void olmoe_expert_gate_forward(const olmoe_expert_t *e,
                               const olmoe_act_t *x, size_t n_tokens,
                               olmoe_act_t *out);

/* OLMOE_KIND_EXPERT_UP: x @ expert.up_proj^T -> [n_tokens, inter]. */
void olmoe_expert_up_forward(const olmoe_expert_t *e,
                             const olmoe_act_t *x, size_t n_tokens,
                             olmoe_act_t *out);

/* OLMOE_KIND_EXPERT_DOWN: x @ expert.down_proj^T -> [n_tokens, hidden]. */
void olmoe_expert_down_forward(const olmoe_expert_t *e,
                               const olmoe_act_t *x, size_t n_tokens,
                               olmoe_act_t *out);

/* Top-level orchestrator. Wires the per-kind ops above in the OLMoE forward
 * order using `scratch` for intermediate storage. Stubs: returns OLMOE_OK
 * after NULL checks. Lifetime: `m` and `scratch` are borrowed for the call;
 * `logits_out` is caller-owned (commonly `scratch->logits`).
 *
 * Example:
 *     olmoe_scratch_t s;
 *     olmoe_scratch_init(&s, seq_len);
 *     olmoe_forward(m, ids, seq_len, &s, s.logits);
 *     olmoe_scratch_free(&s);
 */
olmoe_status_t olmoe_forward(const olmoe_model_t *m, const int *token_ids,
                             size_t seq_len, olmoe_scratch_t *scratch,
                             olmoe_act_t *logits_out);

#endif /* OLMOE_ENGINE_H */
