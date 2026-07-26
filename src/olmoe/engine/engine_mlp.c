/* MoE router + expert ops (1:1 with OLMOE_KIND_MLP_GATE /
 * EXPERT_GATE / EXPERT_UP / EXPERT_DOWN). Stubs only; see
 * docs/engine_module.md. The silu gating (gate * up) and top-K routing
 * glue are future ops in this file. */

#include "olmoe/engine/engine.h"

void olmoe_mlp_gate_forward(const olmoe_bf16_t *w,
                            const olmoe_act_t *x, size_t seq_len,
                            int *topk_idx, olmoe_act_t *topk_w)
{
    /* TODO: router_logits = x @ w^T; softmax; top-K of K=OLMOE_N_EXPERTS. */
}

void olmoe_expert_gate_forward(const olmoe_expert_t *e,
                               const olmoe_act_t *x, size_t n_tokens,
                               olmoe_act_t *out)
{
    /* TODO: x @ gate_proj^T -> out[n_tokens, inter]. */
}

void olmoe_expert_up_forward(const olmoe_expert_t *e,
                             const olmoe_act_t *x, size_t n_tokens,
                             olmoe_act_t *out)
{
    /* TODO: x @ up_proj^T -> out[n_tokens, inter]. */
}

void olmoe_expert_down_forward(const olmoe_expert_t *e,
                               const olmoe_act_t *x, size_t n_tokens,
                               olmoe_act_t *out)
{
    /* TODO: x @ down_proj^T -> out[n_tokens, hidden]. */
}