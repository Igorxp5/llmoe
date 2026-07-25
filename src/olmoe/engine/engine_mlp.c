/* MoE router + expert ops (1:1 with OLMOE_KIND_MLP_GATE /
 * EXPERT_GATE / EXPERT_UP / EXPERT_DOWN). Stubs only; see
 * docs/engine_module.md. The silu gating (gate * up) and top-K routing
 * glue are future ops in this file. */

#include "olmoe/engine/engine.h"

olmoe_status_t olmoe_mlp_gate_forward(const olmoe_bf16_t *w,
                                      const olmoe_act_t *x, size_t seq_len,
                                      int *topk_idx, olmoe_act_t *topk_w)
{
    if (seq_len == 0) {
        return OLMOE_OK;
    }
    if (!w || !x || !topk_idx || !topk_w) {
        return OLMOE_ERR_NULL;
    }
    /* TODO: router_logits = x @ w^T; softmax; top-K of K=OLMOE_N_EXPERTS. */
    return OLMOE_OK;
}

olmoe_status_t olmoe_expert_gate_forward(const olmoe_expert_t *e,
                                         const olmoe_act_t *x, size_t n_tokens,
                                         olmoe_act_t *out)
{
    if (n_tokens == 0) {
        return OLMOE_OK;
    }
    if (!e || !e->gate_proj || !x || !out) {
        return OLMOE_ERR_NULL;
    }
    /* TODO: x @ gate_proj^T -> out[n_tokens, inter]. */
    return OLMOE_OK;
}

olmoe_status_t olmoe_expert_up_forward(const olmoe_expert_t *e,
                                       const olmoe_act_t *x, size_t n_tokens,
                                       olmoe_act_t *out)
{
    if (n_tokens == 0) {
        return OLMOE_OK;
    }
    if (!e || !e->up_proj || !x || !out) {
        return OLMOE_ERR_NULL;
    }
    /* TODO: x @ up_proj^T -> out[n_tokens, inter]. */
    return OLMOE_OK;
}

olmoe_status_t olmoe_expert_down_forward(const olmoe_expert_t *e,
                                         const olmoe_act_t *x, size_t n_tokens,
                                         olmoe_act_t *out)
{
    if (n_tokens == 0) {
        return OLMOE_OK;
    }
    if (!e || !e->down_proj || !x || !out) {
        return OLMOE_ERR_NULL;
    }
    /* TODO: x @ down_proj^T -> out[n_tokens, hidden]. */
    return OLMOE_OK;
}