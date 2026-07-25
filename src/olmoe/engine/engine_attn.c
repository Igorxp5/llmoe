/* Attention projections (1:1 with OLMOE_KIND_Q_PROJ / K_PROJ / V_PROJ /
 * O_PROJ). Stubs only; see docs/engine_module.md. The attention compute
 * itself (RoPE + sdpa) is a future op living in this file. */

#include "olmoe/engine/engine.h"

olmoe_status_t olmoe_q_proj_forward(const olmoe_self_attn_t *a,
                                    const olmoe_act_t *x, size_t seq_len,
                                    olmoe_act_t *out)
{
    if (seq_len == 0) {
        return OLMOE_OK;
    }
    if (!a || !a->q_proj || !x || !out) {
        return OLMOE_ERR_NULL;
    }
    /* TODO: x @ q_proj^T -> out[seq, hidden]. */
    return OLMOE_OK;
}

olmoe_status_t olmoe_k_proj_forward(const olmoe_self_attn_t *a,
                                    const olmoe_act_t *x, size_t seq_len,
                                    olmoe_act_t *out)
{
    if (seq_len == 0) {
        return OLMOE_OK;
    }
    if (!a || !a->k_proj || !x || !out) {
        return OLMOE_ERR_NULL;
    }
    /* TODO: x @ k_proj^T -> out[seq, hidden]. */
    return OLMOE_OK;
}

olmoe_status_t olmoe_v_proj_forward(const olmoe_self_attn_t *a,
                                    const olmoe_act_t *x, size_t seq_len,
                                    olmoe_act_t *out)
{
    if (seq_len == 0) {
        return OLMOE_OK;
    }
    if (!a || !a->v_proj || !x || !out) {
        return OLMOE_ERR_NULL;
    }
    /* TODO: x @ v_proj^T -> out[seq, hidden]. */
    return OLMOE_OK;
}

olmoe_status_t olmoe_o_proj_forward(const olmoe_self_attn_t *a,
                                    const olmoe_act_t *x, size_t seq_len,
                                    olmoe_act_t *out)
{
    if (seq_len == 0) {
        return OLMOE_OK;
    }
    if (!a || !a->o_proj || !x || !out) {
        return OLMOE_ERR_NULL;
    }
    /* TODO: attn_out @ o_proj^T -> out[seq, hidden]. */
    return OLMOE_OK;
}