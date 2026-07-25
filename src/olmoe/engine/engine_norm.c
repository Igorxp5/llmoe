/* RMSNorm family (1:1 with OLMOE_KIND_NORM / INPUT_LN / POST_LN / Q_NORM /
 * K_NORM). Stubs only; see docs/engine_module.md. */

#include "olmoe/engine/engine.h"

olmoe_status_t olmoe_final_norm_forward(const olmoe_bf16_t *w,
                                        const olmoe_act_t *x, size_t seq_len,
                                        olmoe_act_t *out)
{
    if (seq_len == 0) {
        return OLMOE_OK;
    }
    if (!w || !x || !out) {
        return OLMOE_ERR_NULL;
    }
    /* TODO: RMSNorm(x, w). */
    return OLMOE_OK;
}

olmoe_status_t olmoe_input_ln_forward(const olmoe_bf16_t *w,
                                      const olmoe_act_t *x, size_t seq_len,
                                      olmoe_act_t *out)
{
    if (seq_len == 0) {
        return OLMOE_OK;
    }
    if (!w || !x || !out) {
        return OLMOE_ERR_NULL;
    }
    /* TODO: RMSNorm(x, w). */
    return OLMOE_OK;
}

olmoe_status_t olmoe_post_ln_forward(const olmoe_bf16_t *w,
                                     const olmoe_act_t *x, size_t seq_len,
                                     olmoe_act_t *out)
{
    if (seq_len == 0) {
        return OLMOE_OK;
    }
    if (!w || !x || !out) {
        return OLMOE_ERR_NULL;
    }
    /* TODO: RMSNorm(x, w). */
    return OLMOE_OK;
}

olmoe_status_t olmoe_q_norm_forward(const olmoe_bf16_t *w,
                                    olmoe_act_t *q, size_t seq_len)
{
    if (seq_len == 0) {
        return OLMOE_OK;
    }
    if (!w || !q) {
        return OLMOE_ERR_NULL;
    }
    /* TODO: in-place RMSNorm over q. */
    return OLMOE_OK;
}

olmoe_status_t olmoe_k_norm_forward(const olmoe_bf16_t *w,
                                    olmoe_act_t *k, size_t seq_len)
{
    if (seq_len == 0) {
        return OLMOE_OK;
    }
    if (!w || !k) {
        return OLMOE_ERR_NULL;
    }
    /* TODO: in-place RMSNorm over k. */
    return OLMOE_OK;
}
