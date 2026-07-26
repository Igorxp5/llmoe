/* Embedding + LM head ops (1:1 with OLMOE_KIND_EMBED / OLMOE_KIND_LM_HEAD).
 * embed_forward is real; lm_head_forward is stubbed until the matmul PR. */

#include <stddef.h>

#include "olmoe/engine/engine.h"
#include "olmoe/engine/engine_internal.h"

olmoe_status_t olmoe_embed_forward(const olmoe_model_t *m,
                                   const int *token_ids, size_t seq_len,
                                   olmoe_act_t *hidden_out)
{
    if (seq_len == 0) {
        return OLMOE_OK;
    }
    if (!m || !m->embed_tokens || !token_ids || !hidden_out) {
        return OLMOE_ERR_NULL;
    }

    for (size_t i = 0; i < seq_len; ++i) {
        const olmoe_bf16_t *row = m->embed_tokens + (size_t)token_ids[i] * OLMOE_HIDDEN;
        olmoe_act_t *out = hidden_out + i * OLMOE_HIDDEN;
        for (size_t k = 0; k < OLMOE_HIDDEN; k += 16) {
            _mm512_storeu_ps(out + k, olmoe_engine_bf16x16_to_fp32(row + k));
        }
    }
    return OLMOE_OK;
}

olmoe_status_t olmoe_lm_head_forward(const olmoe_model_t *m,
                                     const olmoe_act_t *x, size_t seq_len,
                                     olmoe_act_t *logits_out)
{
    if (seq_len == 0) {
        return OLMOE_OK;
    }
    if (!m || !m->lm_head || !x || !logits_out) {
        return OLMOE_ERR_NULL;
    }
    /* TODO: x @ lm_head^T -> logits[seq, vocab]. */
    return OLMOE_OK;
}
