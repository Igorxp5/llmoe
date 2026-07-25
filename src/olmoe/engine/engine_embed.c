/* Embedding + LM head ops (1:1 with OLMOE_KIND_EMBED / OLMOE_KIND_LM_HEAD).
 * Stubs only; see docs/engine_module.md. */

#include "olmoe/engine/engine.h"

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
    /* TODO: lookup token_ids[i] -> row in embed_tokens[hidden]. */
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