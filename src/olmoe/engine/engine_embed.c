/* Embedding + LM head ops (1:1 with OLMOE_KIND_EMBED / OLMOE_KIND_LM_HEAD).
 * embed_forward is real; lm_head_forward reduces x @ lm_head^T to the shared
 * AVX512-BF16 matmul kernel (kernels/cpu_matmul.h). */

#include <stddef.h>

#include "olmoe/engine/engine.h"
#include "kernels/kernels.h"
#include "kernels/cpu_matmul.h"

void olmoe_embed_forward(const olmoe_model_t * restrict m,
                         const int * restrict token_ids, size_t seq_len,
                         olmoe_act_t * restrict hidden_out)
{
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < seq_len; ++i) {
        const olmoe_bf16_t *row = m->embed_tokens + (size_t)token_ids[i] * OLMOE_HIDDEN;
        olmoe_act_t *out = hidden_out + i * OLMOE_HIDDEN;
        for (size_t k = 0; k < OLMOE_HIDDEN; k += 16) {
            _mm512_storeu_ps(out + k, kernels_bf16x16_to_fp32(row + k));
        }
    }
}

void olmoe_lm_head_forward(const olmoe_model_t * restrict m,
                           const olmoe_act_t * restrict x, size_t seq_len,
                           olmoe_act_t * restrict logits_out)
{
    cpu_matmul_bf16(logits_out + ((seq_len - 1) * OLMOE_VOCAB), x + ((seq_len - 1) * OLMOE_HIDDEN), m->lm_head, 1, OLMOE_VOCAB, OLMOE_HIDDEN);
}
