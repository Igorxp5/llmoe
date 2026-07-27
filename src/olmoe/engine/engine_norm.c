/* RMSNorm family (1:1 with OLMOE_KIND_NORM / INPUT_LN / POST_LN / Q_NORM /
 * K_NORM). The shared AVX512 core lives in kernels/cpu_rmsnorm.h; see
 * docs/engine_module.md. */

#include "olmoe/engine/engine.h"
#include "kernels/cpu_rmsnorm.h"

/* RMSNorm epsilon. config.json does not carry rms_norm_eps, so this matches
 * the HF OlmEConfig default (1e-5). Shared by every norm kind in this TU. */
static const float INPUT_LN_EPS = 1e-5f;

void olmoe_final_norm_forward(const olmoe_bf16_t *w,
                                const olmoe_act_t *x, size_t seq_len,
                                olmoe_act_t *out)
{
    /* Model-final RMSNorm before the LM head; same kernel/eps as input_ln. */
    cpu_rmsnorm(out, x, w, seq_len, OLMOE_HIDDEN, INPUT_LN_EPS);
}

void olmoe_input_ln_forward(const olmoe_bf16_t *w,
                             const olmoe_act_t *x, size_t seq_len,
                             olmoe_act_t *out)
{
    /* out = x * rsqrt(mean(x^2) + eps) * w, per OLMoE HF RMSNorm. Out-of-place
     * required; in-place (x == out) is also safe (each row finishes its read
     * pass before its write pass). */
    cpu_rmsnorm(out, x, w, seq_len, OLMOE_HIDDEN, INPUT_LN_EPS);
}

void olmoe_post_ln_forward(const olmoe_bf16_t *w,
                            const olmoe_act_t *x, size_t seq_len,
                            olmoe_act_t *out)
{
    /* Post-attention RMSNorm before the MLP; same kernel/eps as input_ln,
     * applied out-of-place (see input_ln note on in-place safety). */
    cpu_rmsnorm(out, x, w, seq_len, OLMOE_HIDDEN, INPUT_LN_EPS);
}

void olmoe_q_norm_forward(const olmoe_bf16_t *w,
                           olmoe_act_t *q, size_t seq_len)
{
    /* OLMoE QK norm is per-head: each of the OLMOE_NUM_HEADS heads
     * (OLMOE_HEAD_DIM lanes) is independently RMS-normed using the
     * corresponding slice of the weight vector. The full-weight flat norm
     * that input_ln/post_ln use would cross-contaminate head statistics. */
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t h = 0; h < OLMOE_NUM_HEADS; ++h) {
            size_t off = i * OLMOE_HIDDEN + h * OLMOE_HEAD_DIM;
            cpu_rmsnorm_row(q + off, q + off,
                            w + h * OLMOE_HEAD_DIM,
                            OLMOE_HEAD_DIM, INPUT_LN_EPS);
        }
    }
}

void olmoe_k_norm_forward(const olmoe_bf16_t *w,
                           olmoe_act_t *k, size_t seq_len)
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t h = 0; h < OLMOE_NUM_HEADS; ++h) {
            size_t off = i * OLMOE_HIDDEN + h * OLMOE_HEAD_DIM;
            cpu_rmsnorm_row(k + off, k + off,
                            w + h * OLMOE_HEAD_DIM,
                            OLMOE_HEAD_DIM, INPUT_LN_EPS);
        }
    }
}
