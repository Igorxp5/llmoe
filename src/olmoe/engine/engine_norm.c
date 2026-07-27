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
    /* In-place: each row's read pass completes before its write pass. */
    cpu_rmsnorm(q, q, w, seq_len, OLMOE_HIDDEN, INPUT_LN_EPS);
}

void olmoe_k_norm_forward(const olmoe_bf16_t *w,
                           olmoe_act_t *k, size_t seq_len)
{
    /* In-place over k; same kernel as q_norm (q/k norms are identical ops). */
    cpu_rmsnorm(k, k, w, seq_len, OLMOE_HIDDEN, INPUT_LN_EPS);
}
