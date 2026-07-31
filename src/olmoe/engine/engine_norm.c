/* RMSNorm family (1:1 with OLMOE_KIND_NORM / INPUT_LN / POST_LN / Q_NORM /
 * K_NORM). The shared AVX512 core lives in kernels/cpu_rmsnorm.h; see
 * docs/engine_module.md. */

#include "olmoe/engine/engine.h"
#include "kernels/cpu_rmsnorm.h"

/* RMSNorm epsilon. config.json does not carry rms_norm_eps, so this matches
 * the HF OlmEConfig default (1e-5). Shared by every norm kind in this TU. */
static const float INPUT_LN_EPS = 1e-5f;

void olmoe_final_norm_forward(const olmoe_bf16_t * restrict w,
                                const olmoe_act_t * restrict x, size_t seq_len,
                                olmoe_act_t * restrict out)
{
    /* Model-final RMSNorm before the LM head; same kernel/eps as input_ln. */
    cpu_rmsnorm(out, x, w, seq_len, OLMOE_HIDDEN, INPUT_LN_EPS);
}

void olmoe_input_ln_forward(const olmoe_bf16_t * restrict w,
                             const olmoe_act_t * restrict x, size_t seq_len,
                             olmoe_act_t * restrict out)
{
    /* out = x * rsqrt(mean(x^2) + eps) * w, per OLMoE HF RMSNorm. Must be
     * out-of-place: `out` and `x` are `restrict`-qualified, so aliasing them
     * (in-place) is undefined. */
    cpu_rmsnorm(out, x, w, seq_len, OLMOE_HIDDEN, INPUT_LN_EPS);
}

void olmoe_post_ln_forward(const olmoe_bf16_t * restrict w,
                            const olmoe_act_t * restrict x, size_t seq_len,
                            olmoe_act_t * restrict out)
{
    /* Post-attention RMSNorm before the MLP; same kernel/eps as input_ln.
     * Must be out-of-place — see input_ln note. */
    cpu_rmsnorm(out, x, w, seq_len, OLMOE_HIDDEN, INPUT_LN_EPS);
}

void olmoe_q_norm_forward(const olmoe_bf16_t * restrict w,
                           olmoe_act_t * restrict q, size_t seq_len)
{
    /* OLMoE QK norm normalizes the flat hidden vector using the full 2048-dim
     * weight (HF: `self.q_norm(self.q_proj(hidden_states))` applied before the
     * head reshape, not per-head).  Identical to input_ln/post_ln. */
    cpu_rmsnorm(q, q, w, seq_len, OLMOE_HIDDEN, INPUT_LN_EPS);
}

void olmoe_k_norm_forward(const olmoe_bf16_t * restrict w,
                           olmoe_act_t * restrict k, size_t seq_len)
{
    cpu_rmsnorm(k, k, w, seq_len, OLMOE_HIDDEN, INPUT_LN_EPS);
}
