#include <math.h>
#include <stdio.h>
#include <string.h>

#include "olmoe/engine/engine.h"

#include "test_engine_helpers.h"

/* Pure-C scalar RMSNorm using the identical BF16->FP32 weight promotion as
 * the SIMD impl; the only divergence expected is FP32 round-off. */
static void scalar_input_ln(const olmoe_bf16_t *w, const olmoe_act_t *x,
                            size_t seq_len, olmoe_act_t *out)
{
    const float eps = 1e-5f;
    for (size_t i = 0; i < seq_len; ++i) {
        const olmoe_act_t *xr = x + i * OLMOE_HIDDEN;
        olmoe_act_t *or_ = out + i * OLMOE_HIDDEN;
        float ss = 0.0f;
        for (size_t k = 0; k < OLMOE_HIDDEN; ++k) ss += xr[k] * xr[k];
        float scale = 1.0f / sqrtf(ss / (float)OLMOE_HIDDEN + eps);
        for (size_t k = 0; k < OLMOE_HIDDEN; ++k)
            or_[k] = xr[k] * scale * bf16_to_f32(w[k]);
    }
}

/* olmoe_input_ln_forward: deterministic + random rows compared against an
 * in-test pure-C scalar RMSNorm that promotes the same BF16 weights, so the
 * only divergence is FP32 SIMD-vs-scalar round-off. */
static int test_input_ln_matches_scalar(void)
{
    enum { ROWS = 3 };
    olmoe_bf16_t w[OLMOE_HIDDEN];
    olmoe_act_t x[ROWS * OLMOE_HIDDEN];
    olmoe_act_t got[ROWS * OLMOE_HIDDEN];
    olmoe_act_t want[ROWS * OLMOE_HIDDEN];

    /* Weights: ramp 1..hidden so every lane is distinct. */
    for (size_t k = 0; k < OLMOE_HIDDEN; ++k) w[k] = f32_to_bf16((float)(k + 1));

    /* Row 0: constant 1. Row 1: alternating +/-1. Row 2: pseudo-random
     * via a cheap LCG (no libc dependency). */
    for (size_t k = 0; k < OLMOE_HIDDEN; ++k) x[k] = 1.0f;
    for (size_t k = 0; k < OLMOE_HIDDEN; ++k) x[OLMOE_HIDDEN + k] = (k & 1) ? -1.0f : 1.0f;
    unsigned int rng = 0xdeadbeefu;
    for (size_t k = 0; k < OLMOE_HIDDEN; ++k) {
        rng = rng * 1664525u + 1013904223u;
        x[2 * OLMOE_HIDDEN + k] = (float)((int)rng % 1001) / 500.0f - 1.0f;
    }

    int failed = 0;
    olmoe_input_ln_forward(w, x, ROWS, got);
    scalar_input_ln(w, x, ROWS, want);

    /* SIMD and scalar differ only by FP32 round-off; rtol 1e-4, atol 1e-5. */
    for (size_t i = 0; i < ROWS * OLMOE_HIDDEN && !failed; ++i) {
        float a = got[i], b = want[i];
        float diff = fabsf(a - b);
        if (diff > 1e-5f && diff > 1e-4f * fabsf(b)) {
            printf("FAIL: input_ln lane %zu got=%.7f want=%.7f\n", i, a, b);
            ++failed;
        }
    }
    if (!failed) printf("PASS: input_ln_forward matches scalar RMSNorm\n");
    return failed;
}

/* Pure-C scalar flat RMSNorm reference (in-place safe when x == out).
 * Used for QK norm (flat, per OLMoE HF semantics) as well as
 * input_ln/post_ln/final_norm. */
static void scalar_rmsnorm_inplace(const olmoe_bf16_t *w,
                                    const olmoe_act_t *x, size_t seq_len,
                                    olmoe_act_t *out)
{
    const float eps = 1e-5f;
    for (size_t i = 0; i < seq_len; ++i) {
        const olmoe_act_t *xr = x + i * OLMOE_HIDDEN;
        olmoe_act_t *or_ = out + i * OLMOE_HIDDEN;
        float ss = 0.0f;
        for (size_t k = 0; k < OLMOE_HIDDEN; ++k) ss += xr[k] * xr[k];
        float scale = 1.0f / sqrtf(ss / (float)OLMOE_HIDDEN + eps);
        for (size_t k = 0; k < OLMOE_HIDDEN; ++k)
            or_[k] = xr[k] * scale * bf16_to_f32(w[k]);
    }
}

/* olmoe_q_norm_forward applies flat RMSNorm in-place over q[seq, hidden]
 * (HF semantics: self.q_norm(self.q_proj(...)) before the head reshape).
 * Same data-generation pattern as the other norm tests. */
static int test_q_norm_matches_scalar(void)
{
    enum { ROWS = 3 };
    olmoe_bf16_t w[OLMOE_HIDDEN];
    olmoe_act_t q[ROWS * OLMOE_HIDDEN];
    olmoe_act_t ref[ROWS * OLMOE_HIDDEN];

    for (size_t k = 0; k < OLMOE_HIDDEN; ++k) w[k] = f32_to_bf16((float)(k + 1));

    for (size_t k = 0; k < OLMOE_HIDDEN; ++k) q[k] = 1.0f;
    for (size_t k = 0; k < OLMOE_HIDDEN; ++k)
        q[OLMOE_HIDDEN + k] = (k & 1) ? -1.0f : 1.0f;
    unsigned int rng = 0xfeedfaceu;
    for (size_t k = 0; k < OLMOE_HIDDEN; ++k) {
        rng = rng * 1664525u + 1013904223u;
        q[2 * OLMOE_HIDDEN + k] = (float)((int)rng % 1001) / 500.0f - 1.0f;
    }

    memcpy(ref, q, sizeof q);
    olmoe_q_norm_forward(w, q, ROWS);
    scalar_rmsnorm_inplace(w, ref, ROWS, ref);

    int failed = lanes_match(q, ref, (size_t)ROWS * OLMOE_HIDDEN);
    if (failed) printf("FAIL: q_norm matches scalar RMSNorm\n");
    else printf("PASS: q_norm matches scalar RMSNorm\n");
    return failed;
}

/* olmoe_k_norm_forward: in-place flat RMSNorm over k, same data pattern. */
static int test_k_norm_matches_scalar(void)
{
    enum { ROWS = 3 };
    olmoe_bf16_t w[OLMOE_HIDDEN];
    olmoe_act_t k[ROWS * OLMOE_HIDDEN];
    olmoe_act_t ref[ROWS * OLMOE_HIDDEN];

    for (size_t i = 0; i < OLMOE_HIDDEN; ++i) w[i] = f32_to_bf16((float)(i + 1));
    for (size_t i = 0; i < OLMOE_HIDDEN; ++i) k[i] = 1.0f;
    for (size_t i = 0; i < OLMOE_HIDDEN; ++i)
        k[OLMOE_HIDDEN + i] = (i & 1) ? -1.0f : 1.0f;
    unsigned int rng = 0xbadcafeu;
    for (size_t i = 0; i < OLMOE_HIDDEN; ++i) {
        rng = rng * 1664525u + 1013904223u;
        k[2 * OLMOE_HIDDEN + i] = (float)((int)rng % 1001) / 500.0f - 1.0f;
    }

    memcpy(ref, k, sizeof k);
    olmoe_k_norm_forward(w, k, ROWS);
    scalar_rmsnorm_inplace(w, ref, ROWS, ref);

    int failed = lanes_match(k, ref, (size_t)ROWS * OLMOE_HIDDEN);
    if (failed) printf("FAIL: k_norm matches scalar RMSNorm\n");
    else printf("PASS: k_norm matches scalar RMSNorm\n");
    return failed;
}

/* olmoe_final_norm_forward: out-of-place RMSNorm before the LM head,
 * identical to input_ln. Reuses scalar_input_ln (same RMSNorm formula) and
 * lanes_match for the rtol 1e-4 / atol 1e-5 comparison. */
static int test_final_norm_matches_scalar(void)
{
    enum { ROWS = 3 };
    olmoe_bf16_t w[OLMOE_HIDDEN];
    olmoe_act_t x[ROWS * OLMOE_HIDDEN];
    olmoe_act_t got[ROWS * OLMOE_HIDDEN];
    olmoe_act_t want[ROWS * OLMOE_HIDDEN];

    for (size_t k = 0; k < OLMOE_HIDDEN; ++k) w[k] = f32_to_bf16((float)(k + 1));

    for (size_t k = 0; k < OLMOE_HIDDEN; ++k) x[k] = 1.0f;
    for (size_t k = 0; k < OLMOE_HIDDEN; ++k)
        x[OLMOE_HIDDEN + k] = (k & 1) ? -1.0f : 1.0f;
    unsigned int rng = 0xcafef00du;
    for (size_t k = 0; k < OLMOE_HIDDEN; ++k) {
        rng = rng * 1664525u + 1013904223u;
        x[2 * OLMOE_HIDDEN + k] = (float)((int)rng % 1001) / 500.0f - 1.0f;
    }

    olmoe_final_norm_forward(w, x, ROWS, got);
    scalar_input_ln(w, x, ROWS, want);

    int failed = lanes_match(got, want, (size_t)ROWS * OLMOE_HIDDEN);
    if (failed) printf("FAIL: final_norm matches scalar RMSNorm\n");
    else printf("PASS: final_norm matches scalar RMSNorm\n");
    return failed;
}

/* olmoe_post_ln_forward: out-of-place RMSNorm after attention, before the
 * MLP. Identical op to final_norm/input_ln, so it reuses scalar_input_ln as
 * the scalar reference and lanes_match for the rtol 1e-4 / atol 1e-5 check. */
static int test_post_ln_matches_scalar(void)
{
    enum { ROWS = 3 };
    olmoe_bf16_t w[OLMOE_HIDDEN];
    olmoe_act_t x[ROWS * OLMOE_HIDDEN];
    olmoe_act_t got[ROWS * OLMOE_HIDDEN];
    olmoe_act_t want[ROWS * OLMOE_HIDDEN];

    for (size_t k = 0; k < OLMOE_HIDDEN; ++k) w[k] = f32_to_bf16((float)(k + 1));

    for (size_t k = 0; k < OLMOE_HIDDEN; ++k) x[k] = 1.0f;
    for (size_t k = 0; k < OLMOE_HIDDEN; ++k)
        x[OLMOE_HIDDEN + k] = (k & 1) ? -1.0f : 1.0f;
    unsigned int rng = 0xd0d0feedu;
    for (size_t k = 0; k < OLMOE_HIDDEN; ++k) {
        rng = rng * 1664525u + 1013904223u;
        x[2 * OLMOE_HIDDEN + k] = (float)((int)rng % 1001) / 500.0f - 1.0f;
    }

    olmoe_post_ln_forward(w, x, ROWS, got);
    scalar_input_ln(w, x, ROWS, want);

    int failed = lanes_match(got, want, (size_t)ROWS * OLMOE_HIDDEN);
    if (failed) printf("FAIL: post_ln matches scalar RMSNorm\n");
    else printf("PASS: post_ln matches scalar RMSNorm\n");
    return failed;
}

int test_engine_norm_pass(void)
{
    int failed = 0;
    failed += test_input_ln_matches_scalar();
    failed += test_q_norm_matches_scalar();
    failed += test_k_norm_matches_scalar();
    failed += test_final_norm_matches_scalar();
    failed += test_post_ln_matches_scalar();
    return failed;
}
