#ifndef OLMOE_TEST_ENGINE_HELPERS_H
#define OLMOE_TEST_ENGINE_HELPERS_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "olmoe/engine/engine.h"

/* File-local BF16 utilities, used only by test_input_ln_matches_scalar so
 * both the test's weights and its scalar reference share the same BF16
 * rounding. Round-to-nearest-even, matching the on-disk format. */
static inline olmoe_bf16_t f32_to_bf16(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    uint32_t lsb = (u >> 16) & 1;
    uint32_t rounding_bias = 0x7fff + lsb;
    uint32_t rounded = u + rounding_bias;
    return (olmoe_bf16_t)(rounded >> 16);
}

static inline float bf16_to_f32(olmoe_bf16_t b)
{
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

/* Pure-C scalar matmul reference: same BF16->FP32 weight promotion as the
 * SIMD kernel, so the only divergence is FP32 accumulation order. Shared by
 * the matmul and mlp test modules to avoid a duplicate reference copy. */
static inline float scalar_dot_bf16(const olmoe_act_t *a,
                                     const olmoe_bf16_t *w, size_t k)
{
    float s = 0.0f;
    for (size_t l = 0; l < k; ++l)
        s += a[l] * bf16_to_f32(w[l]);
    return s;
}

static inline void scalar_matmul_bf16(olmoe_act_t *out, const olmoe_act_t *a,
                                      const olmoe_bf16_t *w,
                                      size_t m, size_t n, size_t k)
{
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            out[i * n + j] = scalar_dot_bf16(a + i * k, w + j * k, k);
}

/* rtol 1e-4, atol 1e-5 — matches the kernel-vs-scalar FP32 round-off budget. */
static inline int lanes_match(const olmoe_act_t *got, const olmoe_act_t *want,
                              size_t n)
{
    int failed = 0;
    for (size_t i = 0; i < n; ++i) {
        float d = fabsf(got[i] - want[i]);
        if (d > 1e-5f && d > 1e-4f * fabsf(want[i])) {
            printf("FAIL: lane %zu got=%.7f want=%.7f\n", i, got[i], want[i]);
            ++failed;
        }
    }
    return failed;
}

#endif /* OLMOE_TEST_ENGINE_HELPERS_H */