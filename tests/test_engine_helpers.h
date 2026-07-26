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