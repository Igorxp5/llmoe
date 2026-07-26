#ifndef OLMOE_KERNEL_CPU_SOFTMAX_H
#define OLMOE_KERNEL_CPU_SOFTMAX_H

#include <math.h>

#include "olmoe/engine/engine_internal.h"

/* Numerically stable softmax over a single row of n floats:
 * subtract the row max before expf so the largest exponent is 0.
 *
 * Example:
 *     float probs[OLMOE_N_EXPERTS];
 *     cpu_softmax(probs, logits_row, OLMOE_N_EXPERTS);
 */
static inline float cpu_softmax_row_max(const float *in, size_t n)
{
    float mx = in[0];
    for (size_t r = 1; r < n; ++r)
        if (in[r] > mx) mx = in[r];
    return mx;
}

static inline void cpu_softmax(float *out, const float *in, size_t n)
{
    float mx = cpu_softmax_row_max(in, n);
    float sum = 0.0f;
    for (size_t r = 0; r < n; ++r) {
        out[r] = expf(in[r] - mx);
        sum += out[r];
    }
    for (size_t r = 0; r < n; ++r)
        out[r] /= sum;
}

#endif /* OLMOE_KERNEL_CPU_SOFTMAX_H */