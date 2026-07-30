#ifndef KERNELS_CPU_SOFTMAX_H
#define KERNELS_CPU_SOFTMAX_H

#include <math.h>

#include "kernels/kernels.h"

/* Find the maximum value in `input` of length `dim` (numerical-stability
 * helper for softmax). */
static inline float cpu_softmax_row_max(const float *input, size_t dim)
{
    float mx = input[0];
    for (size_t r = 1; r < dim; ++r)
        if (input[r] > mx) mx = input[r];
    return mx;
}

/* Stable softmax: output[i] = exp(input[i] - max) / sum(exp(input - max)).
 * Both vectors have length `dim`. */
static inline void cpu_softmax(float *output, const float *input, size_t dim)
{
    float mx = cpu_softmax_row_max(input, dim);
    float sum = 0.0f;
    for (size_t r = 0; r < dim; ++r) {
        output[r] = expf(input[r] - mx);
        sum += output[r];
    }
    for (size_t r = 0; r < dim; ++r)
        output[r] /= sum;
}

#endif /* KERNELS_CPU_SOFTMAX_H */
