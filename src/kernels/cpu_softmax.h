#ifndef KERNELS_CPU_SOFTMAX_H
#define KERNELS_CPU_SOFTMAX_H

#include <immintrin.h>
#include <math.h>

#include "kernels/kernels.h"

/* Find the maximum value in `input` of length `dim` (numerical-stability
 * helper for softmax). Uses a tree-structured reduce: 16-wide AVX-512
 * chunks are reduced to scalars, then those scalars are reduced
 * iteratively until one value remains. */
static inline float cpu_softmax_row_max(const float *input, size_t dim)
{
    const float *cur_input = input;
    size_t cur_dim = dim;
    float maxes[dim / 16 + 1];

    do {
        size_t nmax = 0;
        size_t r = 0;
        for (; r + 16 <= cur_dim; r += 16) {
            __m512 v = _mm512_loadu_ps(cur_input + r);
            maxes[nmax++] = _mm512_reduce_max_ps(v);
        }
        if (r < cur_dim) {
            __mmask16 mask = (1ULL << (cur_dim - r)) - 1;
            __m512 v = _mm512_mask_loadu_ps(_mm512_set1_ps(-INFINITY), mask, cur_input + r);
            maxes[nmax++] = _mm512_reduce_max_ps(v);
        }
        cur_input = maxes;
        cur_dim = nmax;
    } while (cur_dim > 1);

    return maxes[0];
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
