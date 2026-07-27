#ifndef KERNELS_CPU_SOFTMAX_H
#define KERNELS_CPU_SOFTMAX_H

#include <math.h>

#include "kernels/kernels.h"

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

#endif /* KERNELS_CPU_SOFTMAX_H */
