#ifndef KERNELS_CPU_RMSNORM_H
#define KERNELS_CPU_RMSNORM_H

#include <math.h>

#include "kernels/kernels.h"

/* Compute 1 / sqrt(mean(input_row^2) + epsilon) for an RMSNorm row.
 * `input_row` has length `dim`. The loop strides by 16 (AVX-512 register width). */
static inline float cpu_rmsnorm_scale(const float *input_row, size_t dim,
                                       float epsilon)
{
    __m512 acc = _mm512_setzero_ps();
    for (size_t k = 0; k < dim; k += 16) {
        __m512 v = _mm512_loadu_ps(input_row + k);
        acc = _mm512_add_ps(acc, _mm512_mul_ps(v, v));
    }
    float sum = _mm512_reduce_add_ps(acc);
    return 1.0f / sqrtf(sum / (float)dim + epsilon);
}

/* Apply the pre-computed RMSNorm scale factor: output = input * scale * weight */
static inline void cpu_rmsnorm_apply(float *output,
                                      const float *input,
                                      float scale_factor,
                                      const uint16_t *weight, size_t dim)
{
    __m512 vscale = _mm512_set1_ps(scale_factor);
    for (size_t k = 0; k < dim; k += 16) {
        __m512 vx = _mm512_loadu_ps(input + k);
        __m512 vw = kernels_bf16x16_to_fp32(weight + k);
        __m512 vout = _mm512_mul_ps(_mm512_mul_ps(vx, vscale), vw);
        _mm512_storeu_ps(output + k, vout);
    }
}

/* Full RMSNorm on one row: output = input / sqrt(mean(input^2) + epsilon) * weight */
static inline void cpu_rmsnorm_row(float *output, const float *input,
                                    const uint16_t *weight, size_t dim,
                                    float epsilon)
{
    float scale = cpu_rmsnorm_scale(input, dim, epsilon);
    cpu_rmsnorm_apply(output, input, scale, weight, dim);
}

/* Batched RMSNorm over `num_rows` vectors of length `dim`.
 * Each row i: output[i] = input[i] / sqrt(mean(input[i]^2) + epsilon) * weight */
static inline void cpu_rmsnorm(float *output, const float *input,
                                const uint16_t *weight, size_t num_rows,
                                size_t dim, float epsilon)
{
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num_rows; ++i) {
        size_t off = i * dim;
        cpu_rmsnorm_row(output + off, input + off, weight, dim, epsilon);
    }
}

#endif /* KERNELS_CPU_RMSNORM_H */
