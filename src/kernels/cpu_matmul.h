#ifndef KERNELS_CPU_MATMUL_H
#define KERNELS_CPU_MATMUL_H

#include "kernels/kernels.h"

/* Dot product of one FP32 row vector `a_row` and one BF16 column vector
 * `b_col`, each of length `dim_k`. Used as the inner loop of cpu_matmul_bf16. */
static inline float cpu_matmul_dot_bf16(const float *a_row,
                                        const uint16_t *b_col, size_t dim_k)
{
    __m512 acc = _mm512_setzero_ps();
    for (size_t l = 0; l < dim_k; l += 16)
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(a_row + l),
                              kernels_bf16x16_to_fp32(b_col + l), acc);
    return _mm512_reduce_add_ps(acc);
}

/* Full BF16 matrix multiplication: output[M][N] = a_matrix[M][K] @ b_matrix[N][K]^T.
 * a_matrix is FP32, b_matrix is BF16. The inner dimension `inner_dim_k` must be
 * a multiple of 16 (AVX-512 requirement). */
static inline void cpu_matmul_bf16(float *output, const float *a_matrix,
                                   const uint16_t *b_matrix,
                                   size_t rows_m, size_t cols_n,
                                   size_t inner_dim_k)
{
    #pragma omp parallel for schedule(static) collapse(2)
    for (size_t i = 0; i < rows_m; ++i)
        for (size_t j = 0; j < cols_n; ++j)
            output[i * cols_n + j] =
                cpu_matmul_dot_bf16(a_matrix + i * inner_dim_k,
                                    b_matrix + j * inner_dim_k, inner_dim_k);
}

#endif /* KERNELS_CPU_MATMUL_H */
