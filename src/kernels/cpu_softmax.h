#ifndef KERNELS_CPU_SOFTMAX_H
#define KERNELS_CPU_SOFTMAX_H

#include <immintrin.h>
#include <math.h>
#include <omp.h>

#include "kernels/kernels.h"

/* Find the maximum value in `input` of length `dim` (numerical-stability
 * helper for softmax). Uses a tree-structured reduce: 16-wide AVX-512
 * chunks are reduced to scalars, then those scalars are reduced
 * iteratively until one value remains. */
static inline __attribute__((always_inline)) float cpu_softmax_row_max(const float * restrict input,
                                        size_t dim)
{
    const float *cur_input = input;
    size_t cur_dim = dim;
    float maxes[dim / 16 + 1];
    maxes[0] = 0; // avoid "warning: ‘*maxes[0]’ may be used uninitialized"

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
static inline __attribute__((always_inline)) void cpu_softmax_row_norm(float * restrict output, size_t dim,
                                        float sum)
{
    __m512 vsum = _mm512_set1_ps(sum);
    size_t r = 0;
    for (; r + 16 <= dim; r += 16) {
        __m512 v = _mm512_loadu_ps(output + r);
        _mm512_storeu_ps(output + r, _mm512_div_ps(v, vsum));
    }
    for (; r < dim; ++r)
        output[r] /= sum;
}

/* Fixed-order AVX-512 sum over `n` floats: 16-lane chunks are accumulated
 * with _mm512_add_ps, reduced with _mm512_reduce_add_ps in a fixed order,
 * then the scalar tail is folded in ascending index. Folds either per-thread
 * partial sums (parallel path) or a full-dim exp row (serial path). */
static inline __attribute__((always_inline)) float cpu_softmax_vec_sum(
    const float * restrict vals, size_t n)
{
    size_t r = 0;
    __m512 vsum = _mm512_setzero_ps();
    for (; r + 16 <= n; r += 16)
        vsum = _mm512_add_ps(vsum, _mm512_loadu_ps(vals + r));
    float sum = _mm512_reduce_add_ps(vsum);
    if (__builtin_expect(r < n, 0))
        for (; r < n; ++r)
            sum += vals[r];
    return sum;
}

/* Serial exp + fixed-order sum for small dims: no fork/join, just a
 * streaming expf store folded by cpu_softmax_vec_sum. Router logits
 * (64 elements) take this path every layer. */
static inline float cpu_softmax_exp_sum_serial(float * restrict output,
                                               const float * restrict input,
                                               float mx, size_t dim)
{
    for (size_t r = 0; r < dim; ++r)
        output[r] = expf(input[r] - mx);
    return cpu_softmax_vec_sum(output, dim);
}

/* Multi-thread exp + partial-sum for large dims: each thread owns a static
 * contiguous chunk, its partial lands in partial_sums, folded in fixed
 * order by cpu_softmax_vec_sum. */
static inline float cpu_softmax_exp_sum_parallel(float * restrict output,
                                                 const float * restrict input,
                                                 float mx, size_t dim)
{
    int nthreads = omp_get_max_threads();
    float partial_sums[nthreads];
    for (int t = 0; t < nthreads; ++t)
        partial_sums[t] = 0.0f;
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        float partial = 0.0f;
        #pragma omp for schedule(static)
        for (size_t r = 0; r < dim; ++r) {
            output[r] = expf(input[r] - mx);
            partial += output[r];
        }
        partial_sums[tid] = partial;
    }
    return cpu_softmax_vec_sum(partial_sums, (size_t)nthreads);
}

/* Stable softmax with a small-dim fast path: a fork/join for the 64 router
 * logits costs more than the whole computation, so dims at or below
 * CPU_SOFTMAX_SERIAL_DIM use the serial AVX-512 path; larger dims fall back
 * to the multi-thread path. Both paths sum in a fixed order, so the result
 * is deterministic regardless of thread count. */
#define CPU_SOFTMAX_SERIAL_DIM 256

static inline void cpu_softmax(float * restrict output,
                               const float * restrict input, size_t dim)
{
    float mx = cpu_softmax_row_max(input, dim);
    float sum;
    if (__builtin_expect(dim <= CPU_SOFTMAX_SERIAL_DIM, 1))
        sum = cpu_softmax_exp_sum_serial(output, input, mx, dim);
    else
        sum = cpu_softmax_exp_sum_parallel(output, input, mx, dim);
    cpu_softmax_row_norm(output, dim, sum);
}

#endif /* KERNELS_CPU_SOFTMAX_H */
