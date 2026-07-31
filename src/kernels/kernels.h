#ifndef KERNELS_H
#define KERNELS_H

#include <immintrin.h>
#include <stdint.h>

#pragma GCC target("avx512f,avx512bw,avx512vl,avx512bf16")

/* Load 16 BF16 values from bf16_block and convert to one AVX-512 FP32 vector.
 * Used as the building-block for BF16 matmul and RMSNorm kernels. */
static inline __attribute__((always_inline)) __m512 kernels_bf16x16_to_fp32(const uint16_t * restrict bf16_block)
{
    __m256i src = _mm256_loadu_si256((const __m256i *)bf16_block);
    return _mm512_cvtpbh_ps((__m256bh)src);
}

/* FP32 dot product of two vectors of length `dim`: 16-lane FMA chunks reduced
 * with a tree reduce. A masked tail load zero-fills the partial remainder so
 * it does not pollute the reduce (same pattern as cpu_softmax_row_max). */
static inline __attribute__((always_inline)) float kernels_f32_dot(const float * restrict a,
                                        const float * restrict b, size_t dim)
{
    __m512 acc = _mm512_setzero_ps();
    size_t l = 0;
    for (; l + 16 <= dim; l += 16)
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + l), _mm512_loadu_ps(b + l), acc);
    if (__builtin_expect(l < dim, 0)) {
        __mmask16 mask = (1ULL << (dim - l)) - 1;
        __m512 va = _mm512_mask_loadu_ps(_mm512_setzero_ps(), mask, a + l);
        __m512 vb = _mm512_mask_loadu_ps(_mm512_setzero_ps(), mask, b + l);
        acc = _mm512_fmadd_ps(va, vb, acc);
    }
    return _mm512_reduce_add_ps(acc);
}

/* Zero `dim` floats with 512-bit stores; a masked store handles the tail. */
static inline __attribute__((always_inline)) void kernels_f32_zero(float * restrict out, size_t dim)
{
    __m512 z = _mm512_setzero_ps();
    size_t l = 0;
    for (; l + 16 <= dim; l += 16)
        _mm512_storeu_ps(out + l, z);
    if (__builtin_expect(l < dim, 0)) {
        __mmask16 mask = (1ULL << (dim - l)) - 1;
        _mm512_mask_storeu_ps(out + l, mask, z);
    }
}

/* out[i] += w * in[i] for i in [0, dim), 512-bit FMA with a masked tail. */
static inline __attribute__((always_inline)) void kernels_f32_axpy(float * restrict out,
                                        const float * restrict in, float w, size_t dim)
{
    __m512 vw = _mm512_set1_ps(w);
    size_t l = 0;
    for (; l + 16 <= dim; l += 16) {
        __m512 vo = _mm512_loadu_ps(out + l);
        vo = _mm512_fmadd_ps(vw, _mm512_loadu_ps(in + l), vo);
        _mm512_storeu_ps(out + l, vo);
    }
    if (__builtin_expect(l < dim, 0)) {
        __mmask16 mask = (1ULL << (dim - l)) - 1;
        __m512 vo = _mm512_mask_loadu_ps(_mm512_setzero_ps(), mask, out + l);
        __m512 vi = _mm512_mask_loadu_ps(_mm512_setzero_ps(), mask, in + l);
        _mm512_mask_storeu_ps(out + l, mask, _mm512_fmadd_ps(vw, vi, vo));
    }
}

#endif /* KERNELS_H */
