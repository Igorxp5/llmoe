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

#endif /* KERNELS_H */
