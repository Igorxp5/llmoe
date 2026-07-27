#ifndef KERNELS_H
#define KERNELS_H

#include <immintrin.h>
#include <stdint.h>

#pragma GCC target("avx512f,avx512bw,avx512vl,avx512bf16")

static inline __m512 kernels_bf16x16_to_fp32(const uint16_t *p)
{
    __m256i src = _mm256_loadu_si256((const __m256i *)p);
    return _mm512_cvtpbh_ps((__m256bh)src);
}

#endif /* KERNELS_H */
