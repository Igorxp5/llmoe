#ifndef OLMOE_ENGINE_INTERNAL_H
#define OLMOE_ENGINE_INTERNAL_H

/* Shared internal helpers for the engine sub-modules. Not part of the
 * public API; only `engine_*.c` files under src/olmoe/engine include this.
 * Public callers use engine.h. */

#include <immintrin.h>

#include "olmoe/engine/engine.h"

/* Scoped AVX512 enablement for the inline helper below and any engine TU
 * that includes this header. All 15 engine ops live in their own .c under
 * src/olmoe/engine/, so only those TUs get the feature; nothing else is
 * affected. */
#pragma GCC target("avx512f,avx512bw,avx512vl,avx512bf16")

/* Load 16 BF16 (uint16) lanes and promote to 16 FP32 lanes using the
 * dedicated AVX512-BF16 conversion intrinsic. _mm512_cvtpbh_ps takes a
 * __m256i of packed BF16 and emits a __m512 of FP32. Used by both the
 * RMSNorm ops (input_ln) and the embedding row-gather. */
static inline __m512 olmoe_engine_bf16x16_to_fp32(const olmoe_bf16_t *p)
{
    __m256i src = _mm256_loadu_si256((const __m256i *)p);
    return _mm512_cvtpbh_ps((__m256bh)src);
}

/* Compute n * elemsz and require the result to fit in size_t. Returns 0 on
 * overflow. Used by olmoe_scratch_init before each malloc. Header-inline so
 * the engine sub-modules share one definition without a util .c. */
static inline size_t olmoe_engine_safe_array_size(size_t n, size_t elemsz)
{
    if (n == 0 || elemsz == 0) {
        return 0;
    }
    if (n > (size_t)-1 / elemsz) {
        return 0;
    }
    return n * elemsz;
}

#endif /* OLMOE_ENGINE_INTERNAL_H */
