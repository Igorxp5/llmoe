/* RMSNorm family (1:1 with OLMOE_KIND_NORM / INPUT_LN / POST_LN / Q_NORM /
 * K_NORM). Stubs only; see docs/engine_module.md. */

#include <immintrin.h>
#include <math.h>

#include "olmoe/engine/engine.h"

/* Scoped AVX512 enablement: only this TU gets the feature, so the global
 * CFLAGS baseline stays clean and the other 14 stubs/tests are untouched.
 * We use AVX512F/BW/VL lane math; avx512bf16 is enabled because the weights
 * are BF16 and the dedicated BF16 path (promotion via shift) relies on the
 * 16-bit lane granularity of BW. We deliberately do NOT use dpbf16_ps: it
 * quantizes FP32 activations and drifts from a canonical FP32 RMSNorm. */
#pragma GCC target("avx512f,avx512bw,avx512vl,avx512bf16")

/* RMSNorm epsilon. config.json does not carry rms_norm_eps, so this matches
 * the HF OlmEConfig default (1e-5). Local to this module until a second
 * norm kind goes real; promoting a shared constant then is one-line. */
static const float INPUT_LN_EPS = 1e-5f;

/* Load 16 BF16 (uint16) lanes and promote to 16 FP32 lanes using the
 * dedicated AVX512-BF16 conversion intrinsic. _mm512_cvtpbh_ps takes a
 * __m256i of packed BF16 and emits a __m512 of FP32. */
static inline __m512 bf16x16_to_fp32(const olmoe_bf16_t *p)
{
    __m256i src = _mm256_loadu_si256((const __m256i *)p);
    return _mm512_cvtpbh_ps((__m256bh)src);
}

olmoe_status_t olmoe_final_norm_forward(const olmoe_bf16_t *w,
                                        const olmoe_act_t *x, size_t seq_len,
                                        olmoe_act_t *out)
{
    if (seq_len == 0) {
        return OLMOE_OK;
    }
    if (!w || !x || !out) {
        return OLMOE_ERR_NULL;
    }
    /* TODO: RMSNorm(x, w). */
    return OLMOE_OK;
}

/* Sum of squares of one OLMOE_HIDDEN-length FP32 row, reduced to a scalar.
 * OLMOE_HIDDEN (2048) is a multiple of 16, so no tail. */
static float row_sum_of_squares(const olmoe_act_t *row)
{
    __m512 acc = _mm512_setzero_ps();
    for (size_t k = 0; k < OLMOE_HIDDEN; k += 16) {
        __m512 v = _mm512_loadu_ps(row + k);
        acc = _mm512_add_ps(acc, _mm512_mul_ps(v, v));
    }
    return _mm512_reduce_add_ps(acc);
}

/* Apply out = x * scale * w (elementwise) for one row; weights are BF16. */
static void row_scale_by_weight(const olmoe_act_t *x, float scale,
                                 const olmoe_bf16_t *w, olmoe_act_t *out)
{
    __m512 vscale = _mm512_set1_ps(scale);
    for (size_t k = 0; k < OLMOE_HIDDEN; k += 16) {
        __m512 vx = _mm512_loadu_ps(x + k);
        __m512 vw = bf16x16_to_fp32(w + k);
        __m512 vout = _mm512_mul_ps(_mm512_mul_ps(vx, vscale), vw);
        _mm512_storeu_ps(out + k, vout);
    }
}

olmoe_status_t olmoe_input_ln_forward(const olmoe_bf16_t *w,
                                      const olmoe_act_t *x, size_t seq_len,
                                      olmoe_act_t *out)
{
    if (seq_len == 0) {
        return OLMOE_OK;
    }
    if (!w || !x || !out) {
        return OLMOE_ERR_NULL;
    }

    /* out = x * rsqrt(mean(x^2) + eps) * w, per OLMoE HF RMSNorm. Two passes
     * per row: pass 1 reads x for the squared-mean, pass 2 reads x again and
     * writes out. Out-of-place is required; in-place (x == out) is also safe
     * because pass 1 finishes before pass 2 begins for the same row. */
    for (size_t i = 0; i < seq_len; ++i) {
        size_t off = i * OLMOE_HIDDEN;
        float ss = row_sum_of_squares(x + off);
        float mean = ss / (float)OLMOE_HIDDEN;
        float scale = 1.0f / sqrtf(mean + INPUT_LN_EPS);
        row_scale_by_weight(x + off, scale, w, out + off);
    }
    return OLMOE_OK;
}

olmoe_status_t olmoe_post_ln_forward(const olmoe_bf16_t *w,
                                     const olmoe_act_t *x, size_t seq_len,
                                     olmoe_act_t *out)
{
    if (seq_len == 0) {
        return OLMOE_OK;
    }
    if (!w || !x || !out) {
        return OLMOE_ERR_NULL;
    }
    /* TODO: RMSNorm(x, w). */
    return OLMOE_OK;
}

olmoe_status_t olmoe_q_norm_forward(const olmoe_bf16_t *w,
                                    olmoe_act_t *q, size_t seq_len)
{
    if (seq_len == 0) {
        return OLMOE_OK;
    }
    if (!w || !q) {
        return OLMOE_ERR_NULL;
    }
    /* TODO: in-place RMSNorm over q. */
    return OLMOE_OK;
}

olmoe_status_t olmoe_k_norm_forward(const olmoe_bf16_t *w,
                                    olmoe_act_t *k, size_t seq_len)
{
    if (seq_len == 0) {
        return OLMOE_OK;
    }
    if (!w || !k) {
        return OLMOE_ERR_NULL;
    }
    /* TODO: in-place RMSNorm over k. */
    return OLMOE_OK;
}
