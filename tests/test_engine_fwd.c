#include <math.h>
#include <stdio.h>
#include <string.h>

#include "olmoe/engine/engine.h"
#include "kernels/cpu_rope.h"
#include "kernels/cpu_sdpa.h"
#include "kernels/cpu_silu.h"

#include "test_engine_helpers.h"

/* ===================== kernel unit tests ================================ */

static int test_silu_matches_scalar(void)
{
    float pts[] = {-3.0f,-1.5f,-0.5f,0.0f,0.25f,0.5f,1.0f,2.0f,3.5f,5.0f};
    int failed = 0;
    for (size_t i = 0; i < sizeof(pts)/sizeof(pts[0]); ++i) {
        float x = pts[i];
        float want = x / (1.0f + expf(-x));
        float got = cpu_silu(x);
        float d = fabsf(got - want);
        if (d > 1e-6f * fabsf(want) && d > 1e-7f) {
            printf("FAIL: silu(%g) got=%.8f want=%.8f\n", x, got, want);
            ++failed;
        }
    }
    if (!failed) printf("PASS: silu matches scalar\n");
    return failed;
}

/* Scalar HF rotate_half RoPE reference (out-of-place). */
static void scalar_rope(olmoe_act_t *out, const olmoe_act_t *x,
                        size_t seq, size_t nh, size_t hd, float theta)
{
    size_t h2 = hd / 2;
    for (size_t i = 0; i < seq; ++i)
        for (size_t hh = 0; hh < nh; ++hh) {
            const olmoe_act_t *src = x + i*nh*hd + hh*hd;
            olmoe_act_t *dst = out + i*nh*hd + hh*hd;
            for (size_t k = 0; k < h2; ++k) {
                float inv = 1.0f/powf(theta, (2.0f*(float)k)/(float)hd);
                float a = cosf((float)i*inv), b = sinf((float)i*inv);
                dst[k]      = src[k]*a - src[h2+k]*b;
                dst[h2+k]   = src[h2+k]*a + src[k]*b;
            }
        }
}

static int test_rope_matches_scalar(void)
{
    enum { SEQ=3, NH=2, HD=8 };
    size_t n = SEQ*NH*HD;
    olmoe_act_t q[n], saved[n], ref[n];
    for (size_t i = 0; i < n; ++i)
        q[i] = (float)(((int)(i*7 % 23)) - 11) / 13.0f;
    memcpy(saved, q, sizeof q);

    /* pre-norm per head for norm-preservation check */
    float pre_norm[NH*SEQ];
    for (size_t i = 0; i < SEQ; ++i)
        for (size_t hh = 0; hh < NH; ++hh) {
            float s = 0.0f;
            for (size_t d = 0; d < HD; ++d) { float v = q[i*NH*HD+hh*HD+d]; s += v*v; }
            pre_norm[hh*SEQ+i] = s;
        }

    cpu_rope(q, SEQ, 0, NH, HD, OLMOE_ROPE_THETA);
    scalar_rope(ref, saved, SEQ, NH, HD, OLMOE_ROPE_THETA);

    int failed = 0;
    for (size_t i = 0; i < n; ++i) {
        float d = fabsf(q[i] - ref[i]);
        if (d > 1e-5f && d > 1e-4f * fabsf(ref[i])) {
            printf("FAIL: rope lane %zu got=%.7f want=%.7f\n", i, q[i], ref[i]);
            ++failed;
        }
    }
    /* norm preservation: RoPE is a rotation per (head, pair), so per-head
     * sum of squares is invariant. */
    for (size_t i = 0; i < SEQ && !failed; ++i)
        for (size_t hh = 0; hh < NH && !failed; ++hh) {
            float s = 0.0f;
            for (size_t d = 0; d < HD; ++d) { float v = q[i*NH*HD+hh*HD+d]; s += v*v; }
            if (fabsf(s - pre_norm[hh*SEQ+i]) > 1e-3f * pre_norm[hh*SEQ+i]) {
                printf("FAIL: rope norm not preserved head %zu tok %zu\n", hh, i);
                ++failed;
            }
        }
    if (!failed) printf("PASS: rope matches scalar\n");
    return failed;
}

/* Scalar causal SDPA reference (out-of-place). */
static void scalar_sdpa(olmoe_act_t *out, const olmoe_act_t *q,
                        const olmoe_act_t *k, const olmoe_act_t *v,
                        size_t seq, size_t nh, size_t hd, float scale)
{
    size_t st = nh*hd;
    for (size_t hh = 0; hh < nh; ++hh)
        for (size_t i = 0; i < seq; ++i) {
            const olmoe_act_t *qi = q + i*st + hh*hd;
            float scores[256];
            float mx = -INFINITY;
            for (size_t j = 0; j <= i; ++j) {
                float dot = 0.0f;
                const olmoe_act_t *kj = k + j*st + hh*hd;
                for (size_t d = 0; d < hd; ++d) dot += qi[d]*kj[d];
                scores[j] = dot*scale;
                if (scores[j] > mx) mx = scores[j];
            }
            float sum = 0.0f;
            for (size_t j = 0; j <= i; ++j) { scores[j] = expf(scores[j]-mx); sum += scores[j]; }
            olmoe_act_t *oi = out + i*st + hh*hd;
            for (size_t d = 0; d < hd; ++d) oi[d] = 0.0f;
            for (size_t j = 0; j <= i; ++j) {
                float w = scores[j]/sum;
                const olmoe_act_t *vj = v + j*st + hh*hd;
                for (size_t d = 0; d < hd; ++d) oi[d] += w*vj[d];
            }
        }
}

static int test_sdpa_matches_scalar(void)
{
    enum { SEQ=4, NH=2, HD=4 };
    size_t n = SEQ*NH*HD;
    olmoe_act_t q[n], k[n], v[n], got[n], want[n];
    for (size_t i = 0; i < n; ++i) {
        q[i] = (float)(((int)(i*3 % 11)) - 5) / 7.0f;
        k[i] = (float)(((int)(i*5 % 13)) - 6) / 9.0f;
        v[i] = (float)(((int)(i*7 % 17)) - 8) / 11.0f;
    }
    float scale = 1.0f/sqrtf((float)HD);
    olmoe_act_t scores[NH * SEQ];
    cpu_sdpa(got, q, k, v, SEQ, NH, HD, scale, scores, SEQ);
    scalar_sdpa(want, q, k, v, SEQ, NH, HD, scale);

    int failed = 0;
    for (size_t i = 0; i < n; ++i) {
        float d = fabsf(got[i] - want[i]);
        if (d > 1e-5f && d > 1e-4f * fabsf(want[i])) {
            printf("FAIL: sdpa lane %zu got=%.7f want=%.7f\n", i, got[i], want[i]);
            ++failed;
        }
    }
    if (failed) return failed;

    /* causal property: perturb the LAST key (only query seq-1 attends to it)
     * and assert earlier-query outputs are unchanged. */
    olmoe_act_t kp[n], got2[n];
    memcpy(kp, k, sizeof k);
    for (size_t d = 0; d < NH*HD; ++d) kp[(SEQ-1)*NH*HD + d] += 123.456f;
    cpu_sdpa(got2, q, kp, v, SEQ, NH, HD, scale, scores, SEQ);
    for (size_t i = 0; i < SEQ-1 && !failed; ++i)
        for (size_t j = 0; j < NH*HD && !failed; ++j) {
            size_t p = i*NH*HD + j;
            float d = fabsf(got[p] - got2[p]);
            if (d > 1e-6f) {
                printf("FAIL: sdpa causal leak tok %zu lane %zu d=%g\n", i, j, d);
                ++failed;
            }
        }
    if (!failed) printf("PASS: sdpa matches scalar\n");
    return failed;
}

int test_engine_fwd_pass(void)
{
    int failed = 0;
    failed += test_silu_matches_scalar();
    failed += test_rope_matches_scalar();
    failed += test_sdpa_matches_scalar();
    return failed;
}