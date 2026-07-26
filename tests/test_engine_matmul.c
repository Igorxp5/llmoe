#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "olmoe/engine/engine.h"
#include "olmoe/engine/kernels/cpu_matmul.h"

#include "test_engine_helpers.h"

/* Pure-C scalar matmul reference: same BF16->FP32 weight promotion as the
 * SIMD kernel, so the only divergence is FP32 accumulation order. */
static float scalar_dot_bf16(const olmoe_act_t *a, const olmoe_bf16_t *w, size_t k)
{
    float s = 0.0f;
    for (size_t l = 0; l < k; ++l)
        s += a[l] * bf16_to_f32(w[l]);
    return s;
}

static void scalar_matmul_bf16(olmoe_act_t *out, const olmoe_act_t *a,
                               const olmoe_bf16_t *w,
                               size_t m, size_t n, size_t k)
{
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            out[i * n + j] = scalar_dot_bf16(a + i * k, w + j * k, k);
}

/* Small-dim full check of cpu_matmul_bf16: every output lane compared. */
static int check_matmul_kernel_small_dim(void)
{
    enum { M = 2, N = 48, K = 64 };
    static olmoe_act_t a[M * K];
    static olmoe_bf16_t w[N * K];
    static olmoe_act_t got[M * N];
    static olmoe_act_t want[M * N];
    for (size_t i = 0; i < M * K; ++i)
        a[i] = (float)((int)(i % 17) - 8) / 8.0f;
    for (size_t j = 0; j < N * K; ++j)
        w[j] = f32_to_bf16((float)((int)(j % 13)) / 13.0f);
    cpu_matmul_bf16(got, a, w, M, N, K);
    scalar_matmul_bf16(want, a, w, M, N, K);
    return lanes_match(got, want, (size_t)M * N);
}

/* Fill the lm_head rows the smoke test later samples, so those rows carry
 * meaningful BF16 values rather than the zero memset background. */
static void fill_lm_head_sampled_rows(olmoe_bf16_t *lm)
{
    const size_t rows[] = {0, 1, 5, 7, 1000, OLMOE_VOCAB - 1};
    for (size_t r = 0; r < sizeof(rows) / sizeof(rows[0]); ++r) {
        olmoe_bf16_t *row = lm + rows[r] * OLMOE_HIDDEN;
        for (size_t l = 0; l < OLMOE_HIDDEN; ++l)
            row[l] = f32_to_bf16((float)(((int)(rows[r] + l) % 101) - 50) / 100.0f);
    }
}

static int verify_lm_head_sampled(const olmoe_bf16_t *lm,
                                 const olmoe_act_t *x,
                                 const olmoe_act_t *got)
{
    struct { size_t i, j; } pts[] = {
        {0, 0}, {0, 1}, {0, 7}, {0, 1000}, {0, (size_t)OLMOE_VOCAB - 1},
        {1, 5},
    };
    int failed = 0;
    for (size_t p = 0; p < sizeof(pts) / sizeof(pts[0]); ++p) {
        size_t i = pts[p].i, j = pts[p].j;
        float want = scalar_dot_bf16(x + i * OLMOE_HIDDEN,
                                     lm + j * OLMOE_HIDDEN, OLMOE_HIDDEN);
        float d = fabsf(got[i * OLMOE_VOCAB + j] - want);
        if (d > 1e-5f && d > 1e-4f * fabsf(want)) {
            printf("FAIL: lm_head[%zu,%zu] got=%.7f want=%.7f\n", i, j,
                   got[i * OLMOE_VOCAB + j], want);
            ++failed;
        }
    }
    return failed;
}

/* Real-dim smoke for olmoe_lm_head_forward only: full vocab*hidden lm_head
 * buffer but sampled-row verification (cheap; no full 50304-col scalar). */
static int check_lm_head_forward_sampled(void)
{
    static olmoe_model_t muc;
    memset(&muc, 0, sizeof muc);
    size_t lm_n = (size_t)OLMOE_VOCAB * OLMOE_HIDDEN;
    olmoe_bf16_t *lm = malloc(lm_n * sizeof *lm);
    if (!lm) { printf("FAIL: lm_head malloc OOM\n"); return 1; }
    memset(lm, 0, lm_n * sizeof *lm);
    fill_lm_head_sampled_rows(lm);
    muc.lm_head = lm;

    static olmoe_act_t x[2 * OLMOE_HIDDEN];
    for (size_t i = 0; i < 2 * OLMOE_HIDDEN; ++i)
        x[i] = (float)((int)(i % 7)) / 100.0f - 0.03f;

    static olmoe_act_t got[2 * OLMOE_VOCAB];
    olmoe_lm_head_forward(&muc, x, 2, got);

    int failed = verify_lm_head_sampled(lm, x, got);
    free(lm);
    return failed;
}

static int test_lm_head_matmul_matches_scalar(void)
{
    int failed = 0;
    failed += check_matmul_kernel_small_dim();
    failed += check_lm_head_forward_sampled();
    if (!failed) printf("PASS: lm_head matmul matches scalar\n");
    return failed;
}

/* Real-dim check of olmoe_q_proj_forward: full [hidden, hidden] q_proj
 * weight (8 MiB BF16) at seq_len 2, all [2, hidden] output lanes compared
 * against the scalar matmul. */
static int test_q_proj_matmul_matches_scalar(void)
{
    enum { SEQ = 2 };
    olmoe_self_attn_t attn;
    memset(&attn, 0, sizeof attn);

    size_t wn = (size_t)OLMOE_HIDDEN * OLMOE_HIDDEN;
    olmoe_bf16_t *w = malloc(wn * sizeof *w);
    if (!w) { printf("FAIL: q_proj malloc OOM\n"); return 1; }
    for (size_t i = 0; i < wn; ++i)
        w[i] = f32_to_bf16((float)(((int)(i % 17)) - 8) * 0.125f);
    attn.q_proj = w;

    olmoe_act_t x[SEQ * OLMOE_HIDDEN];
    for (size_t i = 0; i < SEQ * OLMOE_HIDDEN; ++i)
        x[i] = (float)((int)(i % 11)) / 100.0f - 0.05f;

    olmoe_act_t got[SEQ * OLMOE_HIDDEN];
    olmoe_act_t want[SEQ * OLMOE_HIDDEN];
    olmoe_q_proj_forward(&attn, x, SEQ, got);
    scalar_matmul_bf16(want, x, w, SEQ, OLMOE_HIDDEN, OLMOE_HIDDEN);

    int failed = lanes_match(got, want, (size_t)SEQ * OLMOE_HIDDEN);
    free(w);
    if (!failed) printf("PASS: q_proj matmul matches scalar\n");
    return failed;
}

/* Real-dim check of olmoe_k_proj_forward: full [hidden, hidden] k_proj
 * weight (8 MiB BF16) at seq_len 2, all [2, hidden] output lanes compared
 * against the scalar matmul. */
static int test_k_proj_matmul_matches_scalar(void)
{
    enum { SEQ = 2 };
    olmoe_self_attn_t attn;
    memset(&attn, 0, sizeof attn);

    size_t wn = (size_t)OLMOE_HIDDEN * OLMOE_HIDDEN;
    olmoe_bf16_t *w = malloc(wn * sizeof *w);
    if (!w) { printf("FAIL: k_proj malloc OOM\n"); return 1; }
    for (size_t i = 0; i < wn; ++i)
        w[i] = f32_to_bf16((float)(((int)(i % 19)) - 9) * 0.1f);
    attn.k_proj = w;

    olmoe_act_t x[SEQ * OLMOE_HIDDEN];
    for (size_t i = 0; i < SEQ * OLMOE_HIDDEN; ++i)
        x[i] = (float)((int)(i % 13)) / 100.0f - 0.06f;

    olmoe_act_t got[SEQ * OLMOE_HIDDEN];
    olmoe_act_t want[SEQ * OLMOE_HIDDEN];
    olmoe_k_proj_forward(&attn, x, SEQ, got);
    scalar_matmul_bf16(want, x, w, SEQ, OLMOE_HIDDEN, OLMOE_HIDDEN);

    int failed = lanes_match(got, want, (size_t)SEQ * OLMOE_HIDDEN);
    free(w);
    if (!failed) printf("PASS: k_proj matmul matches scalar\n");
    return failed;
}

/* Real-dim check of olmoe_v_proj_forward: full [hidden, hidden] v_proj
 * weight (8 MiB BF16) at seq_len 2, all [2, hidden] output lanes compared
 * against the scalar matmul. */
static int test_v_proj_matmul_matches_scalar(void)
{
    enum { SEQ = 2 };
    olmoe_self_attn_t attn;
    memset(&attn, 0, sizeof attn);

    size_t wn = (size_t)OLMOE_HIDDEN * OLMOE_HIDDEN;
    olmoe_bf16_t *w = malloc(wn * sizeof *w);
    if (!w) { printf("FAIL: v_proj malloc OOM\n"); return 1; }
    for (size_t i = 0; i < wn; ++i)
        w[i] = f32_to_bf16((float)(((int)(i % 23)) - 11) * 0.05f);
    attn.v_proj = w;

    olmoe_act_t x[SEQ * OLMOE_HIDDEN];
    for (size_t i = 0; i < SEQ * OLMOE_HIDDEN; ++i)
        x[i] = (float)((int)(i % 9)) / 100.0f - 0.04f;

    olmoe_act_t got[SEQ * OLMOE_HIDDEN];
    olmoe_act_t want[SEQ * OLMOE_HIDDEN];
    olmoe_v_proj_forward(&attn, x, SEQ, got);
    scalar_matmul_bf16(want, x, w, SEQ, OLMOE_HIDDEN, OLMOE_HIDDEN);

    int failed = lanes_match(got, want, (size_t)SEQ * OLMOE_HIDDEN);
    free(w);
    if (!failed) printf("PASS: v_proj matmul matches scalar\n");
    return failed;
}

/* Real-dim check of olmoe_o_proj_forward: full [hidden, hidden] o_proj
 * weight (8 MiB BF16) at seq_len 2, all [2, hidden] output lanes compared
 * against the scalar matmul. */
static int test_o_proj_matmul_matches_scalar(void)
{
    enum { SEQ = 2 };
    olmoe_self_attn_t attn;
    memset(&attn, 0, sizeof attn);

    size_t wn = (size_t)OLMOE_HIDDEN * OLMOE_HIDDEN;
    olmoe_bf16_t *w = malloc(wn * sizeof *w);
    if (!w) { printf("FAIL: o_proj malloc OOM\n"); return 1; }
    for (size_t i = 0; i < wn; ++i)
        w[i] = f32_to_bf16((float)(((int)(i % 29)) - 14) * 0.0625f);
    attn.o_proj = w;

    olmoe_act_t x[SEQ * OLMOE_HIDDEN];
    for (size_t i = 0; i < SEQ * OLMOE_HIDDEN; ++i)
        x[i] = (float)((int)(i % 7)) / 100.0f - 0.03f;

    olmoe_act_t got[SEQ * OLMOE_HIDDEN];
    olmoe_act_t want[SEQ * OLMOE_HIDDEN];
    olmoe_o_proj_forward(&attn, x, SEQ, got);
    scalar_matmul_bf16(want, x, w, SEQ, OLMOE_HIDDEN, OLMOE_HIDDEN);

    int failed = lanes_match(got, want, (size_t)SEQ * OLMOE_HIDDEN);
    free(w);
    if (!failed) printf("PASS: o_proj matmul matches scalar\n");
    return failed;
}

int test_engine_matmul_pass(void)
{
    int failed = 0;
    failed += test_lm_head_matmul_matches_scalar();
    failed += test_q_proj_matmul_matches_scalar();
    failed += test_k_proj_matmul_matches_scalar();
    failed += test_v_proj_matmul_matches_scalar();
    failed += test_o_proj_matmul_matches_scalar();
    return failed;
}