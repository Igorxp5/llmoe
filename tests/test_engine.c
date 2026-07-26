#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "olmoe/engine/engine.h"
#include "olmoe/engine/kernels/cpu_matmul.h"

#include "test_engine.h"

/* File-local BF16 utilities, used only by test_input_ln_matches_scalar so
 * both the test's weights and its scalar reference share the same BF16
 * rounding. Round-to-nearest-even, matching the on-disk format. */
static olmoe_bf16_t f32_to_bf16(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    uint32_t lsb = (u >> 16) & 1;
    uint32_t rounding_bias = 0x7fff + lsb;
    uint32_t rounded = u + rounding_bias;
    return (olmoe_bf16_t)(rounded >> 16);
}

static float bf16_to_f32(olmoe_bf16_t b)
{
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

/* Pure-C scalar RMSNorm using the identical BF16->FP32 weight promotion as
 * the SIMD impl; the only divergence expected is FP32 round-off. */
static void scalar_input_ln(const olmoe_bf16_t *w, const olmoe_act_t *x,
                            size_t seq_len, olmoe_act_t *out)
{
    const float eps = 1e-5f;
    for (size_t i = 0; i < seq_len; ++i) {
        const olmoe_act_t *xr = x + i * OLMOE_HIDDEN;
        olmoe_act_t *or_ = out + i * OLMOE_HIDDEN;
        float ss = 0.0f;
        for (size_t k = 0; k < OLMOE_HIDDEN; ++k) ss += xr[k] * xr[k];
        float scale = 1.0f / sqrtf(ss / (float)OLMOE_HIDDEN + eps);
        for (size_t k = 0; k < OLMOE_HIDDEN; ++k)
            or_[k] = xr[k] * scale * bf16_to_f32(w[k]);
    }
}

/* ---------- checks ------------------------------------------------------ */

/* init + free must roundtrip without leaking or crashing for a typical
 * short sequence. */
static int test_scratch_init_free_roundtrip(void)
{
    olmoe_scratch_t s;
    olmoe_status_t st = olmoe_scratch_init(&s, 16);
    if (st != OLMOE_OK) {
        printf("FAIL: scratch_init(16) -> %d (want OK)\n", st);
        return 1;
    }
    int failed = 0;
    if (s.seq_len != 16) { printf("FAIL: seq_len=%zu (want 16)\n", s.seq_len); ++failed; }
    if (!s.hidden_in || !s.hidden_out || !s.q || !s.k || !s.v || !s.ctx ||
        !s.router_logits || !s.topk_idx || !s.topk_w ||
        !s.expert_in || !s.expert_out || !s.logits) {
        printf("FAIL: scratch_init left a NULL buffer\n");
        ++failed;
    }
    olmoe_scratch_free(&s);
    if (s.hidden_in || s.logits || s.seq_len != 0) {
        printf("FAIL: scratch_free did not zero the struct\n");
        ++failed;
    }
    if (!failed) printf("PASS: scratch init/free roundtrip\n");
    return failed;
}

/* init(NULL) must not dereference; returns OLMOE_ERR_NULL. */
static int test_scratch_init_null_returns_err(void)
{
    olmoe_status_t st = olmoe_scratch_init(NULL, 8);
    if (st != OLMOE_ERR_NULL) {
        printf("FAIL: scratch_init(NULL) -> %d (want ERR_NULL)\n", st);
        return 1;
    }
    printf("PASS: scratch_init(NULL) returns ERR_NULL\n");
    return 0;
}

/* free(NULL) must be a no-op (mirrors olmoe_model_free's NULL-safety). */
static int test_scratch_free_null_is_safe(void)
{
    olmoe_scratch_free(NULL);
    printf("PASS: scratch_free(NULL) is safe\n");
    return 0;
}

/* olmoe_forward stub: a zeroed model is acceptable because the orchestrator
 * only NULL-checks its direct args (per-kind ops are not called by the
 * stub yet). */
static int test_forward_stub_returns_ok(void)
{
    olmoe_scratch_t s;
    olmoe_scratch_init(&s, 4);
    olmoe_model_t empty_model;
    memset(&empty_model, 0, sizeof empty_model);
    int ids[] = {1, 2, 3, 4};

    int failed = 0;
    olmoe_status_t st = olmoe_forward(&empty_model, ids, 4, &s, s.logits);
    if (st != OLMOE_OK) {
        printf("FAIL: forward(stub) -> %d (want OK)\n", st);
        ++failed;
    }
    if (!failed) printf("PASS: forward stub returns OK\n");
    olmoe_scratch_free(&s);
    return failed;
}

/* forward must reject a seq_len larger than the scratch was sized for. */
static int test_forward_oversize_seq_returns_shape(void)
{
    olmoe_scratch_t s;
    olmoe_scratch_init(&s, 4);
    olmoe_model_t empty_model;
    memset(&empty_model, 0, sizeof empty_model);
    int ids[] = {1, 2, 3, 4, 5, 6, 7, 8};

    int failed = 0;
    olmoe_status_t st = olmoe_forward(&empty_model, ids, 8, &s, s.logits);
    if (st != OLMOE_ERR_SHAPE) {
        printf("FAIL: forward(seq=8 on scratch=4) -> %d (want ERR_SHAPE)\n", st);
        ++failed;
    } else {
        printf("PASS: forward oversize seq rejected with ERR_SHAPE\n");
    }
    olmoe_scratch_free(&s);
    return failed;
}

/* olmoe_input_ln_forward: deterministic + random rows compared against an
 * in-test pure-C scalar RMSNorm that promotes the same BF16 weights, so the
 * only divergence is FP32 SIMD-vs-scalar round-off. */
static int test_input_ln_matches_scalar(void)
{
    enum { ROWS = 3 };
    olmoe_bf16_t w[OLMOE_HIDDEN];
    olmoe_act_t x[ROWS * OLMOE_HIDDEN];
    olmoe_act_t got[ROWS * OLMOE_HIDDEN];
    olmoe_act_t want[ROWS * OLMOE_HIDDEN];

    /* Weights: ramp 1..hidden so every lane is distinct. */
    for (size_t k = 0; k < OLMOE_HIDDEN; ++k) w[k] = f32_to_bf16((float)(k + 1));

    /* Row 0: constant 1. Row 1: alternating +/-1. Row 2: pseudo-random
     * via a cheap LCG (no libc dependency). */
    for (size_t k = 0; k < OLMOE_HIDDEN; ++k) x[k] = 1.0f;
    for (size_t k = 0; k < OLMOE_HIDDEN; ++k) x[OLMOE_HIDDEN + k] = (k & 1) ? -1.0f : 1.0f;
    unsigned int rng = 0xdeadbeefu;
    for (size_t k = 0; k < OLMOE_HIDDEN; ++k) {
        rng = rng * 1664525u + 1013904223u;
        x[2 * OLMOE_HIDDEN + k] = (float)((int)rng % 1001) / 500.0f - 1.0f;
    }

    int failed = 0;
    olmoe_input_ln_forward(w, x, ROWS, got);
    scalar_input_ln(w, x, ROWS, want);

    /* SIMD and scalar differ only by FP32 round-off; rtol 1e-4, atol 1e-5. */
    for (size_t i = 0; i < ROWS * OLMOE_HIDDEN && !failed; ++i) {
        float a = got[i], b = want[i];
        float diff = fabsf(a - b);
        if (diff > 1e-5f && diff > 1e-4f * fabsf(b)) {
            printf("FAIL: input_ln lane %zu got=%.7f want=%.7f\n", i, a, b);
            ++failed;
        }
    }
    if (!failed) printf("PASS: input_ln_forward matches scalar RMSNorm\n");
    return failed;
}

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

/* rtol 1e-4, atol 1e-5 — matches the kernel-vs-scalar FP32 round-off budget. */
static int lanes_match(const olmoe_act_t *got, const olmoe_act_t *want, size_t n)
{
    int failed = 0;
    for (size_t i = 0; i < n; ++i) {
        float d = fabsf(got[i] - want[i]);
        if (d > 1e-5f && d > 1e-4f * fabsf(want[i])) {
            printf("FAIL: lane %zu got=%.7f want=%.7f\n", i, got[i], want[i]);
            ++failed;
        }
    }
    return failed;
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

/* ---------- dispatcher --------------------------------------------------- */

int test_engine_stubs_pass(void)
{
    int failed = 0;
    failed += test_scratch_init_free_roundtrip();
    failed += test_scratch_init_null_returns_err();
    failed += test_scratch_free_null_is_safe();
    failed += test_forward_stub_returns_ok();
    failed += test_forward_oversize_seq_returns_shape();
    failed += test_input_ln_matches_scalar();
    failed += test_lm_head_matmul_matches_scalar();
    failed += test_q_proj_matmul_matches_scalar();
    return failed;
}
