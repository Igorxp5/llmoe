#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "olmoe/engine/engine.h"
#include "kernels/cpu_matmul.h"

#include "test_engine_helpers.h"

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

static int test_matmul_kernel_matches_scalar(void)
{
    int failed = check_matmul_kernel_small_dim();
    if (!failed) printf("PASS: cpu_matmul_bf16 small dim matches scalar\n");
    return failed;
}

/* Real-dim check of olmoe_q_proj_forward: full [hidden, hidden] q_proj
 * weight (8 MiB BF16) at seq_len 2, all [2, hidden] output lanes compared
 * against the scalar matmul. The struct is static so its 32 MiB lives in
 * BSS (the test only writes one field). */
static int test_q_proj_matmul_matches_scalar(void)
{
    enum { SEQ = 2 };
    static olmoe_self_attn_t attn;

    size_t wn = (size_t)OLMOE_HIDDEN * OLMOE_HIDDEN;
    olmoe_bf16_t *w = malloc(wn * sizeof *w);
    if (!w) { printf("FAIL: q_proj malloc OOM\n"); return 1; }
    for (size_t i = 0; i < wn; ++i)
        w[i] = f32_to_bf16((float)(((int)(i % 17)) - 8) * 0.125f);
    memcpy(attn.q_proj, w, wn * sizeof *w);
    free(w);

    olmoe_act_t x[SEQ * OLMOE_HIDDEN];
    for (size_t i = 0; i < SEQ * OLMOE_HIDDEN; ++i)
        x[i] = (float)((int)(i % 11)) / 100.0f - 0.05f;

    olmoe_act_t got[SEQ * OLMOE_HIDDEN];
    olmoe_act_t want[SEQ * OLMOE_HIDDEN];
    olmoe_q_proj_forward(&attn, x, SEQ, got);
    scalar_matmul_bf16(want, x, attn.q_proj, SEQ, OLMOE_HIDDEN, OLMOE_HIDDEN);

    int failed = lanes_match(got, want, (size_t)SEQ * OLMOE_HIDDEN);
    if (!failed) printf("PASS: q_proj matmul matches scalar\n");
    return failed;
}

/* Real-dim check of olmoe_k_proj_forward: full [hidden, hidden] k_proj
 * weight (8 MiB BF16) at seq_len 2, all [2, hidden] output lanes compared
 * against the scalar matmul. */
static int test_k_proj_matmul_matches_scalar(void)
{
    enum { SEQ = 2 };
    static olmoe_self_attn_t attn;

    size_t wn = (size_t)OLMOE_HIDDEN * OLMOE_HIDDEN;
    olmoe_bf16_t *w = malloc(wn * sizeof *w);
    if (!w) { printf("FAIL: k_proj malloc OOM\n"); return 1; }
    for (size_t i = 0; i < wn; ++i)
        w[i] = f32_to_bf16((float)(((int)(i % 19)) - 9) * 0.1f);
    memcpy(attn.k_proj, w, wn * sizeof *w);
    free(w);

    olmoe_act_t x[SEQ * OLMOE_HIDDEN];
    for (size_t i = 0; i < SEQ * OLMOE_HIDDEN; ++i)
        x[i] = (float)((int)(i % 13)) / 100.0f - 0.06f;

    olmoe_act_t got[SEQ * OLMOE_HIDDEN];
    olmoe_act_t want[SEQ * OLMOE_HIDDEN];
    olmoe_k_proj_forward(&attn, x, SEQ, got);
    scalar_matmul_bf16(want, x, attn.k_proj, SEQ, OLMOE_HIDDEN, OLMOE_HIDDEN);

    int failed = lanes_match(got, want, (size_t)SEQ * OLMOE_HIDDEN);
    if (!failed) printf("PASS: k_proj matmul matches scalar\n");
    return failed;
}

/* Real-dim check of olmoe_v_proj_forward: full [hidden, hidden] v_proj
 * weight (8 MiB BF16) at seq_len 2, all [2, hidden] output lanes compared
 * against the scalar matmul. */
static int test_v_proj_matmul_matches_scalar(void)
{
    enum { SEQ = 2 };
    static olmoe_self_attn_t attn;

    size_t wn = (size_t)OLMOE_HIDDEN * OLMOE_HIDDEN;
    olmoe_bf16_t *w = malloc(wn * sizeof *w);
    if (!w) { printf("FAIL: v_proj malloc OOM\n"); return 1; }
    for (size_t i = 0; i < wn; ++i)
        w[i] = f32_to_bf16((float)(((int)(i % 23)) - 11) * 0.05f);
    memcpy(attn.v_proj, w, wn * sizeof *w);
    free(w);

    olmoe_act_t x[SEQ * OLMOE_HIDDEN];
    for (size_t i = 0; i < SEQ * OLMOE_HIDDEN; ++i)
        x[i] = (float)((int)(i % 9)) / 100.0f - 0.04f;

    olmoe_act_t got[SEQ * OLMOE_HIDDEN];
    olmoe_act_t want[SEQ * OLMOE_HIDDEN];
    olmoe_v_proj_forward(&attn, x, SEQ, got);
    scalar_matmul_bf16(want, x, attn.v_proj, SEQ, OLMOE_HIDDEN, OLMOE_HIDDEN);

    int failed = lanes_match(got, want, (size_t)SEQ * OLMOE_HIDDEN);
    if (!failed) printf("PASS: v_proj matmul matches scalar\n");
    return failed;
}

/* Real-dim check of olmoe_o_proj_forward: full [hidden, hidden] o_proj
 * weight (8 MiB BF16) at seq_len 2, all [2, hidden] output lanes compared
 * against the scalar matmul. */
static int test_o_proj_matmul_matches_scalar(void)
{
    enum { SEQ = 2 };
    static olmoe_self_attn_t attn;

    size_t wn = (size_t)OLMOE_HIDDEN * OLMOE_HIDDEN;
    olmoe_bf16_t *w = malloc(wn * sizeof *w);
    if (!w) { printf("FAIL: o_proj malloc OOM\n"); return 1; }
    for (size_t i = 0; i < wn; ++i)
        w[i] = f32_to_bf16((float)(((int)(i % 29)) - 14) * 0.0625f);
    memcpy(attn.o_proj, w, wn * sizeof *w);
    free(w);

    olmoe_act_t x[SEQ * OLMOE_HIDDEN];
    for (size_t i = 0; i < SEQ * OLMOE_HIDDEN; ++i)
        x[i] = (float)((int)(i % 7)) / 100.0f - 0.03f;

    olmoe_act_t got[SEQ * OLMOE_HIDDEN];
    olmoe_act_t want[SEQ * OLMOE_HIDDEN];
    olmoe_o_proj_forward(&attn, x, SEQ, got);
    scalar_matmul_bf16(want, x, attn.o_proj, SEQ, OLMOE_HIDDEN, OLMOE_HIDDEN);

    int failed = lanes_match(got, want, (size_t)SEQ * OLMOE_HIDDEN);
    if (!failed) printf("PASS: o_proj matmul matches scalar\n");
    return failed;
}

/* Fill one [hidden, hidden] attn weight field with a base-specific value
 * pattern so the three projections carry distinct weights (a cross-wired
 * pointer in the fused path would then fail the lane comparison). */
static int fill_weight_field(olmoe_bf16_t * restrict field, int base)
{
    size_t wn = (size_t)OLMOE_HIDDEN * OLMOE_HIDDEN;
    olmoe_bf16_t *w = malloc(wn * sizeof *w);
    if (!w) {
        return 1;
    }
    for (size_t i = 0; i < wn; ++i)
        w[i] = f32_to_bf16((float)(((int)(i % base) - base / 2) * 0.125f));
    memcpy(field, w, wn * sizeof *w);
    free(w);
    return 0;
}

/* The fused q/k/v path must equal three independent scalar matmuls for all
 * three projections: it reads the same input once and splits the work across
 * one OpenMP region instead of three. */
static int test_qkv_proj_fused_matches_scalar(void)
{
    enum { SEQ = 2 };
    static olmoe_self_attn_t attn;

    if (fill_weight_field(attn.q_proj, 17) ||
        fill_weight_field(attn.k_proj, 19) ||
        fill_weight_field(attn.v_proj, 23)) {
        printf("FAIL: qkv weight malloc OOM\n");
        return 1;
    }

    olmoe_act_t x[SEQ * OLMOE_HIDDEN];
    for (size_t i = 0; i < SEQ * OLMOE_HIDDEN; ++i)
        x[i] = (float)((int)(i % 11)) / 100.0f - 0.05f;

    olmoe_act_t gq[SEQ * OLMOE_HIDDEN], gk[SEQ * OLMOE_HIDDEN];
    olmoe_act_t gv[SEQ * OLMOE_HIDDEN], wq[SEQ * OLMOE_HIDDEN];
    olmoe_act_t wk[SEQ * OLMOE_HIDDEN], wv[SEQ * OLMOE_HIDDEN];
    olmoe_qkv_proj_forward(&attn, x, SEQ, gq, gk, gv);
    scalar_matmul_bf16(wq, x, attn.q_proj, SEQ, OLMOE_HIDDEN, OLMOE_HIDDEN);
    scalar_matmul_bf16(wk, x, attn.k_proj, SEQ, OLMOE_HIDDEN, OLMOE_HIDDEN);
    scalar_matmul_bf16(wv, x, attn.v_proj, SEQ, OLMOE_HIDDEN, OLMOE_HIDDEN);

    int failed = lanes_match(gq, wq, (size_t)SEQ * OLMOE_HIDDEN);
    failed += lanes_match(gk, wk, (size_t)SEQ * OLMOE_HIDDEN);
    failed += lanes_match(gv, wv, (size_t)SEQ * OLMOE_HIDDEN);
    if (!failed) printf("PASS: fused qkv matmul matches scalar\n");
    return failed;
}

int test_engine_matmul_pass(void)
{
    int failed = 0;
    failed += test_matmul_kernel_matches_scalar();
    failed += test_q_proj_matmul_matches_scalar();
    failed += test_k_proj_matmul_matches_scalar();
    failed += test_v_proj_matmul_matches_scalar();
    failed += test_o_proj_matmul_matches_scalar();
    failed += test_qkv_proj_fused_matches_scalar();
    return failed;
}
