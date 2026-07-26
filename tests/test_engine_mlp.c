#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#include "olmoe/engine/engine.h"
#include "olmoe/engine/kernels/cpu_softmax.h"
#include "olmoe/engine/kernels/cpu_topk.h"

#include "test_engine_helpers.h"

/* Scalar reference for the OLMoE router: numerically stable softmax over
 * n_experts, then top-K descending with smallest-index tie-break, then
 * renormalize the K selected weights to sum to 1. Mirrors the kernel's
 * semantics so the only divergence is FP32 round-off. */
static void scalar_softmax(float *out, const float *in, size_t n)
{
    float mx = in[0];
    for (size_t r = 1; r < n; ++r)
        if (in[r] > mx) mx = in[r];
    float sum = 0.0f;
    for (size_t r = 0; r < n; ++r) {
        out[r] = expf(in[r] - mx);
        sum += out[r];
    }
    for (size_t r = 0; r < n; ++r)
        out[r] /= sum;
}

static void scalar_topk_desc(const float *scores, size_t n, size_t k,
                             int *idx_out, float *val_out)
{
    bool used[64] = { false };
    for (size_t r = 0; r < k; ++r) {
        size_t best = n;
        float best_v = -INFINITY;
        for (size_t j = 0; j < n; ++j) {
            if (used[j]) continue;
            if (best == n || scores[j] > best_v) {
                best = j;
                best_v = scores[j];
            }
        }
        used[best] = true;
        idx_out[r] = (int)best;
        val_out[r] = scores[best];
    }
}

static void scalar_renorm(float *w, size_t k)
{
    float sum = 0.0f;
    for (size_t r = 0; r < k; ++r) sum += w[r];
    for (size_t r = 0; r < k; ++r) w[r] /= sum;
}

/* Fill w [n_experts*hidden] with deterministic BF16 so every (expert, lane)
 * differs and the router produces a non-degenerate top-K. */
static void fill_router_weights(olmoe_bf16_t *w, size_t n_experts, size_t hidden)
{
    for (size_t j = 0; j < n_experts; ++j)
        for (size_t l = 0; l < hidden; ++l)
            w[j * hidden + l] =
                f32_to_bf16((float)(((int)((j * 7 + l * 3) % 53)) - 26) / 100.0f);
}

/* Fill x [seq*hidden] with small deterministic FP32, distinct per lane. */
static void fill_router_inputs(olmoe_act_t *x, size_t seq, size_t hidden)
{
    for (size_t i = 0; i < seq; ++i)
        for (size_t l = 0; l < hidden; ++l)
            x[i * hidden + l] =
                (float)(((int)((i * 5 + l * 11) % 19)) - 9) / 200.0f;
}

/* Compare kernel vs scalar topk_idx (exact int) and topk_w (rtol 1e-4
 * atol 1e-5). Prints per-cell FAIL lines. */
static int compare_router_outputs(const int *got_idx, const int *want_idx,
                                  const olmoe_act_t *got_w,
                                  const olmoe_act_t *want_w,
                                  size_t seq, size_t k)
{
    int failed = 0;
    for (size_t i = 0; i < seq; ++i) {
        for (size_t r = 0; r < k; ++r) {
            size_t p = i * k + r;
            if (got_idx[p] != want_idx[p]) {
                printf("FAIL: mlp_gate idx mismatch at token %zu rank %zu: "
                       "got=%d want=%d\n", i, r, got_idx[p], want_idx[p]);
                ++failed;
            }
            float d = fabsf(got_w[p] - want_w[p]);
            if (d > 1e-5f && d > 1e-4f * fabsf(want_w[p])) {
                printf("FAIL: mlp_gate weight mismatch at token %zu rank %zu: "
                       "got=%.7f want=%.7f\n", i, r, got_w[p], want_w[p]);
                ++failed;
            }
        }
    }
    return failed;
}

/* Real-dim check of olmoe_mlp_gate_forward: full [64, 2048] BF16 router
 * weight at seq_len 3, compared against a pure-C softmax+topk+renorm. */
static int test_mlp_gate_matches_scalar(void)
{
    enum { SEQ = 3 };
    size_t n = (size_t)OLMOE_N_EXPERTS, h = (size_t)OLMOE_HIDDEN;
    size_t k = (size_t)OLMOE_N_EXPERTS_PER_TOK;

    olmoe_bf16_t *w = malloc(n * h * sizeof *w);
    olmoe_act_t *x = malloc(SEQ * h * sizeof *x);
    float *logits = malloc(SEQ * n * sizeof *logits);
    if (!w || !x || !logits) {
        printf("FAIL: mlp_gate malloc OOM\n");
        free(w); free(x); free(logits);
        return 1;
    }
    fill_router_weights(w, n, h);
    fill_router_inputs(x, SEQ, h);

    int got_idx[SEQ * OLMOE_N_EXPERTS_PER_TOK];
    olmoe_act_t got_w[SEQ * OLMOE_N_EXPERTS_PER_TOK];
    olmoe_mlp_gate_forward(w, x, SEQ, got_idx, got_w);

    scalar_matmul_bf16(logits, x, w, SEQ, n, h);
    int want_idx[SEQ * OLMOE_N_EXPERTS_PER_TOK];
    olmoe_act_t want_w[SEQ * OLMOE_N_EXPERTS_PER_TOK];
    for (size_t i = 0; i < SEQ; ++i) {
        float probs[64];
        scalar_softmax(probs, logits + i * n, n);
        scalar_topk_desc(probs, n, k,
                         want_idx + i * k, (float *)want_w + i * k);
        scalar_renorm((float *)want_w + i * k, k);
    }

    int failed = compare_router_outputs(got_idx, want_idx,
                                        got_w, want_w, SEQ, k);
    free(w); free(x); free(logits);
    if (!failed) printf("PASS: mlp_gate matches scalar\n");
    return failed;
}

int test_engine_mlp_pass(void)
{
    int failed = 0;
    failed += test_mlp_gate_matches_scalar();
    return failed;
}