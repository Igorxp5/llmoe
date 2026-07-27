#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#include "olmoe/engine/engine.h"
#include "kernels/cpu_softmax.h"
#include "kernels/cpu_topk.h"

#include "test_engine_helpers.h"

/* Scalar reference for the OLMoE router: numerically stable softmax over
 * n_experts, then top-K descending with smallest-index tie-break.
 * norm_topk_prob is false for this model, so no renormalization. */
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
    }

    int failed = compare_router_outputs(got_idx, want_idx,
                                        got_w, want_w, SEQ, k);
    free(w); free(x); free(logits);
    if (!failed) printf("PASS: mlp_gate matches scalar\n");
    return failed;
}

/* Real-dim check of olmoe_expert_gate_forward: full [inter, hidden] BF16
 * gate_proj (4 MiB) at n_tokens 2, compared against the shared scalar
 * matmul reference over all 2*inter output lanes. */
static int test_expert_gate_matches_scalar(void)
{
    enum { TOK = 2 };
    size_t inter = (size_t)OLMOE_INTER, hidden = (size_t)OLMOE_HIDDEN;

    olmoe_expert_t expert;
    memset(&expert, 0, sizeof expert);
    expert.gate_proj = malloc(inter * hidden * sizeof *expert.gate_proj);
    olmoe_act_t *x = malloc(TOK * hidden * sizeof *x);
    olmoe_act_t *out = malloc(TOK * inter * sizeof *out);
    olmoe_act_t *scalar_out = malloc(TOK * inter * sizeof *scalar_out);
    if (!expert.gate_proj || !x || !out || !scalar_out) {
        printf("FAIL: expert_gate malloc OOM\n");
        free(expert.gate_proj); free(x); free(out); free(scalar_out);
        return 1;
    }

    for (size_t j = 0; j < inter; ++j)
        for (size_t l = 0; l < hidden; ++l)
            expert.gate_proj[j * hidden + l] =
                f32_to_bf16((float)(((int)((j * hidden + l) % 13)) - 6) * 0.0625f);
    for (size_t i = 0; i < TOK; ++i)
        for (size_t l = 0; l < hidden; ++l)
            x[i * hidden + l] =
                (float)(((int)((i * 5 + l * 3) % 17)) - 8) * 0.03125f;

    olmoe_expert_gate_forward(&expert, x, TOK, out);
    scalar_matmul_bf16(scalar_out, x, expert.gate_proj, TOK, inter, hidden);

    int failed = lanes_match(out, scalar_out, TOK * inter);
    free(expert.gate_proj); free(x); free(out); free(scalar_out);
    if (!failed) printf("PASS: expert_gate matmul matches scalar\n");
    else         printf("FAIL: expert_gate matmul matches scalar\n");
    return failed;
}

/* Real-dim check of olmoe_expert_up_forward: full [inter, hidden] BF16
 * up_proj (4 MiB) at n_tokens 2, compared against the shared scalar
 * matmul reference over all 2*inter output lanes. Uses a distinct weight
 * pattern from expert_gate so up_proj is independently exercised. */
static int test_expert_up_matches_scalar(void)
{
    enum { TOK = 2 };
    size_t inter = (size_t)OLMOE_INTER, hidden = (size_t)OLMOE_HIDDEN;

    olmoe_expert_t expert;
    memset(&expert, 0, sizeof expert);
    expert.up_proj = malloc(inter * hidden * sizeof *expert.up_proj);
    olmoe_act_t *x = malloc(TOK * hidden * sizeof *x);
    olmoe_act_t *out = malloc(TOK * inter * sizeof *out);
    olmoe_act_t *scalar_out = malloc(TOK * inter * sizeof *scalar_out);
    if (!expert.up_proj || !x || !out || !scalar_out) {
        printf("FAIL: expert_up malloc OOM\n");
        free(expert.up_proj); free(x); free(out); free(scalar_out);
        return 1;
    }

    for (size_t j = 0; j < inter; ++j)
        for (size_t l = 0; l < hidden; ++l)
            expert.up_proj[j * hidden + l] =
                f32_to_bf16((float)(((int)((j * hidden + l) % 11)) - 5) * 0.125f);
    for (size_t i = 0; i < TOK; ++i)
        for (size_t l = 0; l < hidden; ++l)
            x[i * hidden + l] =
                (float)(((int)((i * 5 + l * 3) % 17)) - 8) * 0.03125f;

    olmoe_expert_up_forward(&expert, x, TOK, out);
    scalar_matmul_bf16(scalar_out, x, expert.up_proj, TOK, inter, hidden);

    int failed = lanes_match(out, scalar_out, TOK * inter);
    free(expert.up_proj); free(x); free(out); free(scalar_out);
    if (!failed) printf("PASS: expert_up matmul matches scalar\n");
    else         printf("FAIL: expert_up matmul matches scalar\n");
    return failed;
}

/* Real-dim check of olmoe_expert_down_forward: full [hidden, inter] BF16
 * down_proj (4 MiB) at n_tokens 2, comparing all 2*hidden output lanes
 * against the shared scalar matmul reference. Input is the inter activation
 * [tok, inter]; down projects it back to [tok, hidden]. */
static int test_expert_down_matches_scalar(void)
{
    enum { TOK = 2 };
    size_t inter = (size_t)OLMOE_INTER, hidden = (size_t)OLMOE_HIDDEN;

    olmoe_expert_t expert;
    memset(&expert, 0, sizeof expert);
    expert.down_proj = malloc(hidden * inter * sizeof *expert.down_proj);
    olmoe_act_t *x = malloc(TOK * inter * sizeof *x);
    olmoe_act_t *out = malloc(TOK * hidden * sizeof *out);
    olmoe_act_t *scalar_out = malloc(TOK * hidden * sizeof *scalar_out);
    if (!expert.down_proj || !x || !out || !scalar_out) {
        printf("FAIL: expert_down malloc OOM\n");
        free(expert.down_proj); free(x); free(out); free(scalar_out);
        return 1;
    }

    for (size_t j = 0; j < hidden; ++j)
        for (size_t l = 0; l < inter; ++l)
            expert.down_proj[j * inter + l] =
                f32_to_bf16((float)(((int)((j * inter + l) % 7)) - 3) * 0.25f);
    for (size_t i = 0; i < TOK; ++i)
        for (size_t l = 0; l < inter; ++l)
            x[i * inter + l] =
                (float)(((int)((i * 5 + l * 3) % 17)) - 8) * 0.03125f;

    olmoe_expert_down_forward(&expert, x, TOK, out);
    scalar_matmul_bf16(scalar_out, x, expert.down_proj, TOK, hidden, inter);

    int failed = lanes_match(out, scalar_out, TOK * hidden);
    free(expert.down_proj); free(x); free(out); free(scalar_out);
    if (!failed) printf("PASS: expert_down matmul matches scalar\n");
    else         printf("FAIL: expert_down matmul matches scalar\n");
    return failed;
}

int test_engine_mlp_pass(void)
{
    int failed = 0;
    failed += test_mlp_gate_matches_scalar();
    failed += test_expert_gate_matches_scalar();
    failed += test_expert_up_matches_scalar();
    failed += test_expert_down_matches_scalar();
    return failed;
}
