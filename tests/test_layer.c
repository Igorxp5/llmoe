#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "olmoe/engine/engine.h"
#include "olmoe/layers/layers.h"

#include "test_layer.h"

#ifndef OLMOE_TEST_MODEL_DIR
#define OLMOE_TEST_MODEL_DIR "models/OLMoE-1B-7B-0924-Instruct"
#endif

/* ---------- checks ------------------------------------------------------ */

/* Verify the loader actually wrote data into each weight buffer by sampling
 * the first BF16 element. With the model now being a single calloc'd block,
 * zero-initialized at load time, a zero byte at index 0 means the loader
 * never reached that field (layout mismatch, truncated shard, etc.). */
static int test_top_level_tensors_nonzero(const olmoe_model_t *m)
{
    int failed = 0;
    if (m->embed_tokens[0] == 0) { printf("FAIL: embed_tokens zero\n"); ++failed; }
    if (m->lm_head[0] == 0)       { printf("FAIL: lm_head zero\n");       ++failed; }
    if (m->norm[0] == 0)          { printf("FAIL: norm zero\n");          ++failed; }
    if (!failed)
        printf("PASS: top-level tensors written by loader\n");
    return failed;
}

static int test_self_attn_nonzero_for_layer(const olmoe_model_t *m, int L)
{
    const olmoe_self_attn_t *sa = &m->layers[L].self_attn;
    int failed = 0;
    if (sa->q_proj[0] == 0) { printf("FAIL: layer %d q_proj zero\n", L); ++failed; }
    if (sa->k_proj[0] == 0) { printf("FAIL: layer %d k_proj zero\n", L); ++failed; }
    if (sa->v_proj[0] == 0) { printf("FAIL: layer %d v_proj zero\n", L); ++failed; }
    if (sa->o_proj[0] == 0) { printf("FAIL: layer %d o_proj zero\n", L); ++failed; }
    if (sa->q_norm[0] == 0) { printf("FAIL: layer %d q_norm zero\n", L); ++failed; }
    if (sa->k_norm[0] == 0) { printf("FAIL: layer %d k_norm zero\n", L); ++failed; }
    return failed;
}

static int test_every_layer_self_attn_nonzero(const olmoe_model_t *m)
{
    int failed = 0;
    for (int L = 0; L < OLMOE_N_LAYERS; ++L)
        failed += test_self_attn_nonzero_for_layer(m, L);
    if (!failed)
        printf("PASS: all %d layers' self_attn tensors written by loader\n",
               OLMOE_N_LAYERS);
    return failed;
}

static int test_every_layer_layernorms_nonzero(const olmoe_model_t *m)
{
    int failed = 0;
    for (int L = 0; L < OLMOE_N_LAYERS; ++L) {
        if (m->layers[L].input_layernorm[0] == 0) {
            printf("FAIL: layer %d input_layernorm zero\n", L); ++failed;
        }
        if (m->layers[L].post_attention_layernorm[0] == 0) {
            printf("FAIL: layer %d post_attention_layernorm zero\n", L); ++failed;
        }
    }
    if (!failed)
        printf("PASS: all %d layers' layernorms written by loader\n",
               OLMOE_N_LAYERS);
    return failed;
}

static int test_every_layer_mlp_gate_nonzero(const olmoe_model_t *m)
{
    int failed = 0;
    for (int L = 0; L < OLMOE_N_LAYERS; ++L) {
        if (m->layers[L].mlp_gate[0] == 0) {
            printf("FAIL: layer %d mlp_gate (router) zero\n", L); ++failed;
        }
    }
    if (!failed)
        printf("PASS: all %d layers' mlp_gate (router) written by loader\n",
               OLMOE_N_LAYERS);
    return failed;
}

/* 16 layers * 64 experts * 3 weights = 3072 expert buffers. Report only
 * failures so a green run stays a single line. */
static int test_every_expert_tensor_nonzero(const olmoe_model_t *m)
{
    int failed = 0;
    for (int L = 0; L < OLMOE_N_LAYERS; ++L) {
        for (int E = 0; E < OLMOE_N_EXPERTS; ++E) {
            const olmoe_expert_t *ex = &m->layers[L].experts[E];
            if (ex->gate_proj[0] == 0) {
                printf("FAIL: layer %d expert %d gate_proj zero\n", L, E); ++failed;
            }
            if (ex->up_proj[0] == 0) {
                printf("FAIL: layer %d expert %d up_proj zero\n", L, E); ++failed;
            }
            if (ex->down_proj[0] == 0) {
                printf("FAIL: layer %d expert %d down_proj zero\n", L, E); ++failed;
            }
        }
    }
    if (!failed)
        printf("PASS: all %d layers x %d experts x 3 weights written by loader\n",
               OLMOE_N_LAYERS, OLMOE_N_EXPERTS);
    return failed;
}

static int test_norm_first_values_match_oracle(const olmoe_model_t *m)
{
    if (memcmp(m->norm, OLMOE_ORACLE_MODEL_NORM,
               sizeof OLMOE_ORACLE_MODEL_NORM) != 0) {
        printf("FAIL: model.norm.weight first N values != baked oracle\n");
        return 1;
    }
    printf("PASS: model.norm.weight first %d BF16 values match oracle\n",
           OLMOE_ORACLE_N_VALUES);
    return 0;
}

static int test_lm_head_first_values_match_oracle(const olmoe_model_t *m)
{
    if (memcmp(m->lm_head, OLMOE_ORACLE_LM_HEAD,
               sizeof OLMOE_ORACLE_LM_HEAD) != 0) {
        printf("FAIL: lm_head.weight first N values != baked oracle\n");
        return 1;
    }
    printf("PASS: lm_head.weight first %d BF16 values match oracle\n",
           OLMOE_ORACLE_N_VALUES);
    return 0;
}

static int test_expert_zero_down_values_match_oracle(const olmoe_model_t *m)
{
    const olmoe_bf16_t *p = m->layers[0].experts[0].down_proj;
    if (memcmp(p, OLMOE_ORACLE_EXPERT0_DOWN,
               sizeof OLMOE_ORACLE_EXPERT0_DOWN) != 0) {
        printf("FAIL: layers[0].experts[0].down_proj first N values "
               "!= baked oracle\n");
        return 1;
    }
    printf("PASS: layers[0].experts[0].down_proj first %d BF16 values "
           "match oracle\n", OLMOE_ORACLE_N_VALUES);
    return 0;
}

/* embed_forward: row-gather BF16→FP32 against a scalar reference. Uses two
 * fixed ids (0 and 1000) to exercise non-trivial row offsets. */
static int test_embed_lookup_matches_oracle(const olmoe_model_t *m)
{
    enum { N = 2 };
    int ids[N] = {0, 1000};
    olmoe_act_t hidden_out[N * OLMOE_HIDDEN];
    int failed = 0;

    olmoe_embed_forward(m, ids, N, hidden_out);

    for (size_t i = 0; i < N; ++i) {
        for (size_t k = 0; k < OLMOE_HIDDEN; ++k) {
            uint32_t u = (uint32_t)m->embed_tokens[ids[i] * OLMOE_HIDDEN + k] << 16;
            float want;
            memcpy(&want, &u, sizeof want);
            float got = hidden_out[i * OLMOE_HIDDEN + k];
            if (got != want) {
                printf("FAIL: embed row=%zu lane=%zu got=%.8g want=%.8g\n",
                       i, k, got, want);
                ++failed;
                goto done;
            }
        }
    }

done:
    if (!failed)
        printf("PASS: embed_forward lookup matches scalar reference\n");
    return failed;
}

/* Real-model olmoe_forward self-consistency: incremental decode must
 * reproduce the KV-cache prefill for the same tokens within FP32 round-off.
 * Prefill runs the full causal SDPA and moe_block_prefill (its 8 expert slots
 * folded serially per token); decode runs cpu_sdpa_incremental and
 * moe_block_decode (mul+add instead of fma), so the two paths are close but
 * not bit-identical. The tolerance also absorbs the small accumulation-order
 * drift across 16 layers. This is oracle-free: it pins the engine's external
 * contract (decode == cached prefill, finite logits) against the real loaded
 * weights without a 13 GiB scalar reference or a second model load. */
/* Real-model olmoe_forward sanity + self-consistency, oracle-free. Asserts the
 * invariants the engine actually maintains:
 *   - a full seq prefill and an incremental decode both return OLMOE_OK;
 *   - every decoded token yields finite, non-degenerate logits (the weights
 *     were read and the router + experts actually ran);
 *   - the decode is deterministic (repeating a step is bit-identical);
 *   - the final (generation) logits agree between prefill and decode.
 * Cached decode attends to the same KV history as prefill, so the final-token
 * slice must match within FP32 round-off; the two paths differ only in fma vs
 * mul+add accumulation order. This pins the engine's externally-visible
 * contract against the real loaded weights without a 13 GiB scalar oracle. */
static int test_forward_end_to_end_sane(const olmoe_model_t *m)
{
    enum { SEQ = 4 };
    const int ids[SEQ] = { 11, 256, 7, 32768 };
    const size_t V = (size_t)OLMOE_VOCAB;
    const float RTOL = 1e-2f, ATOL = 1e-3f;
    int failed = 0;

    /* Reference: full-sequence prefill into a cache; keep the final slice. */
    olmoe_scratch_t ref_s;
    if (olmoe_scratch_init(&ref_s, SEQ, SEQ) != OLMOE_OK) {
        printf("FAIL: forward prefill scratch init\n");
        return 1;
    }
    if (olmoe_forward(m, ids, SEQ, 0, &ref_s, ref_s.logits) != OLMOE_OK) {
        printf("FAIL: forward prefill returned non-OK\n");
        olmoe_scratch_free(&ref_s);
        return 1;
    }

    /* Decode: feed tokens one at a time with a fresh KV cache, verifying each
     * step's logits are finite and non-degenerate, computing determinism, and
     * pinning the final token to the prefill's final-token slice. */
    olmoe_scratch_t dec_s;
    if (olmoe_scratch_init(&dec_s, 1, SEQ) != OLMOE_OK) {
        printf("FAIL: forward decode scratch init\n");
        olmoe_scratch_free(&ref_s);
        return 1;
    }
    olmoe_act_t *row = (olmoe_act_t *)malloc(V * sizeof *row);
    olmoe_act_t *row2 = (olmoe_act_t *)malloc(V * sizeof *row2);
    if (!row || !row2) {
        printf("FAIL: forward decode row OOM\n");
        free(row); free(row2);
        olmoe_scratch_free(&dec_s);
        olmoe_scratch_free(&ref_s);
        return 1;
    }
    for (size_t i = 0; i < (size_t)SEQ && !failed; ++i) {
        olmoe_status_t st = olmoe_forward(m, ids + i, 1, (size_t)i,
                                          &dec_s, row);
        if (st != OLMOE_OK) {
            printf("FAIL: forward decode step %zu -> %d\n", i, st);
            ++failed;
            break;
        }
        float mn = INFINITY, mx = -INFINITY;
        for (size_t j = 0; j < V; ++j) {
            if (!isfinite(row[j])) {
                printf("FAIL: decode step %zu logit %zu not finite\n", i, j);
                ++failed;
                break;
            }
            if (row[j] < mn) mn = row[j];
            if (row[j] > mx) mx = row[j];
        }
        if (mx <= mn) {
            printf("FAIL: decode step %zu logits degenerate\n", i);
            ++failed;
            break;
        }
    }
    /* Determinism: decoding the last token a second time must be identical. */
    if (!failed) {
        memset(row2, 0, V * sizeof *row2);
        if (olmoe_forward(m, ids + SEQ - 1, 1, (size_t)SEQ - 1,
                          &dec_s, row2) != OLMOE_OK) {
            printf("FAIL: forward decode repeat returned non-OK\n");
            ++failed;
        } else if (memcmp(row2, row, V * sizeof *row) != 0) {
            printf("FAIL: forward decode is not deterministic\n");
            ++failed;
        }
    }
    /* Cross-path: final-token decode slice vs prefill final slice. This is the
     * position generation consumes, and the only one prefill produces
     * non-degenerate output for. */
    if (!failed) {
        for (size_t j = 0; j < V && !failed; ++j) {
            float want = ref_s.logits[((size_t)SEQ - 1) * V + j];
            float d = fabsf(row[j] - want);
            if (d > ATOL && d > RTOL * fabsf(want)) {
                printf("FAIL: final-token decode vs prefill lane %zu "
                       "got=%.6f want=%.6f\n", j, row[j], want);
                ++failed;
            }
        }
    }
    free(row); free(row2);
    olmoe_scratch_free(&dec_s);
    olmoe_scratch_free(&ref_s);
    if (!failed)
        printf("PASS: forward sane + final-token decode == prefill\n");
    return failed;
}

/* ---------- dispatcher -------------------------------------------------- */

static const char *resolve_model_dir(void)
{
    const char *env = getenv("OLMOE_TEST_MODEL_DIR");
    return (env && *env) ? env : OLMOE_TEST_MODEL_DIR;
}

/* Loads once, runs every check, frees, returns the cumulative failure count.
 * test_main.c sums this with the tokenizer-test failures for the exit code.
 * Re-loading the ~13 GiB model per check would be wasteful, so granular
 * checks are static and share the single loaded model. */
int test_layer_loads_and_validates(void)
{
    const char *dir = resolve_model_dir();
    printf("--- layer tests (model dir: %s) ---\n", dir);

    olmoe_model_t *m = olmoe_model_load(dir);
    if (!m) {
        printf("FAIL: olmoe_model_load returned NULL for %s\n", dir);
        return 1;
    }

    int failed = 0;
    failed += test_top_level_tensors_nonzero(m);
    failed += test_every_layer_self_attn_nonzero(m);
    failed += test_every_layer_layernorms_nonzero(m);
    failed += test_every_layer_mlp_gate_nonzero(m);
    failed += test_every_expert_tensor_nonzero(m);
    failed += test_norm_first_values_match_oracle(m);
    failed += test_lm_head_first_values_match_oracle(m);
    failed += test_expert_zero_down_values_match_oracle(m);
    failed += test_embed_lookup_matches_oracle(m);
    failed += test_forward_end_to_end_sane(m);

    olmoe_model_free(m);
    return failed;
}


