#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "olmoe/engine/engine.h"
#include "olmoe/layers/layers.h"

#include "test_engine_helpers.h"
#include "test_layer.h"

#ifndef OLMOE_TEST_MODEL_DIR
#define OLMOE_TEST_MODEL_DIR "models/OLMoE-1B-7B-0924-Instruct"
#endif

/* ---------- checks ------------------------------------------------------ */

/* Compare `dst` (a loaded weight buffer) against an oracle row of the first
 * OLMOE_ORACLE_N_VALUES BF16 values. Reports one FAIL line per mismatch. */
static int memcmp_oracle_row(const void *dst, const uint16_t *oracle,
                             const char *what, int L, int E)
{
    if (memcmp(dst, oracle, OLMOE_ORACLE_N_VALUES * sizeof(uint16_t)) != 0) {
        if (E >= 0)
            printf("FAIL: %s layer %d expert %d first %d values != oracle\n",
                   what, L, E, OLMOE_ORACLE_N_VALUES);
        else
            printf("FAIL: %s layer %d first %d values != oracle\n",
                   what, L, OLMOE_ORACLE_N_VALUES);
        return 1;
    }
    return 0;
}

/* Every layer-scoped weight buffer must reproduce its per-layer oracle table
 * at the leading values. This replaces the old "is it nonzero" probe: the
 * loader is expected to fill these buffers with exact BF16 bytes, so a real
 * value comparison, not a zero check, validates the load. */
static int test_every_layer_tensors_match_oracle(const olmoe_model_t *m)
{
    int failed = 0;
    for (int L = 0; L < OLMOE_N_LAYERS; ++L) {
        const olmoe_self_attn_t *sa = &m->layers[L].self_attn;
        failed += memcmp_oracle_row(sa->q_proj, OLMOE_ORACLE_Q_PROJ[L],
                                    "q_proj", L, -1);
        failed += memcmp_oracle_row(sa->k_proj, OLMOE_ORACLE_K_PROJ[L],
                                    "k_proj", L, -1);
        failed += memcmp_oracle_row(sa->v_proj, OLMOE_ORACLE_V_PROJ[L],
                                    "v_proj", L, -1);
        failed += memcmp_oracle_row(sa->o_proj, OLMOE_ORACLE_O_PROJ[L],
                                    "o_proj", L, -1);
        failed += memcmp_oracle_row(sa->q_norm, OLMOE_ORACLE_Q_NORM[L],
                                    "q_norm", L, -1);
        failed += memcmp_oracle_row(sa->k_norm, OLMOE_ORACLE_K_NORM[L],
                                    "k_norm", L, -1);
        failed += memcmp_oracle_row(m->layers[L].input_layernorm,
                                    OLMOE_ORACLE_INPUT_LN[L],
                                    "input_layernorm", L, -1);
        failed += memcmp_oracle_row(m->layers[L].post_attention_layernorm,
                                    OLMOE_ORACLE_POST_LN[L],
                                    "post_attention_layernorm", L, -1);
        failed += memcmp_oracle_row(m->layers[L].mlp_gate,
                                    OLMOE_ORACLE_MLP_GATE[L],
                                    "mlp_gate", L, -1);
    }
    if (!failed)
        printf("PASS: all %d layers' weights match per-layer oracle\n",
               OLMOE_N_LAYERS);
    return failed;
}

/* Every expert projection buffer is checked against its own oracle table:
 * 16 layers x 64 experts x 3 weights, all compared to the real BF16 values.
 * Replaces the predecessor test that confirmed only "nonzero". */
static int test_every_expert_tensor_matches_oracle(const olmoe_model_t *m)
{
    int failed = 0;
    for (int L = 0; L < OLMOE_N_LAYERS; ++L) {
        for (int E = 0; E < OLMOE_N_EXPERTS; ++E) {
            const olmoe_expert_t *ex = &m->layers[L].experts[E];
            failed += memcmp_oracle_row(ex->gate_proj,
                                        OLMOE_ORACLE_EXPERT_GATE[L][E],
                                        "expert gate_proj", L, E);
            failed += memcmp_oracle_row(ex->up_proj,
                                        OLMOE_ORACLE_EXPERT_UP[L][E],
                                        "expert up_proj", L, E);
            failed += memcmp_oracle_row(ex->down_proj,
                                        OLMOE_ORACLE_EXPERT_DOWN[L][E],
                                        "expert down_proj", L, E);
        }
    }
    if (!failed)
        printf("PASS: all %d layers x %d experts x 3 weights match oracle\n",
               OLMOE_N_LAYERS, OLMOE_N_EXPERTS);
    return failed;
}

static int test_norm_first_matches_oracle(const olmoe_model_t *m)
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

static int test_lm_head_first_matches_oracle(const olmoe_model_t *m)
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

static int test_embed_first_values_match_oracle(const olmoe_model_t *m)
{
    if (memcmp(m->embed_tokens, OLMOE_ORACLE_MODEL_EMBED,
               sizeof OLMOE_ORACLE_MODEL_EMBED) != 0) {
        printf("FAIL: model.embed_tokens.weight first N values != oracle\n");
        return 1;
    }
    printf("PASS: model.embed_tokens.weight first %d BF16 values match oracle\n",
           OLMOE_ORACLE_N_VALUES);
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

/* olmoe_lm_head_forward: real-dim call against the loaded head on a 2-row
 * input. It must compute only the last (seq_len-1) row into logits_out and
 * leave earlier rows untouched; the last row is compared to the scalar BF16
 * reference over all OLMOE_VOCAB lanes. */
static int test_lm_head_forward_matches_scalar(const olmoe_model_t *m)
{
    enum { SEQ = 2 };
    const size_t V = (size_t)OLMOE_VOCAB, H = (size_t)OLMOE_HIDDEN;
    olmoe_act_t *x = malloc(SEQ * H * sizeof *x);
    olmoe_act_t *got = malloc(SEQ * V * sizeof *got);
    if (!x || !got) {
        printf("FAIL: lm_head malloc OOM\n");
        free(x); free(got);
        return 1;
    }
for (size_t i = 0; i < SEQ * H; ++i)
        x[i] = (float)(int)(i % 7) / 100.0f - 0.03f;
    const float sentinel = 123.456f;
    for (size_t i = 0; i < SEQ * V; ++i) got[i] = sentinel;

    olmoe_lm_head_forward(m, x, SEQ, got);

    int failed = 0;
    /* Earlier row must be untouched. */
    for (size_t j = 0; j < V; ++j) {
        if (got[j] != sentinel) {
            printf("FAIL: lm_head wrote into row 0 at lane %zu\n", j);
            failed = 1;
            break;
        }
    }
    /* Last row matches scalar dot product. */
    for (size_t j = 0; j < V && !failed; ++j) {
        float want = scalar_dot_bf16(x + (SEQ - 1) * H, m->lm_head + j * H, H);
        float d = fabsf(got[(SEQ - 1) * V + j] - want);
        if (d > 1e-5f && d > 1e-4f * fabsf(want)) {
            printf("FAIL: lm_head lane %zu got=%.6f want=%.6f\n",
                   j, got[(SEQ - 1) * V + j], want);
            ++failed;
            if (failed == 8) break;
        }
    }
    free(x); free(got);
    if (!failed)
        printf("PASS: lm_head_forward computes last-row logits vs scalar\n");
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

    const olmoe_model_t *m = olmoe_model_load(dir);
    if (!m) {
        printf("FAIL: olmoe_model_load returned NULL for %s\n", dir);
        return 1;
    }

    int failed = 0;
    failed += test_embed_first_values_match_oracle(m);
    failed += test_norm_first_matches_oracle(m);
    failed += test_lm_head_first_matches_oracle(m);
    failed += test_every_layer_tensors_match_oracle(m);
    failed += test_every_expert_tensor_matches_oracle(m);
    failed += test_embed_lookup_matches_oracle(m);
    failed += test_lm_head_forward_matches_scalar(m);
    failed += test_forward_end_to_end_sane(m);

    olmoe_model_free(m);
    return failed;
}


