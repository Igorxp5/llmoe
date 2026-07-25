#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "olmoe/layers/layers.h"

#include "test_layer.h"

#ifndef OLMOE_TEST_MODEL_DIR
#define OLMOE_TEST_MODEL_DIR "models/OLMoE-1B-7B-0924-Instruct"
#endif

/* ---------- checks ------------------------------------------------------ */

/* Tests assert the baked dimension constants agree with what the loader
 * actually built. A mismatch would mean layout.inc drifted from the
 * expected config: catch it here instead of silently reading past buffers. */
static int test_dims_match_config(const olmoe_model_t *m)
{
    int failed = 0;
    if (m->n_layers != OLMOE_N_LAYERS) {
        printf("FAIL: n_layers expected %d got %zu\n", OLMOE_N_LAYERS, m->n_layers);
        ++failed;
    }
    if (m->n_bufs != OLMOE_N_TOTAL_TENSORS) {
        printf("FAIL: n_bufs expected %d got %zu\n",
               OLMOE_N_TOTAL_TENSORS, m->n_bufs);
        ++failed;
    }
    if (!failed)
        printf("PASS: dims (layers=%d, total_tensors=%d)\n",
               OLMOE_N_LAYERS, OLMOE_N_TOTAL_TENSORS);
    return failed;
}

static int test_top_level_tensors_nonnull(const olmoe_model_t *m)
{
    int failed = 0;
    if (!m->embed_tokens) { printf("FAIL: embed_tokens NULL\n"); ++failed; }
    if (!m->lm_head)       { printf("FAIL: lm_head NULL\n");       ++failed; }
    if (!m->norm)          { printf("FAIL: norm NULL\n");          ++failed; }
    if (!failed)
        printf("PASS: top-level tensors allocated\n");
    return failed;
}

static int test_self_attn_nonnull_for_layer(const olmoe_model_t *m, int L)
{
    const olmoe_self_attn_t *sa = &m->layers[L].self_attn;
    int failed = 0;
    if (!sa->q_proj) { printf("FAIL: layer %d q_proj NULL\n", L); ++failed; }
    if (!sa->k_proj) { printf("FAIL: layer %d k_proj NULL\n", L); ++failed; }
    if (!sa->v_proj) { printf("FAIL: layer %d v_proj NULL\n", L); ++failed; }
    if (!sa->o_proj) { printf("FAIL: layer %d o_proj NULL\n", L); ++failed; }
    if (!sa->q_norm) { printf("FAIL: layer %d q_norm NULL\n", L); ++failed; }
    if (!sa->k_norm) { printf("FAIL: layer %d k_norm NULL\n", L); ++failed; }
    return failed;
}

static int test_every_layer_self_attn_nonnull(const olmoe_model_t *m)
{
    int failed = 0;
    for (int L = 0; L < OLMOE_N_LAYERS; ++L)
        failed += test_self_attn_nonnull_for_layer(m, L);
    if (!failed)
        printf("PASS: all %d layers' self_attn tensors allocated\n",
               OLMOE_N_LAYERS);
    return failed;
}

static int test_every_layer_layernorms_nonnull(const olmoe_model_t *m)
{
    int failed = 0;
    for (int L = 0; L < OLMOE_N_LAYERS; ++L) {
        if (!m->layers[L].input_layernorm) {
            printf("FAIL: layer %d input_layernorm NULL\n", L); ++failed;
        }
        if (!m->layers[L].post_attention_layernorm) {
            printf("FAIL: layer %d post_attention_layernorm NULL\n", L); ++failed;
        }
    }
    if (!failed)
        printf("PASS: all %d layers' layernorms allocated\n",
               OLMOE_N_LAYERS);
    return failed;
}

static int test_every_layer_mlp_gate_nonnull(const olmoe_model_t *m)
{
    int failed = 0;
    for (int L = 0; L < OLMOE_N_LAYERS; ++L) {
        if (!m->layers[L].mlp_gate) {
            printf("FAIL: layer %d mlp_gate (router) NULL\n", L); ++failed;
        }
    }
    if (!failed)
        printf("PASS: all %d layers' mlp_gate (router) allocated\n",
               OLMOE_N_LAYERS);
    return failed;
}

/* 16 layers * 64 experts * 3 weights = 3072 expert buffers. Report only
 * failures so a green run stays a single line. */
static int test_every_expert_tensor_nonnull(const olmoe_model_t *m)
{
    int failed = 0;
    for (int L = 0; L < OLMOE_N_LAYERS; ++L) {
        for (int E = 0; E < OLMOE_N_EXPERTS; ++E) {
            const olmoe_expert_t *ex = &m->layers[L].experts[E];
            if (!ex->gate_proj) {
                printf("FAIL: layer %d expert %d gate_proj NULL\n", L, E); ++failed;
            }
            if (!ex->up_proj) {
                printf("FAIL: layer %d expert %d up_proj NULL\n", L, E); ++failed;
            }
            if (!ex->down_proj) {
                printf("FAIL: layer %d expert %d down_proj NULL\n", L, E); ++failed;
            }
        }
    }
    if (!failed)
        printf("PASS: all %d layers x %d experts x 3 weights allocated\n",
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
    failed += test_dims_match_config(m);
    failed += test_top_level_tensors_nonnull(m);
    failed += test_every_layer_self_attn_nonnull(m);
    failed += test_every_layer_layernorms_nonnull(m);
    failed += test_every_layer_mlp_gate_nonnull(m);
    failed += test_every_expert_tensor_nonnull(m);
    failed += test_norm_first_values_match_oracle(m);
    failed += test_lm_head_first_values_match_oracle(m);
    failed += test_expert_zero_down_values_match_oracle(m);

    olmoe_model_free(m);
    return failed;
}
