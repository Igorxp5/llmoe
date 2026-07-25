#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "olmoe/engine/engine.h"

#include "test_engine.h"

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

/* A single NULL arg to each op must trip OLMOE_ERR_NULL rather than a
 * segfault. Covers the stub contract before any compute exists. */
static int test_null_input_returns_err(void)
{
    int failed = 0;
    olmoe_scratch_t s;
    olmoe_scratch_init(&s, 4);

    olmoe_model_t empty_model;
    memset(&empty_model, 0, sizeof empty_model);

    if (olmoe_embed_forward(NULL, (const int[]){0}, 1, s.hidden_out) != OLMOE_ERR_NULL) {
        printf("FAIL: embed_forward(model=NULL)\n"); ++failed;
    }
    if (olmoe_embed_forward(&empty_model, (const int[]){0}, 1, s.hidden_out) != OLMOE_ERR_NULL) {
        printf("FAIL: embed_forward(embed_tokens=NULL)\n"); ++failed;
    }
    if (olmoe_final_norm_forward(NULL, s.hidden_in, 1, s.hidden_out) != OLMOE_ERR_NULL) {
        printf("FAIL: final_norm_forward(w=NULL)\n"); ++failed;
    }
    if (olmoe_q_proj_forward(NULL, s.hidden_in, 1, s.q) != OLMOE_ERR_NULL) {
        printf("FAIL: q_proj_forward(attn=NULL)\n"); ++failed;
    }
    if (olmoe_mlp_gate_forward(NULL, s.hidden_in, 1, s.topk_idx, s.topk_w) != OLMOE_ERR_NULL) {
        printf("FAIL: mlp_gate_forward(w=NULL)\n"); ++failed;
    }
    if (olmoe_expert_gate_forward(NULL, s.hidden_in, 1, s.expert_in) != OLMOE_ERR_NULL) {
        printf("FAIL: expert_gate_forward(e=NULL)\n"); ++failed;
    }
    olmoe_scratch_free(&s);
    if (!failed) printf("PASS: NULL inputs return ERR_NULL\n");
    return failed;
}

/* seq_len == 0 is a legal no-op for every op (the stubs return OK without
 * touching buffers). */
static int test_empty_seq_returns_ok(void)
{
    int failed = 0;
    olmoe_scratch_t s;
    olmoe_scratch_init(&s, 4);

    if (olmoe_embed_forward(NULL, NULL, 0, NULL) != OLMOE_OK) {
        printf("FAIL: embed_forward(seq=0) should short-circuit\n"); ++failed;
    }
    if (olmoe_final_norm_forward(NULL, NULL, 0, NULL) != OLMOE_OK) {
        printf("FAIL: final_norm_forward(seq=0) should short-circuit\n"); ++failed;
    }
    if (olmoe_q_proj_forward(NULL, NULL, 0, NULL) != OLMOE_OK) {
        printf("FAIL: q_proj_forward(seq=0) should short-circuit\n"); ++failed;
    }
    olmoe_scratch_free(&s);
    if (!failed) printf("PASS: empty seq short-circuits to OK\n");
    return failed;
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

/* ---------- dispatcher --------------------------------------------------- */

int test_engine_stubs_pass(void)
{
    int failed = 0;
    failed += test_scratch_init_free_roundtrip();
    failed += test_scratch_init_null_returns_err();
    failed += test_scratch_free_null_is_safe();
    failed += test_null_input_returns_err();
    failed += test_empty_seq_returns_ok();
    failed += test_forward_stub_returns_ok();
    failed += test_forward_oversize_seq_returns_shape();
    return failed;
}