#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "olmoe/engine/engine.h"

#include "test_engine.h"

int test_engine_matmul_pass(void);
int test_engine_norm_pass(void);
int test_engine_mlp_pass(void);
int test_engine_fwd_pass(void);

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

/* olmoe_forward with seq_len==0 short-circuits before touching weights, so
 * a zeroed/empty model is acceptable input (exercises the kept zero-seq
 * contract; the real impl would otherwise dereference a NULL embed table). */
static int test_forward_zero_seq_returns_ok(void)
{
    olmoe_scratch_t s;
    olmoe_scratch_init(&s, 4);
    olmoe_model_t empty_model;
    memset(&empty_model, 0, sizeof empty_model);

    int ids[] = {1, 2, 3, 4};

    int failed = 0;
    olmoe_status_t st = olmoe_forward(&empty_model, ids, 0, &s, s.logits);
    if (st != OLMOE_OK) {
        printf("FAIL: forward(seq=0) -> %d (want OK)\n", st);
        ++failed;
    }
    if (!failed) printf("PASS: forward zero-seq returns OK\n");
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
    failed += test_forward_zero_seq_returns_ok();
    failed += test_forward_oversize_seq_returns_shape();
    failed += test_engine_matmul_pass();
    failed += test_engine_norm_pass();
    failed += test_engine_mlp_pass();
    failed += test_engine_fwd_pass();
    return failed;
}