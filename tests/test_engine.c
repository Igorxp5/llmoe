#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "olmoe/engine/engine.h"

#include "test_engine.h"

/* Forward declarations for the module dispatchers this file aggregates. */
int test_engine_matmul_pass(void);
int test_engine_norm_pass(void);
int test_engine_mlp_pass(void);
int test_engine_fwd_pass(void);
int test_engine_argmax_pass(void);

/* ---------- checks ------------------------------------------------------ */

/* The per-kind leaf forwards write `seq_len` rows and are documented for
 * seq_len >= 1. A zero-length sequence must therefore be a no-op: dense
 * sentinel arrays must emerge untouched. The two model-backed forwards
 * (embed, lm_head) are excluded — embed needs a real model and lm_head
 * indexes row (seq_len-1), overflow for zero. NULL pointers are not part of
 * the void-return contract, so only the shape guard is asserted. */
static int test_leaf_forwards_zero_seq_no_op(void)
{
    static olmoe_self_attn_t attn;
    static olmoe_expert_t expert;
    enum { N = 512 };
    olmoe_bf16_t w[OLMOE_HIDDEN];
    olmoe_act_t x[N], out[N];
    int topk_idx[OLMOE_N_EXPERTS_PER_TOK];
    olmoe_act_t topk_w[OLMOE_N_EXPERTS_PER_TOK];
    const float sentinel = 987.654f;
    /* The weight buffer is never read (seq_len is 0), but zero it so GCC's
     * maybe-uninitialized analysis stays quiet. */
    memset(w, 0, sizeof w);
    for (size_t i = 0; i < N; ++i) { x[i] = 1.0f; out[i] = sentinel; }

    int failed = 0;
    /* RMSNorm forwards (out-of-place and in-place variants). */
    olmoe_input_ln_forward(w, x, 0, out);
    olmoe_final_norm_forward(w, x, 0, out);
    olmoe_post_ln_forward(w, x, 0, out);
    if (out[0] != sentinel) { printf("FAIL: norm seq=0 wrote output\n"); ++failed; }
    olmoe_q_norm_forward(w, x, 0);
    olmoe_k_norm_forward(w, x, 0);
    if (x[0] != 1.0f) { printf("FAIL: q/k_norm seq=0 wrote input\n"); ++failed; }

    /* Attention projections. */
    olmoe_q_proj_forward(&attn, x, 0, out);
    olmoe_k_proj_forward(&attn, x, 0, out);
    olmoe_v_proj_forward(&attn, x, 0, out);
    olmoe_o_proj_forward(&attn, x, 0, out);

    /* MoE router + expert matmuls. */
    olmoe_mlp_gate_forward(w, x, 0, topk_idx, topk_w);
    olmoe_expert_gate_forward(&expert, x, 0, out);
    olmoe_expert_up_forward(&expert, x, 0, out);
    olmoe_expert_down_forward(&expert, x, 0, out);

    if (out[0] != sentinel) { printf("FAIL: a leaf forward seq=0 wrote output\n"); ++failed; }
    if (!failed) printf("PASS: leaf forwards seq_len=0 are no-ops\n");
    return failed;
}

/* init + free must roundtrip without leaking or crashing for a typical
 * short sequence. */
static int test_scratch_init_free_roundtrip(void)
{
    olmoe_scratch_t s;
    olmoe_status_t st = olmoe_scratch_init(&s, 16, 0);
    if (st != OLMOE_OK) {
        printf("FAIL: scratch_init(16,0) -> %d (want OK)\n", st);
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
    olmoe_status_t st = olmoe_scratch_init(NULL, 8, 0);
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
    olmoe_scratch_init(&s, 4, 0);
    /* olmoe_model_t is a ~13 GiB fixed-size struct; the all-zero weights
     * must live on the heap, not the stack (a stack allocation would smash
     * the guard page and SIGSEGV). */
    olmoe_model_t *empty_model = calloc(1, sizeof *empty_model);
    if (!empty_model) {
        printf("FAIL: forward zero-seq model calloc OOM\n");
        olmoe_scratch_free(&s);
        return 1;
    }

    int ids[] = {1, 2, 3, 4};

    int failed = 0;
    olmoe_status_t st = olmoe_forward(empty_model, ids, 0, 0, &s, s.logits);
    if (st != OLMOE_OK) {
        printf("FAIL: forward(seq=0) -> %d (want OK)\n", st);
        ++failed;
    }
    if (!failed) printf("PASS: forward zero-seq returns OK\n");
    free(empty_model);
    olmoe_scratch_free(&s);
    return failed;
}

/* forward must reject a seq_len larger than the scratch was sized for. */
static int test_forward_oversize_seq_returns_shape(void)
{
    olmoe_scratch_t s;
    olmoe_scratch_init(&s, 4, 0);
    /* ~13 GiB all-zero weights on the heap (stack would SIGSEGV). */
    olmoe_model_t *empty_model = calloc(1, sizeof *empty_model);
    if (!empty_model) {
        printf("FAIL: forward oversize model calloc OOM\n");
        olmoe_scratch_free(&s);
        return 1;
    }
    int ids[] = {1, 2, 3, 4, 5, 6, 7, 8};

    int failed = 0;
    olmoe_status_t st = olmoe_forward(empty_model, ids, 8, 0, &s, s.logits);
    if (st != OLMOE_ERR_SHAPE) {
        printf("FAIL: forward(seq=8 on scratch=4) -> %d (want ERR_SHAPE)\n", st);
        ++failed;
    } else {
        printf("PASS: forward oversize seq rejected with ERR_SHAPE\n");
    }
    free(empty_model);
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
    failed += test_leaf_forwards_zero_seq_no_op();
    failed += test_forward_zero_seq_returns_ok();
    failed += test_forward_oversize_seq_returns_shape();
    failed += test_engine_matmul_pass();
    failed += test_engine_norm_pass();
    failed += test_engine_mlp_pass();
    failed += test_engine_fwd_pass();
    failed += test_engine_argmax_pass();
    return failed;
}