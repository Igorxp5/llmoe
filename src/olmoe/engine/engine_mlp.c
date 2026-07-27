/* MoE router + expert ops (1:1 with OLMOE_KIND_MLP_GATE /
 * EXPERT_GATE / EXPERT_UP / EXPERT_DOWN). The router (mlp_gate) is
 * implemented; expert gate/up/down are stubs for later agents. */

#include <stdlib.h>

#include "olmoe/engine/engine.h"
#include "kernels/cpu_matmul.h"
#include "kernels/cpu_softmax.h"
#include "kernels/cpu_topk.h"

/* OLMoE renormalizes the top-K softmax weights over the selected experts
 * so they sum to 1 (HF OlmoeSparseMoE semantics). */
static void renorm_to_sum_one(olmoe_act_t *w, size_t k)
{
    float sum = 0.0f;
    for (size_t r = 0; r < k; ++r) sum += w[r];
    for (size_t r = 0; r < k; ++r) w[r] /= sum;
}

/* Route a single token's logits row: softmax over all experts, pick top-K,
 * renormalize the K weights to sum to 1. */
static void route_one_token(const float *logits_row, int *idx,
                            olmoe_act_t *w, size_t n_experts, size_t k)
{
    float probs[64];
    cpu_softmax(probs, logits_row, n_experts);
    cpu_topk_desc(probs, n_experts, k, idx, w);
    renorm_to_sum_one(w, k);
}

void olmoe_mlp_gate_forward(const olmoe_bf16_t *w,
                            const olmoe_act_t *x, size_t seq_len,
                            int *topk_idx, olmoe_act_t *topk_w)
{
    size_t n = (size_t)OLMOE_N_EXPERTS, k = (size_t)OLMOE_N_EXPERTS_PER_TOK;
    float *logits = malloc(seq_len * n * sizeof(float));
    cpu_matmul_bf16(logits, x, w, seq_len, n, (size_t)OLMOE_HIDDEN);
    for (size_t i = 0; i < seq_len; ++i)
        route_one_token(logits + i * n, topk_idx + i * k,
                        topk_w + i * k, n, k);
    free(logits);
}

/* Shared by the three expert matmuls: gate/up project x [tok, hidden]
 * to [tok, inter], down projects x [tok, inter] back to [tok, hidden].
 * The dim roles are caller-supplied so each op names its own weight rows. */
static void expert_proj_forward(const olmoe_bf16_t *w, const olmoe_act_t *x,
                                size_t n_tokens, size_t n_out, size_t k_in,
                                olmoe_act_t *out)
{
    cpu_matmul_bf16(out, x, w, n_tokens, n_out, k_in);
}

void olmoe_expert_gate_forward(const olmoe_expert_t *e,
                               const olmoe_act_t *x, size_t n_tokens,
                               olmoe_act_t *out)
{
    expert_proj_forward(e->gate_proj, x, n_tokens,
                        (size_t)OLMOE_INTER, (size_t)OLMOE_HIDDEN, out);
}

void olmoe_expert_up_forward(const olmoe_expert_t *e,
                              const olmoe_act_t *x, size_t n_tokens,
                              olmoe_act_t *out)
{
    expert_proj_forward(e->up_proj, x, n_tokens,
                        (size_t)OLMOE_INTER, (size_t)OLMOE_HIDDEN, out);
}

void olmoe_expert_down_forward(const olmoe_expert_t *e,
                               const olmoe_act_t *x, size_t n_tokens,
                               olmoe_act_t *out)
{
    expert_proj_forward(e->down_proj, x, n_tokens,
                        (size_t)OLMOE_HIDDEN, (size_t)OLMOE_INTER, out);
}
