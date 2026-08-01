/* MoE router + expert ops (1:1 with OLMOE_KIND_MLP_GATE /
 * EXPERT_GATE / EXPERT_UP / EXPERT_DOWN). The router (mlp_gate) is
 * implemented; expert gate/up/down are stubs for later agents. */

#include "olmoe/engine/engine.h"
#include "kernels/cpu_matmul.h"
#include "kernels/cpu_softmax.h"
#include "kernels/cpu_topk.h"

/* OLMoE renormalizes the top-K softmax weights over the selected experts
 * so they sum to 1 (HF OlmoeSparseMoE semantics). */
static inline void renorm_to_sum_one(olmoe_act_t * restrict w, size_t k)
{
    float sum = 0.0f;
    for (size_t r = 0; r < k; ++r) sum += w[r];
    for (size_t r = 0; r < k; ++r) w[r] /= sum;
}

/* Route a single token's logits row: softmax over all experts, pick top-K.
 * OLMoE config has norm_topk_prob=false, so the selected weights are the raw
 * softmax probabilities (not renormalized to sum to 1). */
static inline void route_one_token(const float * restrict logits_row,
                            int * restrict idx, olmoe_act_t * restrict w,
                            size_t n_experts, size_t k)
{
    float probs[64];
    cpu_softmax(probs, logits_row, n_experts);
    cpu_topk_desc(probs, n_experts, k, idx, w);
}

/* Router forward: expert logits = x @ mlp_gate^T, then softmax+top-K per
 * token. Logits are computed one token at a time into a fixed stack buffer:
 * the engine never mallocs activations (scratch is caller-owned), and a
 * per-row dot is bit-identical to the batched matmul it replaces — each
 * output lane is an independent k-reduction with a fixed accumulation order. */
void olmoe_mlp_gate_forward(const olmoe_bf16_t * restrict w,
                            const olmoe_act_t * restrict x, size_t seq_len,
                            int * restrict topk_idx,
                            olmoe_act_t * restrict topk_w)
{
    size_t n = (size_t)OLMOE_N_EXPERTS, k = (size_t)OLMOE_N_EXPERTS_PER_TOK;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < seq_len; ++i) {
        float logits[OLMOE_N_EXPERTS];
        for (size_t j = 0; j < n; ++j)
            logits[j] = cpu_matmul_dot_bf16(x + i * (size_t)OLMOE_HIDDEN,
                                            w + j * (size_t)OLMOE_HIDDEN,
                                            (size_t)OLMOE_HIDDEN);
        route_one_token(logits, topk_idx + i * k, topk_w + i * k, n, k);
    }
}

/* Shared by the three expert matmuls: gate/up project x [tok, hidden]
 * to [tok, inter], down projects x [tok, inter] back to [tok, hidden].
 * The dim roles are caller-supplied so each op names its own weight rows. */
static inline void expert_proj_forward(const olmoe_bf16_t * restrict w,
                                const olmoe_act_t * restrict x,
                                size_t n_tokens, size_t n_out, size_t k_in,
                                olmoe_act_t * restrict out)
{
    cpu_matmul_bf16(out, x, w, n_tokens, n_out, k_in);
}

void olmoe_expert_gate_forward(const olmoe_expert_t * restrict e,
                               const olmoe_act_t * restrict x, size_t n_tokens,
                               olmoe_act_t * restrict out)
{
    expert_proj_forward(e->gate_proj, x, n_tokens,
                        (size_t)OLMOE_INTER, (size_t)OLMOE_HIDDEN, out);
}

void olmoe_expert_up_forward(const olmoe_expert_t * restrict e,
                              const olmoe_act_t * restrict x, size_t n_tokens,
                              olmoe_act_t * restrict out)
{
    expert_proj_forward(e->up_proj, x, n_tokens,
                        (size_t)OLMOE_INTER, (size_t)OLMOE_HIDDEN, out);
}

void olmoe_expert_down_forward(const olmoe_expert_t * restrict e,
                               const olmoe_act_t * restrict x, size_t n_tokens,
                               olmoe_act_t * restrict out)
{
    expert_proj_forward(e->down_proj, x, n_tokens,
                        (size_t)OLMOE_HIDDEN, (size_t)OLMOE_INTER, out);
}
