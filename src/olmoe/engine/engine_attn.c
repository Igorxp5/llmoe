/* Attention projections (1:1 with OLMOE_KIND_Q_PROJ / K_PROJ / V_PROJ /
 * O_PROJ). Each is x @ <weight>^T -> out[seq, hidden]. The attention
 * compute itself (RoPE + sdpa) is a future op living in this file. */

#include "olmoe/engine/engine.h"
#include "kernels/cpu_matmul.h"

/* Shared by the four attn projections (q/k/v/o) so the matmul call site is
 * written once, not copy-pasted four times. */
static inline void attn_proj_forward(const olmoe_bf16_t * restrict w,
                              const olmoe_act_t * restrict x, size_t seq_len,
                              olmoe_act_t * restrict out)
{
    cpu_matmul_bf16(out, x, w, seq_len, OLMOE_HIDDEN, OLMOE_HIDDEN);
}

/* Fused q/k/v path: the three projections read the same `x` and write
 * disjoint buffers, so the whole triple is one OpenMP fork/join instead of
 * the three that separate cpu_matmul_bf16 calls would spawn (48 fewer per
 * token at OLMOE_N_LAYERS=16). Each output element is still computed by one
 * thread with the same dot-product code, so the result is bit-identical to
 * three independent matmuls. o_proj must stay separate: it consumes the
 * SDPA output, not `x`. */
void olmoe_qkv_proj_forward(const olmoe_self_attn_t * restrict a,
                            const olmoe_act_t * restrict x, size_t seq_len,
                            olmoe_act_t * restrict q, olmoe_act_t * restrict k,
                            olmoe_act_t * restrict v)
{
    const size_t hidden = (size_t)OLMOE_HIDDEN;
    const olmoe_bf16_t * const w[3]   = { a->q_proj, a->k_proj, a->v_proj };
    olmoe_act_t * const out[3]        = { q, k, v };
    #pragma omp parallel for schedule(static) collapse(3)
    for (size_t m = 0; m < 3; ++m)
        for (size_t i = 0; i < seq_len; ++i)
            for (size_t j = 0; j < hidden; ++j)
                out[m][i * hidden + j] =
                    cpu_matmul_dot_bf16(x + i * hidden, w[m] + j * hidden,
                                        hidden);
}

void olmoe_q_proj_forward(const olmoe_self_attn_t * restrict a,
                          const olmoe_act_t * restrict x, size_t seq_len,
                          olmoe_act_t * restrict out)
{
    attn_proj_forward(a->q_proj, x, seq_len, out);
}

void olmoe_k_proj_forward(const olmoe_self_attn_t * restrict a,
                          const olmoe_act_t * restrict x, size_t seq_len,
                          olmoe_act_t * restrict out)
{
    attn_proj_forward(a->k_proj, x, seq_len, out);
}

void olmoe_v_proj_forward(const olmoe_self_attn_t * restrict a,
                          const olmoe_act_t * restrict x, size_t seq_len,
                          olmoe_act_t * restrict out)
{
    attn_proj_forward(a->v_proj, x, seq_len, out);
}

void olmoe_o_proj_forward(const olmoe_self_attn_t * restrict a,
                          const olmoe_act_t * restrict x, size_t seq_len,
                          olmoe_act_t * restrict out)
{
    attn_proj_forward(a->o_proj, x, seq_len, out);
}
