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