/* Attention projections (1:1 with OLMOE_KIND_Q_PROJ / K_PROJ / V_PROJ /
 * O_PROJ). Stubs only; see docs/engine_module.md. The attention compute
 * itself (RoPE + sdpa) is a future op living in this file. */

#include "olmoe/engine/engine.h"

void olmoe_q_proj_forward(const olmoe_self_attn_t *a,
                          const olmoe_act_t *x, size_t seq_len,
                          olmoe_act_t *out)
{
    /* TODO: x @ q_proj^T -> out[seq, hidden]. */
}

void olmoe_k_proj_forward(const olmoe_self_attn_t *a,
                          const olmoe_act_t *x, size_t seq_len,
                          olmoe_act_t *out)
{
    /* TODO: x @ k_proj^T -> out[seq, hidden]. */
}

void olmoe_v_proj_forward(const olmoe_self_attn_t *a,
                          const olmoe_act_t *x, size_t seq_len,
                          olmoe_act_t *out)
{
    /* TODO: x @ v_proj^T -> out[seq, hidden]. */
}

void olmoe_o_proj_forward(const olmoe_self_attn_t *a,
                          const olmoe_act_t *x, size_t seq_len,
                          olmoe_act_t *out)
{
    /* TODO: attn_out @ o_proj^T -> out[seq, hidden]. */
}