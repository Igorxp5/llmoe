#ifndef OLMOE_KERNEL_CPU_TOPK_H
#define OLMOE_KERNEL_CPU_TOPK_H

#include <stdbool.h>

#include "olmoe/engine/engine_internal.h"

/* Top-K selection over a score row: writes the K largest scores (descending)
 * into val_out and their indices into idx_out. Ties broken by SMALLEST index
 * so the router is deterministic regardless of FP32 summation order.
 *
 * Example:
 *     int idx[OLMOE_N_EXPERTS_PER_TOK];
 *     float w[OLMOE_N_EXPERTS_PER_TOK];
 *     cpu_topk_desc(probs, OLMOE_N_EXPERTS, OLMOE_N_EXPERTS_PER_TOK, idx, w);
 */
static inline size_t cpu_topk_pick_next(const float *scores, size_t n,
                                        const bool *used)
{
    size_t best = n;
    float best_v = -INFINITY;
    for (size_t r = 0; r < n; ++r) {
        if (used[r]) continue;
        if (best == n || scores[r] > best_v) {
            best = r;
            best_v = scores[r];
        }
    }
    return best;
}

static inline void cpu_topk_desc(const float *scores, size_t n, size_t k,
                                 int *idx_out, float *val_out)
{
    bool used[64] = { false };
    for (size_t r = 0; r < k; ++r) {
        size_t best = cpu_topk_pick_next(scores, n, used);
        used[best] = true;
        idx_out[r] = (int)best;
        val_out[r] = scores[best];
    }
}

#endif /* OLMOE_KERNEL_CPU_TOPK_H */