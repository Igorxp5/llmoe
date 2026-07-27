#ifndef KERNELS_CPU_TOPK_H
#define KERNELS_CPU_TOPK_H

#include <math.h>
#include <stdbool.h>

#include "kernels/kernels.h"

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

#endif /* KERNELS_CPU_TOPK_H */
