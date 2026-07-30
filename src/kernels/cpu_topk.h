#ifndef KERNELS_CPU_TOPK_H
#define KERNELS_CPU_TOPK_H

#include <math.h>
#include <stdbool.h>

#include "kernels/kernels.h"

/* Find the highest-scoring unused index. `used_mask[r] == true` means
 * index r has already been selected. Returns n if all are used. */
static inline size_t cpu_topk_pick_next(const float *scores,
                                        size_t num_scores,
                                        const bool *used_mask)
{
    size_t best = num_scores;
    float best_v = -INFINITY;
    for (size_t r = 0; r < num_scores; ++r) {
        if (used_mask[r]) continue;
        if (best == num_scores || scores[r] > best_v) {
            best = r;
            best_v = scores[r];
        }
    }
    return best;
}

/* Greedy top-k descending: select the `top_k` highest scores from `scores`
 * (length `num_scores`). Writes indices to `indices_out` and values to
 * `values_out`, both of length `top_k`. num_scores must be <= 64 (stack
 * allocation for the used-bitmask). */
static inline void cpu_topk_desc(const float *scores, size_t num_scores,
                                 size_t top_k,
                                 int *indices_out, float *values_out)
{
    bool used[64] = { false };
    for (size_t r = 0; r < top_k; ++r) {
        size_t best = cpu_topk_pick_next(scores, num_scores, used);
        used[best] = true;
        indices_out[r] = (int)best;
        values_out[r] = scores[best];
    }
}

#endif /* KERNELS_CPU_TOPK_H */
