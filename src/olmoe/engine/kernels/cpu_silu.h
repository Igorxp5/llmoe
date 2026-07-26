#ifndef OLMOE_KERNEL_CPU_SILU_H
#define OLMOE_KERNEL_CPU_SILU_H

#include <math.h>

#include "olmoe/engine/engine_internal.h"

/* SiLU (swish) activation: x / (1 + exp(-x)). Scalar; callers loop over
 * the inter dimension. */
static inline float cpu_silu(float x)
{
    return x / (1.0f + expf(-x));
}

#endif /* OLMOE_KERNEL_CPU_SILU_H */