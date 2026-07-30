#ifndef KERNELS_CPU_SILU_H
#define KERNELS_CPU_SILU_H

#include <math.h>

#include "kernels/kernels.h"

/* SiLU (Swish) activation: x * sigmoid(x) = x / (1 + exp(-x)) */
static inline float cpu_silu(float x)
{
    return x / (1.0f + expf(-x));
}

#endif /* KERNELS_CPU_SILU_H */
