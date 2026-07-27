#ifndef KERNELS_CPU_SILU_H
#define KERNELS_CPU_SILU_H

#include <math.h>

#include "kernels/kernels.h"

static inline float cpu_silu(float x)
{
    return x / (1.0f + expf(-x));
}

#endif /* KERNELS_CPU_SILU_H */
