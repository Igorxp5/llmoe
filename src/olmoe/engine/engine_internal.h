#ifndef OLMOE_ENGINE_INTERNAL_H
#define OLMOE_ENGINE_INTERNAL_H

/* Shared internal helpers for the engine sub-modules. Not part of the
 * public API; only `engine_*.c` files under src/olmoe/engine include this.
 * Public callers use engine.h. */

#include "olmoe/engine/engine.h"

/* Compute n * elemsz and require the result to fit in size_t. Returns 0 on
 * overflow. Used by olmoe_scratch_init before each malloc. Header-inline so
 * the engine sub-modules share one definition without a util .c. */
static inline size_t olmoe_engine_safe_array_size(size_t n, size_t elemsz)
{
    if (n == 0 || elemsz == 0) {
        return 0;
    }
    if (n > (size_t)-1 / elemsz) {
        return 0;
    }
    return n * elemsz;
}

#endif /* OLMOE_ENGINE_INTERNAL_H */
