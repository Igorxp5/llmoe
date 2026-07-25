#ifndef LLMOE_NORMALIZER_H
#define LLMOE_NORMALIZER_H

#include <stdbool.h>
#include <stdint.h>

typedef enum {
    NFC
} tkz_normalizer_t;

bool tkz_normalizer(tkz_normalizer_t type, uint8_t* in, uint8_t* out);

#endif
