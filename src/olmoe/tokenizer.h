#ifndef LLMOE_OLMOE_TOKNIZER_H
#define LLMOE_OLMOE_TOKNIZER_H

#include <stdint.h>

#define LLMOE_OLMOE_MAX_OUTPUT_TOKENS 100000
#define LLMOE_OLMOE_TOKENIZER_VOCAB_SIZE 50280

void olmoe_tokenize(uint8_t* in, uint16_t *out);

#endif