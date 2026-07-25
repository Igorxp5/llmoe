#ifndef OLMOE_TOKENIZER_H
#define OLMOE_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

/* Token id type. OLMoE-1B-7B-0924-Instruct vocab fits in 16 bits, but we use
 * uint32_t to keep the public buffer representation fixed-width regardless
 * of tokenizer. */
typedef uint32_t olmoe_token_id_t;

/* Tokenize a NUL-terminated UTF-8 string.
 *
 * Returns the number of token ids needed for `text`.  When `out` is non-NULL
 * and `cap` is large enough to hold the result, also fills `out[0..n)` with
 * the ids.  The output buffer is owned by the caller; the tokenizer never
 * allocates on the caller's behalf.
 *
 * When `cap` is smaller than the required count, returns the required count
 * and writes nothing to `out` (overflow probe).  Pass cap=0 with out=NULL to
 * query the length before allocating.
 *
 * Returns 0 for NULL input or empty text.
 *
 * Example:
 *     size_t n = olmoe_tokenize(s, NULL, 0);
 *     olmoe_token_id_t *ids = malloc(n * sizeof *ids);
 *     olmoe_tokenize(s, ids, n);
 */
size_t olmoe_tokenize(const char *text, olmoe_token_id_t *out, size_t cap);

#endif /* OLMOE_TOKENIZER_H */
