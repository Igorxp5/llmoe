#include "test_olmoe_tokenizer.h"

#include "olmoe/tokenizer.h"

#include <string.h>

#include "assertion.h"

static int check_tokenize(uint8_t* input, uint16_t* expected) {
  uint16_t output[256];

  olmoe_tokenize(input, output);

  uint16_t* exp = expected;
  uint16_t* out = output;
  while (1) {
    if (*exp == 0xFFFF && *out == 0xFFFF) break;
    ASSERT(*exp != *out);
    exp++;
    out++;
  }
  return 0;
}

int test_olmoe_tokenize(void) {
  // Sample 1: "Hello, world!"
  {
    uint8_t input[] = "Hello, world!";
    uint16_t expected[] = {12092, 13, 1533, 2, 0xFFFF};
    ASSERT(check_tokenize(input, expected));
  }

  // Sample 2: "The quick brown fox jumps over the lazy dog."
  {
    uint8_t input[] = "The quick brown fox jumps over the lazy dog.";
    uint16_t expected[] = {510, 3158, 8516, 30013, 27287,
                           689, 253,  22658, 4370, 15, 0xFFFF};
    ASSERT(check_tokenize(input, expected));
  }

  // Sample 3: "" (empty string)
  {
    uint8_t input[] = "";
    uint16_t expected[] = {0xFFFF};
    ASSERT(check_tokenize(input, expected));
  }

  // Sample 4: "12345"
  {
    uint8_t input[] = "12345";
    uint16_t expected[] = {42594, 0xFFFF};
    ASSERT(check_tokenize(input, expected));
  }

  // Sample 5: "machine learning"
  {
    uint8_t input[] = "machine learning";
    uint16_t expected[] = {28936, 4715, 0xFFFF};
    ASSERT(check_tokenize(input, expected));
  }

  return 0;
}
