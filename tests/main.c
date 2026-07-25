#include <stdio.h>

#include "test_olmoe_tokenizer.h"

#include "assertion.h"

int main(void) {
  ASSERT(test_olmoe_tokenize());
  printf("PASS: all tests (IREE tokenizer integrated)\n");
  return 0;
}
