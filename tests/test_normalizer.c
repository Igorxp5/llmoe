#include "tests/test_normalizer.h"
#include "tests/assertion.h"
#include <stdint.h>
#include "tokenizer/normalizer.h"
#include <string.h>

bool test_normalizer_nfc() {
    #include "tests/generated/nfc_tests.inc"
    return 0;
}
