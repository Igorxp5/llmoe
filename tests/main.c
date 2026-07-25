#include <stdio.h>
#include "tests/test_normalizer.h"

int main() {
    if (test_normalizer_nfc()) {
        printf("FAIL: test_normalizer_nfc\n");
        return 1;
    }
    printf("PASS: test_normalizer_nfc\n");
    return 0;
}
