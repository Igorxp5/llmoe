#include <stdio.h>

#include "test_tokenizer.h"

int main(void)
{
    int failed = 0;
    failed += test_oracle_cases_match_expected_ids();
    failed += test_null_input_returns_zero();
    failed += test_empty_input_preserves_out_buffer();
    failed += test_overflow_probe_reports_full_count();

    if (failed == 0) {
        printf("\nAll tokenizer tests passed.\n");
        return 0;
    }
    printf("\n%d test(s) failed.\n", failed);
    return 1;
}