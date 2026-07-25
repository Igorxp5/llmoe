#include <stdio.h>

#include "test_engine.h"
#include "test_layer.h"
#include "test_tokenizer.h"

int main(void)
{
    int failed = 0;
    failed += test_oracle_cases_match_expected_ids();
    failed += test_null_input_returns_zero();
    failed += test_empty_input_preserves_out_buffer();
    failed += test_overflow_probe_reports_full_count();
    failed += test_layer_loads_and_validates();
    failed += test_engine_stubs_pass();

    if (failed == 0) {
        printf("\nAll tests passed.\n");
        return 0;
    }
    printf("\n%d test(s) failed.\n", failed);
    return 1;
}
