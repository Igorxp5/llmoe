#include <stdio.h>

#include "test_engine.h"
#include "test_layer.h"
#include "test_repl.h"
#include "test_tokenizer.h"

int main(void)
{
    int failed = 0;
    failed += test_oracle_cases_match_expected_ids();
    failed += test_null_input_returns_zero();
    failed += test_empty_input_preserves_out_buffer();
    failed += test_overflow_probe_reports_full_count();
    failed += test_oversized_cap_writes_within_cap();
    failed += test_decode_null_ids_returns_zero();
    failed += test_decode_empty_list_returns_zero();
    failed += test_decode_probe_length_and_fill();
    failed += test_decode_roundtrip_plaintext();
    failed += test_decode_exact_cap_fills_buffer();
    failed += test_decode_truncated_cap_reports_full();
    failed += test_layer_loads_and_validates();
    failed += test_engine_stubs_pass();
    failed += test_repl_pass();

    if (failed == 0) {
        printf("\nAll tests passed.\n");
        return 0;
    }
    printf("\n%d test(s) failed.\n", failed);
    return 1;
}
