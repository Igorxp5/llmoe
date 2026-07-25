#ifndef OLMOE_TEST_TOKENIZER_H
#define OLMOE_TEST_TOKENIZER_H

/* Each test function returns the number of failures it observed (0 == pass).
 * test_main.c sums these to decide the process exit code. */
int test_oracle_cases_match_expected_ids(void);
int test_null_input_returns_zero(void);
int test_empty_input_preserves_out_buffer(void);
int test_overflow_probe_reports_full_count(void);

#endif /* OLMOE_TEST_TOKENIZER_H */
