#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <olmoe/tokenizer/tokenizer.h>

#include "test_tokenizer.h"
#include "tokenizer_expected.inc"

static bool ids_equal(const olmoe_token_id_t *a, const olmoe_token_id_t *b,
                      size_t n)
{
    return memcmp(a, b, n * sizeof(*a)) == 0;
}

static int check_one_oracle_case(const tokenizer_test_case_t *c)
{
    /* Probe: ask for size with cap=0/null so we never write past the end. */
    size_t needed = olmoe_tokenize(c->text, NULL, 0);
    if (needed != c->ids_len) {
        printf("FAIL: %-32s expected %zu ids, got %zu\n",
               c->text, c->ids_len, needed);
        return 1;
    }
    /* Allocate the exact buffer and tokenize for real. */
    olmoe_token_id_t *buf =
        (olmoe_token_id_t *)calloc(needed + 1, sizeof(*buf));
    if (!buf) {
        printf("FAIL: %-32s oom allocating %zu ids\n", c->text, needed);
        return 1;
    }
    /* Sentinel so a short write still differs from the expected ids. */
    buf[needed] = (olmoe_token_id_t)0u - 1;
    size_t got = olmoe_tokenize(c->text, buf, needed);
    bool ok = got == c->ids_len && ids_equal(buf, c->ids, c->ids_len) &&
              buf[needed] == (olmoe_token_id_t)0u - 1;
    int failed = 0;
    if (!ok) {
        printf("FAIL: %-32s id mismatch (got len=%zu expected=%zu)\n",
               c->text, got, c->ids_len);
        printf("        expected: ");
        for (size_t i = 0; i < c->ids_len; ++i) printf("%u ", c->ids[i]);
        printf("\n        got:      ");
        for (size_t i = 0; i < got; ++i) printf("%u ", buf[i]);
        printf("\n");
        failed = 1;
    } else {
        printf("PASS: %-32s -> %zu ids\n", c->text, needed);
    }
    free(buf);
    return failed;
}

int test_oracle_cases_match_expected_ids(void)
{
    int failed = 0;
    for (size_t i = 0; i < TEST_CASES_LEN; ++i) {
        failed += check_one_oracle_case(&TEST_CASES[i]);
    }
    return failed;
}

int test_null_input_returns_zero(void)
{
    /* NULL input must yield 0 and never dereference the buffer. */
    if (olmoe_tokenize(NULL, NULL, 0) != 0) {
        printf("FAIL: NULL input returned non-zero\n");
        return 1;
    }
    printf("PASS: NULL input\n");
    return 0;
}

int test_empty_input_preserves_out_buffer(void)
{
    /* Empty string: 0 ids and the caller's buffer must be untouched. */
    olmoe_token_id_t sentinel = 0xABABABABu;
    size_t n = olmoe_tokenize("", &sentinel, 1);
    if (n != 0 || sentinel != 0xABABABABu) {
        printf("FAIL: empty input -> %zu (sentinel tampered)\n", n);
        return 1;
    }
    printf("PASS: empty input\n");
    return 0;
}

int test_overflow_probe_reports_full_count(void)
{
    /* Capacity probe: cap < needed. Verify the reported count equals the
     * full count and that bytes past the small cap are untouched. */
    const char *sample = "Hello, world!";
    size_t full = olmoe_tokenize(sample, NULL, 0);
    olmoe_token_id_t tiny[2] = { 0xDE, 0xAD };
    size_t probe = olmoe_tokenize(sample, tiny,
                                  sizeof(tiny) / sizeof(tiny[0]));
    if (probe != full) {
        printf("FAIL: overflow probe returned %zu, expected %zu\n",
               probe, full);
        return 1;
    }
    if (tiny[1] != 0xAD) {
        printf("FAIL: overflow probe wrote past cap\n");
        return 1;
    }
    printf("PASS: overflow probe (reported %zu, cap=2 untouched)\n", probe);
    return 0;
}
