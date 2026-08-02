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

int test_oversized_cap_writes_within_cap(void)
{
    /* cap > needed: tokens fill the first `full` slots and the trailing
     * slack must remain untouched. */
    const char *sample = "Hi";
    size_t full = olmoe_tokenize(sample, NULL, 0);
    if (full == 0) {
        printf("FAIL: oversized-cap probe reported 0 for non-empty input\n");
        return 1;
    }
    olmoe_token_id_t buf[8];
    for (size_t i = 0; i < 8; ++i) buf[i] = 0xAAu;
    size_t got = olmoe_tokenize(sample, buf, 8);
    if (got != full) {
        printf("FAIL: oversized cap returned %zu, expected %zu\n",
               got, full);
        return 1;
    }
    for (size_t i = full; i < 8; ++i) {
        if (buf[i] != 0xAAu) {
            printf("FAIL: oversized cap wrote past used bytes at %zu\n", i);
            return 1;
        }
    }
    printf("PASS: oversized cap (%zu tokens, cap=8 slack untouched)\n", full);
    return 0;
}

/* olmoe_decode: ids for "Hello, world!" from the oracle corpus. */
static const olmoe_token_id_t HELLO_IDS[] = {12092, 13, 1533, 2};

int test_decode_null_ids_returns_zero(void)
{
    if (olmoe_decode(NULL, 4, NULL, 0) != 0) {
        printf("FAIL: decode(NULL ids) returned non-zero\n");
        return 1;
    }
    printf("PASS: decode NULL ids / zero-count\n");
    return 0;
}

int test_decode_empty_list_returns_zero(void)
{
    olmoe_token_id_t ids[] = { 7 };
    if (olmoe_decode(ids, 0, NULL, 0) != 0) {
        printf("FAIL: decode(n_ids=0) returned non-zero\n");
        return 1;
    }
    if (olmoe_decode(NULL, 0, NULL, 0) != 0) {
        printf("FAIL: decode(NULL n_ids=0) returned non-zero\n");
        return 1;
    }
    printf("PASS: decode empty list returns zero\n");
    return 0;
}

int test_decode_probe_length_and_fill(void)
{
    const size_t n = sizeof(HELLO_IDS) / sizeof(HELLO_IDS[0]);
    /* Probe with a null buffer: returns the exact byte count. */
    size_t need = olmoe_decode(HELLO_IDS, n, NULL, 0);
    if (need == 0) {
        printf("FAIL: decode probe returned 0 for known tokens\n");
        return 1;
    }
    /* Fill a buffer sized full+1 (room for the NUL). */
    char *buf = (char *)calloc(need + 2, 1);
    if (!buf) {
        printf("FAIL: decode oom allocating %zu bytes\n", need + 1);
        return 1;
    }
    buf[need + 1] = 0x7F;
    size_t got = olmoe_decode(HELLO_IDS, n, buf, need + 1);
    if (got != need) {
        printf("FAIL: decode fill returned %zu, expected %zu\n", got, need);
        free(buf);
        return 1;
    }
    if (buf[need] != '\0') {
        printf("FAIL: decode did not NUL-terminate the buffer\n");
        free(buf);
        return 1;
    }
    if (buf[0] == '\0') {
        printf("FAIL: decode wrote nothing to the buffer\n");
        free(buf);
        return 1;
    }
    /* The guarantee does not extend past `need`, so the slot after the NUL
     * is free-by-design; we only enforce termination at [need]. */
    free(buf);
    printf("PASS: decode probe length and fill (need=%zu)\n", need);
    return 0;
}

int test_decode_roundtrip_plaintext(void)
{
    const size_t n = sizeof(HELLO_IDS) / sizeof(HELLO_IDS[0]);
    size_t need = olmoe_decode(HELLO_IDS, n, NULL, 0);
    char *buf = (char *)calloc(need + 1, 1);
    if (!buf) {
        printf("FAIL: roundtrip oom\n");
        return 1;
    }
    olmoe_decode(HELLO_IDS, n, buf, need + 1);
    /* Decoding "Hello, world!" ids must re-tokenize to the same ids. */
    olmoe_token_id_t back[8];
    size_t bn = olmoe_tokenize(buf, back, 8);
    int failed = 0;
    if (bn != n || !ids_equal(back, HELLO_IDS, n)) {
        printf("FAIL: decode/encode roundtrip mismatch (bn=%zu n=%zu)\n",
               bn, n);
        failed = 1;
    }
    free(buf);
    if (!failed)
        printf("PASS: decode/encode roundtrip plaintext\n");
    return failed;
}

int test_decode_exact_cap_fills_buffer(void)
{
    const size_t n = sizeof(HELLO_IDS) / sizeof(HELLO_IDS[0]);
    size_t full = olmoe_decode(HELLO_IDS, n, NULL, 0);
    if (full == 0) {
        printf("FAIL: exact-cap decode probe returned 0\n");
        return 1;
    }
    /* cap == full+1 fits everything plus the NUL. */
    char buf[64];
    memset(buf, 0xAAu, sizeof buf);
    size_t got = olmoe_decode(HELLO_IDS, n, buf, full + 1);
    if (got != full) {
        printf("FAIL: exact-cap decode returned %zu, expected %zu\n",
               got, full);
        return 1;
    }
    if (buf[full] != '\0') {
        printf("FAIL: exact-cap decode missing NUL at %zu\n", full);
        return 1;
    }
    printf("PASS: decode exact cap (need+1) fills and terminates\n");
    return 0;
}

int test_decode_truncated_cap_reports_full(void)
{
    const size_t n = sizeof(HELLO_IDS) / sizeof(HELLO_IDS[0]);
    size_t full = olmoe_decode(HELLO_IDS, n, NULL, 0);
    if (full == 0) return 1;
    /* cap smaller than needed still reports the full length (a truncated
     * copy is written, terminated at cap-1). */
    size_t tiny_cap = 5;
    if (full <= tiny_cap) {
        printf("FAIL: truncated-cap fixture too small for full=%zu\n", full);
        return 1;
    }
    char tiny[8];
    memset(tiny, 0xABu, sizeof tiny);
    size_t got = olmoe_decode(HELLO_IDS, n, tiny, tiny_cap);
    if (got != full) {
        printf("FAIL: truncated decode returned %zu, expected full %zu\n",
               got, full);
        return 1;
    }
    if (tiny[tiny_cap - 1] != '\0') {
        printf("FAIL: truncated decode missing NUL terminator\n");
        return 1;
    }
    printf("PASS: decode truncated cap reports full count\n");
    return 0;
}
