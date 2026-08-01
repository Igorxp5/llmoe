#include <stdio.h>
#include <stdlib.h>

#include "kernels/cpu_argmax.h"

#include "test_engine_helpers.h"

/* Scalar argmax reference: lowest index of the maximum (first-wins
 * tie-break), mirroring the semantics cpu_argmax must preserve. */
static size_t scalar_argmax(const float *a, size_t n)
{
    size_t best = 0;
    for (size_t r = 1; r < n; ++r)
        if (a[r] > a[best]) best = r;
    return best;
}

/* Run one deterministic pseudo-random array of size `n` and compare
 * cpu_argmax against the scalar reference. */
static int check_argmax_one_size(size_t n)
{
    float *a = malloc((n ? n : 1) * sizeof *a);
    if (!a) {
        printf("FAIL: argmax malloc OOM\n");
        return 1;
    }
    for (size_t r = 0; r < n; ++r)
        a[r] = (float)((int)((r * 131 + 17) % 251)) / 10.0f - 12.5f;

    int failed = 0;
    size_t got = cpu_argmax(a, n);
    size_t want = scalar_argmax(a, n);
    if (got != want) {
        printf("FAIL: argmax n=%zu got=%zu want=%zu\n", n, got, want);
        ++failed;
    }
    free(a);
    return failed;
}

/* Duplicate maximum values must resolve to the lowest index (the scalar
 * loop used strict `>`, so first occurrence wins). */
static int test_argmax_tie_break_first_wins(void)
{
    float a[40] = { 0 };
    a[3] = 5.0f;
    a[10] = 5.0f;
    a[11] = 3.0f;
    a[5] = -2.0f;
    size_t got = cpu_argmax(a, 40);
    if (got != 3) {
        printf("FAIL: argmax tie-break got=%zu want=3\n", got);
        return 1;
    }
    printf("PASS: argmax tie-break first-wins\n");
    return 0;
}

/* Single-element and empty inputs must stay in bounds and report index 0. */
static int test_argmax_trivial_inputs(void)
{
    int failed = 0;
    float one = 7.25f;
    if (cpu_argmax(&one, 1) != 0) {
        printf("FAIL: argmax n=1\n");
        ++failed;
    }
    if (cpu_argmax(&one, 0) != 0) {
        printf("FAIL: argmax n=0\n");
        ++failed;
    }
    if (!failed) printf("PASS: argmax trivial inputs\n");
    return failed;
}

int test_engine_argmax_pass(void)
{
    int failed = 0;
    failed += check_argmax_one_size(1);
    failed += check_argmax_one_size(15);
    failed += check_argmax_one_size(31);
    failed += check_argmax_one_size(48);
    failed += check_argmax_one_size(97);
    failed += test_argmax_tie_break_first_wins();
    failed += test_argmax_trivial_inputs();
    if (!failed) printf("PASS: argmax matches scalar\n");
    return failed;
}
