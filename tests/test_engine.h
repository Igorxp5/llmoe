#ifndef OLMOE_TEST_ENGINE_H
#define OLMOE_TEST_ENGINE_H

/* Runs every engine-module check. Stubs only at this stage; the cases here
 * assert the NULL-safety / capacity contract of the public API, not any
 * numerical correctness. Reports the combined failure count (per
 * tests/test_main.c). */
int test_engine_stubs_pass(void);

#endif /* OLMOE_TEST_ENGINE_H */
