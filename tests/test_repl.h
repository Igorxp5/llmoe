#ifndef OLMOE_TEST_REPL_H
#define OLMOE_TEST_REPL_H

/* Runs every REPL-module check. The interactive generation path cannot be
 * exercised without a loaded model, so this covers only what is testable
 * non-interactively: the SIGINT handler install and EOF-termination of the
 * read loop. Returns the combined failure count (per tests/test_main.c). */
int test_repl_pass(void);

#endif /* OLMOE_TEST_REPL_H */
