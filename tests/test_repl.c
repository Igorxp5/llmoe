#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "repl.h"

#include "test_repl.h"

int test_repl_install_sigint_sets_handler(void)
{
    olmoe_repl_install_sigint();

    /* Read back the disposition without installing a new one. A custom
     * handler (neither SIG_DFL nor SIG_IGN) proves olmoe_repl_install_sigint
     * actually wired SIGINT. */
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    if (sigaction(SIGINT, NULL, &sa) != 0) {
        printf("FAIL: sigaction read failed\n");
        return 1;
    }
    if (sa.sa_handler == SIG_DFL || sa.sa_handler == SIG_IGN) {
        printf("FAIL: SIGINT handler not installed\n");
        return 1;
    }
    printf("PASS: install_sigint installs a custom SIGINT handler\n");
    return 0;
}

int test_repl_run_returns_on_eof(void)
{
    /* Point stdin at a closed pipe so fgets hits EOF immediately. No line is
     * ever read, so olmoe_repl_run must return without touching its (NULL)
     * model / scratch / token args. */
    int p[2];
    if (pipe(p) != 0) {
        printf("FAIL: pipe creation\n");
        return 1;
    }
    close(p[1]);
    int saved = dup(STDIN_FILENO);
    if (saved < 0 || dup2(p[0], STDIN_FILENO) < 0) {
        printf("FAIL: dup2 stdin redirect\n");
        close(saved); close(p[0]);
        return 1;
    }
    close(p[0]);

    olmoe_repl_run(NULL, NULL, NULL);

    /* Restore the inherited stdin. */
    dup2(saved, STDIN_FILENO);
    close(saved);
    printf("PASS: repl_run returns on stdin EOF\n");
    return 0;
}

int test_repl_pass(void)
{
    int failed = 0;
    failed += test_repl_install_sigint_sets_handler();
    failed += test_repl_run_returns_on_eof();
    return failed;
}
