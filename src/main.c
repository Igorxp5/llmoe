#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "olmoe/engine/engine.h"
#include "olmoe/tokenizer/tokenizer.h"

#include "repl.h"

static int usage(const char *prog)
{
    fprintf(stderr, "Usage: %s <model_dir>\n", prog);
    return 1;
}

static const olmoe_model_t *load_model(const char *dir)
{
    fprintf(stderr, "[debug] Loading model from %s\n", dir);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    const olmoe_model_t *m = olmoe_model_load(dir);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (!m) {
        fprintf(stderr, "Failed to load model from %s\n", dir);
        return NULL;
    }
    double secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    fprintf(stderr, "[debug] Loaded model: %d layers (%.3f s)\n",
            OLMOE_N_LAYERS, secs);
    return m;
}

static int init_session(const olmoe_model_t **m_out, olmoe_scratch_t *s,
                        olmoe_token_id_t **tokens_out, const char *model_dir)
{
    *m_out = load_model(model_dir);
    if (!*m_out) return 1;

    if (olmoe_scratch_init(s, MAX_SEQ_LEN, MAX_SEQ_LEN) != OLMOE_OK) {
        fprintf(stderr, "scratch init failed\n");
        return 2;
    }

    *tokens_out = malloc(MAX_SEQ_LEN * sizeof(**tokens_out));
    if (!*tokens_out) {
        fprintf(stderr, "malloc failed\n");
        return 3;
    }
    return 0;
}

static void teardown_session(olmoe_scratch_t *s, olmoe_token_id_t *tokens,
                             const olmoe_model_t *m)
{
    olmoe_scratch_free(s);
    free(tokens);
    olmoe_model_free(m);
}

int main(int argc, char **argv)
{
    /* Pin OMP threads to physical cores (no SMT) and keep them there across
     * parallel regions. Thread count is left to OMP_NUM_THREADS/env defaults.
     * OMP_PLACES has no runtime API, so it must go through the environment;
     * libgomp reads it at the first parallel region, after main() starts. */
    setenv("OMP_PLACES", "cores", 1);
    setenv("OMP_PROC_BIND", "close", 1);

    if (argc != 2) return usage(argv[0]);

    int rc;
    const olmoe_model_t *m = NULL;
    olmoe_scratch_t s = {0};
    olmoe_token_id_t *tokens = NULL;

    olmoe_repl_install_sigint();

    rc = init_session(&m, &s, &tokens, argv[1]);
    if (rc == 0) olmoe_repl_run(m, &s, tokens);
    teardown_session(&s, tokens, m);
    return rc;
}
