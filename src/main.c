#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "olmoe/engine/engine.h"
#include "olmoe/tokenizer/tokenizer.h"

#define DEFAULT_MODEL_DIR "models/OLMoE-1B-7B-0924-Instruct"

static int usage(const char *prog)
{
    fprintf(stderr, "Usage: %s [--model <dir>] <prompt>\n", prog);
    return 1;
}

int main(int argc, char **argv)
{
    const char *model_dir = DEFAULT_MODEL_DIR;
    int i = 1;
    if (i + 1 < argc && strcmp(argv[i], "--model") == 0) {
        model_dir = argv[i + 1];
        i += 2;
    }
    if (argc - i != 1) return usage(argv[0]);
    const char *prompt = argv[i];

    olmoe_model_t *m = NULL;
    olmoe_token_id_t *ids = NULL;
    olmoe_scratch_t s;
    int rc = 0;

    memset(&s, 0, sizeof(s));

    m = olmoe_model_load(model_dir);
    if (!m) {
        fprintf(stderr, "Failed to load model from %s\n", model_dir);
        rc = 2;
        goto out;
    }
    fprintf(stderr, "Loaded model: %zu layers\n", m->n_layers);

    size_t n = olmoe_tokenize(prompt, NULL, 0);
    if (n == 0) {
        fprintf(stderr, "Empty prompt\n");
        rc = 3;
        goto out;
    }

    ids = malloc(n * sizeof(*ids));
    if (!ids) {
        fprintf(stderr, "malloc failed\n");
        rc = 4;
        goto out;
    }
    olmoe_tokenize(prompt, ids, n);

    if (olmoe_scratch_init(&s, n) != OLMOE_OK) {
        fprintf(stderr, "scratch init failed\n");
        rc = 5;
        goto out;
    }

    if (olmoe_forward(m, (int *)ids, n, &s, s.logits) != OLMOE_OK) {
        fprintf(stderr, "forward pass failed\n");
        rc = 6;
        goto out;
    }

    size_t last = n - 1;
    int best_tok = 0;
    float best_val = s.logits[last * OLMOE_VOCAB];
    for (int v = 1; v < OLMOE_VOCAB; ++v) {
        float val = s.logits[last * OLMOE_VOCAB + v];
        if (val > best_val) {
            best_val = val;
            best_tok = v;
        }
    }

    printf("Encoded %zu tokens:", n);
    for (size_t k = 0; k < n; ++k) printf(" %u", ids[k]);
    printf("\n");
    printf("Predicted next token: %d (logit: %.6f)\n", best_tok, best_val);

out:
    olmoe_scratch_free(&s);
    free(ids);
    olmoe_model_free(m);
    return rc;
}
