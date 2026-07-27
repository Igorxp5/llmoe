#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "olmoe/engine/engine.h"
#include "olmoe/tokenizer/tokenizer.h"

#define MAX_SEQ_LEN 2048
#define MAX_LINE 8192
#define EOS_TOKEN_ID 50279

static int usage(const char *prog)
{
    fprintf(stderr, "Usage: %s <model_dir>\n", prog);
    return 1;
}

static double elapsed(struct timespec a, struct timespec b)
{
    return (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9;
}

int main(int argc, char **argv)
{
    if (argc != 2) return usage(argv[0]);
    const char *model_dir = argv[1];

    struct timespec load_t0, load_t1, gen_t0, gen_t1;
    olmoe_model_t *m = NULL;
    olmoe_scratch_t s;
    olmoe_token_id_t *tokens = NULL;
    int rc = 0;

    memset(&s, 0, sizeof(s));

    clock_gettime(CLOCK_MONOTONIC, &load_t0);
    m = olmoe_model_load(model_dir);
    clock_gettime(CLOCK_MONOTONIC, &load_t1);
    if (!m) {
        fprintf(stderr, "Failed to load model from %s\n", model_dir);
        rc = 1;
        goto out;
    }
    fprintf(stderr, "[debug] Loaded model: %zu layers (%.3f s)\n",
            m->n_layers, elapsed(load_t0, load_t1));

    if (olmoe_scratch_init(&s, MAX_SEQ_LEN) != OLMOE_OK) {
        fprintf(stderr, "scratch init failed\n");
        rc = 2;
        goto out;
    }

    tokens = malloc(MAX_SEQ_LEN * sizeof(*tokens));
    if (!tokens) {
        fprintf(stderr, "malloc failed\n");
        rc = 3;
        goto out;
    }

    size_t seq_len = 0;
    char line[MAX_LINE];

    for (;;) {
        fprintf(stderr, "> ");
        if (!fgets(line, sizeof(line), stdin)) {
            fprintf(stderr, "\n");
            break;
        }

        size_t linelen = strlen(line);
        if (linelen > 0 && line[linelen - 1] == '\n')
            line[linelen - 1] = '\0';
        if (line[0] == '\0') continue;

        /* Wrap user input in the OLMoE instruct chat template so the
         * model receives the instruction format it was trained on:
         *
         *     <|endoftext|>
         *     <|user|>
         *     {user_input}
         *     <|assistant|>
         *
         * Each prompt is self-contained: seq_len resets per turn rather
         * than accumulating (multi-turn accumulation requires sparse-
         * attention / KV-cache engineering out of scope). */
        char prompt[MAX_LINE + 128];
        int p_len = snprintf(prompt, sizeof(prompt),
                             "<|endoftext|>\n<|user|>\n%s\n<|assistant|>\n",
                             line);
        if (p_len < 0 || (size_t)p_len >= sizeof(prompt)) {
            fprintf(stderr, "[debug] prompt too long\n");
            continue;
        }

        size_t n_tok = olmoe_tokenize(prompt, NULL, 0);
        if (n_tok == 0) {
            fprintf(stderr, "[debug] empty tokenization\n");
            continue;
        }
        if (n_tok > MAX_SEQ_LEN) {
            fprintf(stderr, "[debug] context full, ignoring input\n");
            continue;
        }
        olmoe_tokenize(prompt, tokens, (size_t)n_tok);
        seq_len = n_tok;
        fprintf(stderr, "[debug] input tokens: %zu\n", n_tok);

        size_t gen_start = seq_len;
        size_t output_tokens = 0;
        size_t dec_len = 0;

        clock_gettime(CLOCK_MONOTONIC, &gen_t0);
        for (; seq_len < MAX_SEQ_LEN; ++seq_len) {
            if (olmoe_forward(m, (int *)tokens, seq_len, &s, s.logits)
                != OLMOE_OK) {
                fprintf(stderr, "forward failed at step %zu\n", seq_len);
                break;
            }

            size_t last = seq_len - 1;
            int best_tok = 0;
            float best_val = s.logits[last * OLMOE_VOCAB];
            for (int v = 1; v < OLMOE_VOCAB; ++v) {
                float val = s.logits[last * OLMOE_VOCAB + v];
                if (val > best_val) {
                    best_val = val;
                    best_tok = v;
                }
            }

            if (best_tok == EOS_TOKEN_ID) break;
            tokens[seq_len] = (olmoe_token_id_t)best_tok;
            output_tokens++;

            size_t new_len = olmoe_decode(tokens + gen_start,
                                          output_tokens, NULL, 0);
            if (new_len > dec_len) {
                char *dec = malloc(new_len + 1);
                if (dec) {
                    olmoe_decode(tokens + gen_start, output_tokens,
                                 dec, new_len + 1);
                    fwrite(dec + dec_len, 1, new_len - dec_len, stdout);
                    fflush(stdout);
                    free(dec);
                }
                dec_len = new_len;
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &gen_t1);

        if (dec_len > 0) printf("\n");

        fprintf(stderr, "[debug] output tokens: %zu, speed: %.2f tok/s\n",
                output_tokens,
                output_tokens / elapsed(gen_t0, gen_t1));
    }

out:
    olmoe_scratch_free(&s);
    free(tokens);
    olmoe_model_free(m);
    return rc;
}
