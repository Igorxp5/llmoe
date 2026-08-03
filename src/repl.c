#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "olmoe/engine/engine.h"
#include "olmoe/tokenizer/tokenizer.h"

#include "kernels/cpu_argmax.h"
#include "repl.h"

#define MAX_LINE     8192
#define EOS_TOKEN_ID 50279

static volatile sig_atomic_t stop_flag = 0;

static void handle_sigint(int sig)
{
    (void)sig;
    stop_flag = 1;
}

void olmoe_repl_install_sigint(void)
{
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handle_sigint;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    if (sigaction(SIGINT, &sa, NULL) != 0) {
        perror("sigaction");
    }
}

static int read_user_line(char *line, size_t cap)
{
    if (stop_flag) {
        fprintf(stderr, "\n");
        return 0;
    }
    fprintf(stderr, "> ");
    if (!fgets(line, cap, stdin)) {
        fprintf(stderr, "\n");
        return 0;
    }
    size_t linelen = strlen(line);
    if (linelen > 0 && line[linelen - 1] == '\n')
        line[linelen - 1] = '\0';
    if (line[0] == '\0') return 1;
    return 2;
}

static int wrap_in_chat_template(const char *line, char *out, size_t cap)
{
    int n = snprintf(out, cap,
                     "<|endoftext|>\n<|user|>\n%s\n<|assistant|>",
                     line);
    if (n < 0 || (size_t)n >= cap) return -1;
    return 0;
}

static int prepare_prompt(const char *line, olmoe_scratch_t *s,
                          olmoe_token_id_t *tokens, size_t *n_tok_out)
{
    char prompt[MAX_LINE + 128];
    if (wrap_in_chat_template(line, prompt, sizeof(prompt)) != 0) {
        fprintf(stderr, "[debug] prompt too long\n");
        return -1;
    }
    *n_tok_out = olmoe_tokenize(prompt, tokens, MAX_SEQ_LEN);
    fprintf(stderr, "[debug] input tokens: %zu\n", *n_tok_out);
    s->cache_len = 0;
    return 0;
}

static int prefill_and_first_token(const olmoe_model_t *m, olmoe_scratch_t *s,
                                   olmoe_token_id_t *tokens, size_t n_tok)
{
    if (n_tok == 0) return EOS_TOKEN_ID;
    if (olmoe_forward(m, (int *)tokens, n_tok, 0, s, s->logits) != OLMOE_OK) {
        fprintf(stderr, "prefill failed\n");
        return EOS_TOKEN_ID;
    }
    size_t last = n_tok - 1;
    int next = (int)cpu_argmax(s->logits + last * OLMOE_VOCAB, OLMOE_VOCAB);
    if (next == EOS_TOKEN_ID) return EOS_TOKEN_ID;
    tokens[n_tok] = (olmoe_token_id_t)next;
    return next;
}

static void stream_decoded_tokens(const olmoe_token_id_t *tokens,
                                  size_t n_tok, size_t n_output,
                                  size_t *dec_len)
{
    size_t new_len = olmoe_decode(tokens + n_tok, n_output, NULL, 0);
    if (__builtin_expect(new_len <= *dec_len, 0)) return;
    char *dec = malloc(new_len + 1);
    if (__builtin_expect(!dec, 0)) return;
    olmoe_decode(tokens + n_tok, n_output, dec, new_len + 1);
    fwrite(dec + *dec_len, 1, new_len - *dec_len, stdout);
    fflush(stdout);
    free(dec);
    *dec_len = new_len;
}

static void decode_until_eos(const olmoe_model_t *m, olmoe_scratch_t *s,
                             olmoe_token_id_t *tokens, size_t n_tok,
                             size_t *output_tokens, size_t *dec_len)
{
    for (size_t pos = n_tok; pos + 1 < MAX_SEQ_LEN; ++pos) {
        if (stop_flag || *output_tokens == 0) return;

        if (__builtin_expect(olmoe_forward(m, (int *)&tokens[pos], 1, pos, s,
                                           s->logits) != OLMOE_OK, 0)) {
            fprintf(stderr, "forward failed at pos %zu\n", pos);
            return;
        }

        int next = (int)cpu_argmax(s->logits, OLMOE_VOCAB);
        if (__builtin_expect(next == EOS_TOKEN_ID, 0)) return;
        tokens[pos + 1] = (olmoe_token_id_t)next;
        (*output_tokens)++;
        stream_decoded_tokens(tokens, n_tok, *output_tokens, dec_len);
    }
}

static void generate_response(const olmoe_model_t *m, olmoe_scratch_t *s,
                              olmoe_token_id_t *tokens, size_t n_tok,
                              size_t *output_tokens, size_t *dec_len)
{
    *output_tokens = 0;
    *dec_len = 0;
    int next = prefill_and_first_token(m, s, tokens, n_tok);
    if (next != EOS_TOKEN_ID) *output_tokens = 1;
    decode_until_eos(m, s, tokens, n_tok, output_tokens, dec_len);
}

static void finish_turn_response(size_t output_tokens, struct timespec gen_t0)
{
    struct timespec gen_t1;
    clock_gettime(CLOCK_MONOTONIC, &gen_t1);
    fflush(stdout);
    if (output_tokens > 0) printf("\n");
    double secs = (gen_t1.tv_sec - gen_t0.tv_sec)
                + (gen_t1.tv_nsec - gen_t0.tv_nsec) / 1e9;
    fprintf(stderr, "[debug] output tokens: %zu, speed: %.2f tok/s\n",
            output_tokens, output_tokens / secs);
    stop_flag = 0;
}

static void run_turn(const olmoe_model_t *m, olmoe_scratch_t *s,
                     olmoe_token_id_t *tokens, const char *line)
{
    size_t n_tok;
    if (prepare_prompt(line, s, tokens, &n_tok) != 0) return;

    struct timespec gen_t0;
    clock_gettime(CLOCK_MONOTONIC, &gen_t0);

    size_t output_tokens = 0;
    size_t dec_len = 0;
    generate_response(m, s, tokens, n_tok, &output_tokens, &dec_len);

    finish_turn_response(output_tokens, gen_t0);
}

void olmoe_repl_run(const olmoe_model_t *m, olmoe_scratch_t *s,
                    olmoe_token_id_t *tokens)
{
    char line[MAX_LINE];
    for (;;) {
        int r = read_user_line(line, sizeof(line));
        if (r == 0) break;
        if (r == 1) continue;
        run_turn(m, s, tokens, line);
    }
}
