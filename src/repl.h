#ifndef OLMOE_REPL_H
#define OLMOE_REPL_H

#include "olmoe/engine/engine.h"
#include "olmoe/tokenizer/tokenizer.h"

#define MAX_SEQ_LEN 2048

void olmoe_repl_install_sigint(void);
void olmoe_repl_run(const olmoe_model_t *m, olmoe_scratch_t *s,
                    olmoe_token_id_t *tokens);

#endif
