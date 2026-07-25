#ifndef OLMOE_LAYER_H
#define OLMOE_LAYER_H

#include <stddef.h>
#include <stdint.h>

/* BF16 weight storage. We keep the on-disk BF16 bit pattern verbatim in a
 * uint16_t, deferring any numeric conversion to the consumer. This matches
 * the load contract described in docs/layer_module.md: the loaded model is
 * a faithful in-RAM mirror of the safetensors data. */
typedef uint16_t olmoe_bf16_t;

#include "olmoe/layers/model_layout.inc"

typedef struct {
    olmoe_bf16_t *gate_proj;     /* [inter, hidden]   */
    olmoe_bf16_t *up_proj;      /* [inter, hidden]   */
    olmoe_bf16_t *down_proj;    /* [hidden, inter]   */
} olmoe_expert_t;

typedef struct {
    olmoe_bf16_t *q_proj;       /* [hidden, hidden]  */
    olmoe_bf16_t *k_proj;       /* [hidden, hidden]  */
    olmoe_bf16_t *v_proj;       /* [hidden, hidden]  */
    olmoe_bf16_t *o_proj;       /* [hidden, hidden]  */
    olmoe_bf16_t *q_norm;       /* [hidden]          */
    olmoe_bf16_t *k_norm;       /* [hidden]          */
} olmoe_self_attn_t;

typedef struct {
    olmoe_self_attn_t self_attn;
    olmoe_bf16_t *input_layernorm;        /* [hidden]          */
    olmoe_bf16_t *post_attention_layernorm; /* [hidden]          */
    olmoe_bf16_t *mlp_gate;               /* [n_experts,hidden] (router) */
    olmoe_expert_t experts[OLMOE_N_EXPERTS];
} olmoe_layer_t;

/* Loaded OLMoE model. All weight buffers are owned by the model and freed
 * by olmoe_model_free. `bufs` is the flat pointer table (one entry per
 * tensor read, in shard/read order) that owns the actual allocations; the
 * struct-field pointers above are aliases into the same memory. Lifetime:
 *
 *     olmoe_model_t *m = olmoe_model_load("/path/to/model_dir");
 *     ... read m->layers[i].self_attn.q_proj ...
 *     olmoe_model_free(m);
 */
typedef struct {
    olmoe_bf16_t *embed_tokens;     /* [vocab, hidden]   */
    olmoe_bf16_t *lm_head;           /* [vocab, hidden]   */
    olmoe_bf16_t *norm;              /* [hidden]          */
    size_t n_layers;
    olmoe_layer_t *layers;

    olmoe_bf16_t **bufs;     /* flat ownership table */
    size_t n_bufs;
} olmoe_model_t;

/* Load the OLMoE model from `dir`. `dir` must contain `model-*.safetensors`
 * shards (the index and config are NOT read at runtime; their layout is
 * baked into the binary by scripts/generate_model_layout.py at build time).
 *
 * Returns NULL on any I/O, allocation, or layout mismatch (the loader
 * validates each shard's tensor count, byte sizes, and EOF after the
 * last tensor against the baked layout).
 *
 * Example:
 *     olmoe_model_t *m = olmoe_model_load("models/OLMoE-1B-7B-0924-Instruct");
 *     if (!m) { perror("load"); return 1; }
 *     ... use m->layers[0].self_attn.q_proj ...
 *     olmoe_model_free(m);
 */
olmoe_model_t *olmoe_model_load(const char *dir);

/* Free a model returned by olmoe_model_load. NULL-safe. */
void olmoe_model_free(olmoe_model_t *model);

#endif /* OLMOE_LAYER_H */