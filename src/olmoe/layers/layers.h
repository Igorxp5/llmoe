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
    olmoe_bf16_t gate_proj[OLMOE_INTER * OLMOE_HIDDEN]; /* [inter, hidden]   */
    olmoe_bf16_t up_proj[OLMOE_INTER * OLMOE_HIDDEN];   /* [inter, hidden]   */
    olmoe_bf16_t down_proj[OLMOE_HIDDEN * OLMOE_INTER]; /* [hidden, inter]   */
} olmoe_expert_t;

typedef struct {
    olmoe_bf16_t q_proj[OLMOE_HIDDEN * OLMOE_HIDDEN];   /* [hidden, hidden]  */
    olmoe_bf16_t k_proj[OLMOE_HIDDEN * OLMOE_HIDDEN];   /* [hidden, hidden]  */
    olmoe_bf16_t v_proj[OLMOE_HIDDEN * OLMOE_HIDDEN];   /* [hidden, hidden]  */
    olmoe_bf16_t o_proj[OLMOE_HIDDEN * OLMOE_HIDDEN];   /* [hidden, hidden]  */
    olmoe_bf16_t q_norm[OLMOE_HIDDEN];                  /* [hidden]          */
    olmoe_bf16_t k_norm[OLMOE_HIDDEN];                  /* [hidden]          */
} olmoe_self_attn_t;

typedef struct {
    olmoe_self_attn_t self_attn;
    olmoe_bf16_t input_layernorm[OLMOE_HIDDEN];          /* [hidden]          */
    olmoe_bf16_t post_attention_layernorm[OLMOE_HIDDEN]; /* [hidden]          */
    olmoe_bf16_t mlp_gate[OLMOE_N_EXPERTS * OLMOE_HIDDEN]; /* [n_experts,hidden] (router) */
    olmoe_expert_t experts[OLMOE_N_EXPERTS];
} olmoe_layer_t;

/* Loaded OLMoE model. One contiguous anonymous mmap'd block (page-aligned,
 * zero-initialized) backed by hugetlb 2 MiB pages when reserved, else plain
 * 4 KiB pages; every weight field is a zero-initialized array so there are
 * no individual mallocs or frees. The loader fread's shard bytes directly
 * into the right field offsets, then mlock()s the region and mprotect()s it
 * PROT_READ: the model is read-only and pinned, so the returned pointer is
 * const. Lifetime:
 *
 *     const olmoe_model_t *m = olmoe_model_load("/path/to/model_dir");
 *     ... read m->layers[i].self_attn.q_proj ...
 *     olmoe_model_free(m);
 */
typedef struct {
    olmoe_bf16_t embed_tokens[OLMOE_VOCAB * OLMOE_HIDDEN]; /* [vocab, hidden]   */
    olmoe_bf16_t lm_head[OLMOE_VOCAB * OLMOE_HIDDEN];      /* [vocab, hidden]   */
    olmoe_bf16_t norm[OLMOE_HIDDEN];                       /* [hidden]          */
    olmoe_layer_t layers[OLMOE_N_LAYERS];
} olmoe_model_t;

/* Load the OLMoE model from `dir`. `dir` must contain `model-*.safetensors`
 * shards (the index and config are NOT read at runtime; their layout is
 * baked into the binary by scripts/generate_model_layout.py at build time).
 *
 * The returned model is one anonymous mmap'd region (~12.9 GiB); weights are
 * written in place via fread, then the region is mlock()'d and made
 * read-only (mprotect PROT_READ), so the pointer is const and any write to
 * it faults. Prefers hugetlb 2 MiB pages (requires reserved vm.nr_hugepages)
 * and falls back to 4 KiB pages otherwise. Returns NULL on any I/O, layout
 * mismatch, or mmap failure.
 *
 * Example:
 *     const olmoe_model_t *m = olmoe_model_load("models/OLMoE-1B-7B-0924-Instruct");
 *     if (!m) { perror("load"); return 1; }
 *     ... use m->layers[0].self_attn.q_proj ...
 *     olmoe_model_free(m);
 */
const olmoe_model_t *olmoe_model_load(const char *dir);

/* Free a model returned by olmoe_model_load. NULL-safe. */
void olmoe_model_free(const olmoe_model_t *model);

#endif /* OLMOE_LAYER_H */
