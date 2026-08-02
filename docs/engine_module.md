# Engine module (`src/olmoe/engine/`)

Forward-pass engine for OLMoE. Reads BF16 weights from an
`olmoe_model_t` produced by the layer loader (`src/olmoe/layers/`) and
produces activations the caller owns.

## Status

**Fully implemented.** The top-level `olmoe_forward` is the sole validation
point (NULL + `scratch->seq_len >= seq_len` + `seq_len == 0` + KV-cache
capacity guards); the 15 per-kind ops are pure compute returning `void`,
trusting the integrator to have validated args. Computation lives in
header-only kernels under `src/kernels/`:

  - `cpu_matmul.h`    — BF16-weight x FP32-act matmul (AVX512-BF16 + FMA)
  - `cpu_rmsnorm.h`   — RMSNorm with BF16 weight promotion (AVX512)
  - `cpu_softmax.h`   — numerically stable row softmax (deterministic sum)
  - `cpu_topk.h`      — top-K desc with smallest-index tie-break
  - `cpu_rope.h`      — HF OlmoeRotaryEmbedding rotate_half RoPE (scalar)
  - `cpu_sdpa.h`      — causal multi-head attention (AVX512 dot/axpy/normalize;
                        caller-owned score slices; full + incremental variants)
  - `cpu_silu.h`      — SiLU activation (scalar)
  - `cpu_argmax.h`    — greedy argmax sampling (AVX512), used by the REPL

Auxiliary integration helpers (not kernels) live in `engine_forward.c`:
`add_residual` (AVX512 masked add) and the MoE blocks
(`moe_block_prefill` / `moe_block_decode`).

## Public API (`src/olmoe/engine/engine.h`)

### Activation dtype

```c
typedef float olmoe_act_t;
```

Activations are kept in FP32 throughout the engine. Weights remain BF16
(`olmoe_bf16_t`) and are promoted on the fly by the per-op impls. Swapping
the activation dtype to BF16 later only needs a one-line change here; no
call site is touched because every signature uses `olmoe_act_t`.

### Status

```c
typedef enum {
    OLMOE_OK, OLMOE_ERR_NULL, OLMOE_ERR_SHAPE, OLMOE_ERR_ALLOC
} olmoe_status_t;
```

### Scratch (caller-owned activations)

```c
typedef struct { /* every buffer is capacity seq_len along the seq dim */
    olmoe_act_t *hidden_in, *hidden_out;   /* [seq, hidden]     */
    olmoe_act_t *q, *k, *v, *ctx;          /* [seq, hidden]     */
    olmoe_act_t *scores;                   /* [OLMOE_NUM_HEADS * max(seq_len, max_cache_len)]
                                              per-head SDPA score slices */
    olmoe_act_t *router_logits;            /* [seq, n_experts]  */
    int         *topk_idx;                 /* [seq, K]          */
    olmoe_act_t *topk_w;                   /* [seq, K]          */
    olmoe_act_t *expert_in, *expert_out;   /* [seq, inter / hidden] */
    olmoe_act_t *logits;                   /* [seq, vocab]      */
    size_t       seq_len;
    olmoe_act_t *cache_k, *cache_v;        /* [OLMOE_N_LAYERS * max_cache_len * hidden]
                                              per-layer K/V caches (NULL if no cache) */
    size_t       max_cache_len;            /* KV cache capacity per layer (0 = no cache) */
    size_t       cache_len;                /* current cached token count */
} olmoe_scratch_t;

olmoe_status_t olmoe_scratch_init(olmoe_scratch_t *s, size_t seq_len,
                                  size_t max_cache_len);
void           olmoe_scratch_free(olmoe_scratch_t *s);
```

The engine never `malloc`s activations. The caller sizes a scratch once
for the largest `seq_len` it expects and reuses it across forward calls.
Pass `max_cache_len=0` to skip KV-cache allocation (the engine falls back
to full-recompute for every call). `olmoe_scratch_free` is NULL-safe and
zeroes the struct on return.

### `seq_len == 0` contract

For the per-kind ops, a `seq_len`/`n_tokens` of 0 is a no-op: they write
no rows, so it is legal to call them with NULL/zeroed buffers (covered by
`test_leaf_forwards_zero_seq_no_op`). The top-level
`olmoe_forward` is stricter — passing a NULL `model`, `token_ids`,
`scratch`, or `logits_out` is a programming error regardless of `seq_len`.

### Per-kind ops (one function per `olmoe_slot_kind_t`)

The 15 tensor kinds baked by `generate_model_layout.py` each get one
forward function. This 1:1 mapping keeps per-tensor unit tests trivial and
makes it impossible to forget a kind (the test suite checks NULL-safety
on every public symbol).

| `olmoe_slot_kind_t`       | function                       |
|---------------------------|--------------------------------|
| `OLMOE_KIND_EMBED`        | `olmoe_embed_forward`          |
| `OLMOE_KIND_LM_HEAD`      | `olmoe_lm_head_forward`        |
| `OLMOE_KIND_NORM`         | `olmoe_final_norm_forward`     |
| `OLMOE_KIND_INPUT_LN`     | `olmoe_input_ln_forward`       |
| `OLMOE_KIND_POST_LN`      | `olmoe_post_ln_forward`        |
| `OLMOE_KIND_Q_PROJ`       | `olmoe_q_proj_forward`         |
| `OLMOE_KIND_K_PROJ`       | `olmoe_k_proj_forward`         |
| `OLMOE_KIND_V_PROJ`       | `olmoe_v_proj_forward`         |
| `OLMOE_KIND_O_PROJ`       | `olmoe_o_proj_forward`         |
| `OLMOE_KIND_Q_NORM`       | `olmoe_q_norm_forward`         |
| `OLMOE_KIND_K_NORM`       | `olmoe_k_norm_forward`         |
| `OLMOE_KIND_MLP_GATE`     | `olmoe_mlp_gate_forward`       |
| `OLMOE_KIND_EXPERT_GATE`  | `olmoe_expert_gate_forward`    |
| `OLMOE_KIND_EXPERT_UP`    | `olmoe_expert_up_forward`      |
| `OLMOE_KIND_EXPERT_DOWN`  | `olmoe_expert_down_forward`    |

`q_norm`/`k_norm` apply RMSNorm in-place over their respective Q/K
buffers; the remaining norms write into a separate `out`. The MoE
router (`mlp_gate_forward`) writes both the top-K expert indices and the
(softmax-normalized) weights.

### Orchestrator

```c
olmoe_status_t olmoe_forward(const olmoe_model_t *m, const int *token_ids,
                             size_t seq_len, size_t pos,
                             olmoe_scratch_t *scratch,
                             olmoe_act_t *logits_out);
```

`pos` is the absolute position of the first token within the full
sequence: pass `0` for prefill and the current cached length for
incremental decode. When the scratch has a KV cache (`max_cache_len > 0`)
the engine appends K/V for the new tokens to the cache and runs
`cpu_sdpa_incremental`; without one it recomputes full causal SDPA every
call. The cache-capacity guard (`pos + seq_len > max_cache_len`) trips
`OLMOE_ERR_SHAPE`. After the call the scratch's `cache_len` is advanced.

Wires the per-layer ops in OLMoE forward order using `scratch` for
intermediate storage: `embed -> [layer -> input_ln, q/k/v_proj, q/k_norm,
RoPE q/k, causal/incremental SDPA, o_proj, residual, post_ln, mlp_gate,
MoE top-K dispatch with SiLU gating and weighted down-projection,
residual] -> final_norm -> lm_head -> logits`. The per-layer loop is
driven by the baked `OLMOE_N_LAYERS` constant.

## Topology constants

Several OLMoE-1B-7B-0924-Instruct topology fields are NOT derivable from
the safetensors index (q/k/v_proj are all `[hidden, hidden]`, so head
count / head dim / rope_theta cannot be recovered from tensor shapes).
They are baked in `engine.h` rather than `model_layout.inc` so the router
and loader never read `config.json` at runtime:

  - `OLMOE_N_EXPERTS_PER_TOK` (=8) — per-token routing arity
  - `OLMOE_NUM_HEADS`   (=16) — attention head count
  - `OLMOE_HEAD_DIM`    (=128) — per-head dimension
  - `OLMOE_ROPE_THETA`  (=10000.0) — rotary base frequency

## File layout

```
src/
    kernels/
        kernels.h            shared AVX512 helpers (BF16->FP32 converter)
        cpu_matmul.h         cpu_matmul_bf16 (AVX512-BF16 + FMA)
        cpu_rmsnorm.h        cpu_rmsnorm (AVX512)
        cpu_softmax.h        cpu_softmax
        cpu_topk.h           cpu_topk_desc
        cpu_rope.h           cpu_rope (HF rotate_half)
        cpu_sdpa.h           cpu_sdpa (causal MHA) + cpu_sdpa_incremental
        cpu_silu.h           cpu_silu
        cpu_argmax.h         cpu_argmax (greedy argmax sampling, AVX512)
    olmoe/engine/
        engine.h             public API (olmoe_act_t, status, scratch, 15 ops)
        engine_internal.h    shared internal helpers (safe_array_size)
        engine_embed.c       embed + lm_head
        engine_norm.c        final / input / post / q / k norms
        engine_attn.c        q / k / v / o projections
        engine_mlp.c         mlp router + 3 expert weight ops
        engine_forward.c     olmoe_forward integrator + scratch_init / free,
                             attention_block, add_residual, MoE blocks
```

Each `.c` is well under the 500-line limit; math kernels are header-only
inlines so no god file accumulates.

## Tests (`tests/test_engine*.c`)

Compiled into the unified `tests/test_main.c` runner alongside the
tokenizer and REPL suites. Engine categories:

  - `test_engine.c` — scratch init/free, forward zero-seq / oversize-seq
    contract.
  - `test_engine_matmul.c` — `cpu_matmul` and q/k/v/o + lm_head vs scalar.
  - `test_engine_norm.c` — input/post/final/q/k RMSNorm vs scalar.
  - `test_engine_mlp.c` — mlp_gate router (softmax+topk+renorm) and the
    three expert matmuls vs scalar.
  - `test_engine_fwd.c` — RoPE / SDPA / SiLU kernel unit tests vs scalar,
    plus a full end-to-end `olmoe_forward` vs a pure-C reference (no model
    load; a synthetic 1-layer model with shared nonzero expert weights keeps
    memory < 0.5 GiB).
  - `test_engine_argmax.c` — `cpu_argmax` sampling vs scalar.

The dispatcher `test_engine_stubs_pass()` sums the per-check failure
counts. No model dir is required for the engine tests, so they run
anywhere without the ~13 GiB safetensors shards. KV-cache / incremental
decode correctness is exercised indirectly through the REPL harness
(`tests/test_repl.c`) which drives `olmoe_forward` with a cache.

## Out of scope (explicit, deferred to follow-up PRs)

  - Batch dimension (`batch > 1`); each token buffer is a flat `[seq, ...]`
    sequence with no batch axis.
  - SIMD acceleration of RoPE and SiLU (still scalar); SDPA's inner
    dot/axpy/normalize, matmul, RMSNorm, softmax-BF16-promotion,
    `add_residual`, and argmax already use AVX512.

## Assumptions baked in (and where they are validated)

  - topology (hidden/inter/vocab/experts/layers): inherited from
    `model_layout.inc`; any drift trips the layer tests first.
  - `num_experts_per_tok = 8`, `num_heads = 16`, `head_dim = 128`,
    `rope_theta = 10000.0`: literals in `engine.h`, cross-checked against
    `config.json` at implementation time (not read at runtime).
  - seq_len only, batch = 1: documented in the `olmoe_forward` contract;
    `pos` + KV cache support incremental decode; a batch dim is a
    follow-up.