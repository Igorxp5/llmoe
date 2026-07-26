# Engine module (`src/olmoe/engine/`)

Forward-pass engine for OLMoE. Reads BF16 weights from an
`olmoe_model_t` produced by the layer loader (`src/olmoe/layers/`) and
produces activations the caller owns.

## Status

**Fully implemented.** The top-level `olmoe_forward` is the sole validation
point (NULL + `scratch->seq_len >= seq_len` + `seq_len == 0` guards); the
14 per-kind ops are pure compute returning `void`, trusting the integrator
to have validated args. Computation lives in header-only kernels under
`src/olmoe/engine/kernels/`:

  - `cpu_matmul.h`   — BF16-weight x FP32-act matmul (AVX512-BF16 + FMA)
  - `cpu_rmsnorm.h`   — RMSNorm with BF16 weight promotion (AVX512)
  - `cpu_softmax.h`   — numerically stable row softmax
  - `cpu_topk.h`      — top-K desc with smallest-index tie-break
  - `cpu_rope.h`      — HF OlmoeRotaryEmbedding rotate_half RoPE (scalar)
  - `cpu_sdpa.h`      — causal multi-head attention (scalar, malloc scores)
  - `cpu_silu.h`      — SiLU activation (scalar)

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
    olmoe_act_t *router_logits;            /* [seq, n_experts]  */
    int         *topk_idx;                 /* [seq, K]          */
    olmoe_act_t *topk_w;                   /* [seq, K]          */
    olmoe_act_t *expert_in, *expert_out;   /* [seq, inter / hidden] */
    olmoe_act_t *logits;                   /* [seq, vocab]      */
    size_t       seq_len;
} olmoe_scratch_t;

olmoe_status_t olmoe_scratch_init(olmoe_scratch_t *s, size_t seq_len);
void           olmoe_scratch_free(olmoe_scratch_t *s);
```

The engine never `malloc`s activations. The caller sizes a scratch once
for the largest `seq_len` it expects and reuses it across forward calls.
`olmoe_scratch_free` is NULL-safe and zeroes the struct on return.

### `seq_len == 0` contract

For the per-kind ops, a `seq_len`/`n_tokens` of 0 short-circuits to
`OLMOE_OK` **before** the NULL-arg check: a no-op call is legal with NULL
buffers because nothing is read or written. The top-level
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
                             size_t seq_len, olmoe_scratch_t *scratch,
                             olmoe_act_t *logits_out);
```

Wires the per-kind ops in OLMoE forward order using `scratch` for
intermediate storage: `embed -> [per layer: input_ln, q/k/v_proj,
q/k_norm, RoPE q/k, causal SDPA, o_proj, residual, post_ln, mlp_gate, MoE
top-K expert dispatch with SiLU gating and weighted down-projection,
residual] -> final_norm -> lm_head -> logits`. The per-layer loop runs
`m->n_layers` iterations (not the baked `OLMOE_N_LAYERS`) so synthetic
low-RAM end-to-end validation can set `n_layers` smaller than the real
topology. `logits_out` is caller-owned (commonly `scratch->logits`).

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
src/olmoe/engine/
    engine.h             public API (olmoe_act_t, status, scratch, 15 ops)
    engine_internal.h    shared internal helpers (safe_array_size, BF16->FP32)
    engine_embed.c       embed + lm_head
    engine_norm.c        final / input / post / q / k norms
    engine_attn.c        q / k / v / o projections
    engine_mlp.c         mlp router + 3 expert weight ops
    engine_forward.c     olmoe_forward integrator + scratch_init / free
    kernels/
      cpu_matmul.h       cpu_matmul_bf16 (AVX512-BF16 + FMA)
      cpu_rmsnorm.h      cpu_rmsnorm (AVX512)
      cpu_softmax.h      cpu_softmax
      cpu_topk.h         cpu_topk_desc
      cpu_rope.h         cpu_rope (HF rotate_half)
      cpu_sdpa.h         cpu_sdpa (causal MHA)
      cpu_silu.h         cpu_silu
```

Each `.c` is well under the 500-line limit; math kernels are header-only
inlines so no god file accumulates.

## Tests (`tests/test_engine*.c`)

Compiled into the unified `tests/test_main.c` runner. Categories:

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

The dispatcher `test_engine_stubs_pass()` sums the per-check failure
counts. No model dir is required for the engine tests, so they run
anywhere without the ~13 GiB safetensors shards.

## Out of scope (explicit, deferred to follow-up PRs)

  - KV-cache, batch dim, incremental decode, incremental generation.
  - SIMD acceleration of RoPE / SDPA / SiLU (currently scalar; matmul,
    RMSNorm, softmax-BF16-promotion already use AVX512).

## Assumptions baked in (and where they are validated)

  - topology (hidden/inter/vocab/experts/layers): inherited from
    `model_layout.inc`; any drift trips the layer tests first.
  - `num_experts_per_tok = 8`, `num_heads = 16`, `head_dim = 128`,
    `rope_theta = 10000.0`: literals in `engine.h`, cross-checked against
    `config.json` at implementation time (not read at runtime).
  - seq_len only, batch = 1: documented in the `olmoe_forward` contract;
    a batch dim is a follow-up.