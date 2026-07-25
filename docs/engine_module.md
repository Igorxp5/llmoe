# Engine module (`src/olmoe/engine/`)

Forward-pass engine for OLMoE. Reads BF16 weights from an
`olmoe_model_t` produced by the layer loader (`src/olmoe/layers/`) and
produces activations the caller owns.

## Status

**Stubs only.** Every public function validates its inputs and returns
`OLMOE_OK` (or `OLMOE_ERR_NULL` / `OLMOE_ERR_SHAPE` on contract violation)
without performing any computation. Real impls land in later PRs; until
then `make test` exercises the NULL-safety and capacity contract only.

## Public API (`src/olmoe/engine/engine.h`)

### Activation dtype

```c
typedef float olmoe_act_t;
```

Activations are kept in FP32 during the stub phase. Weights remain BF16
(`olmoe_bf16_t`) and are promoted on the fly by the per-op impls when they
land. Swapping the activation dtype to BF16 later only needs a one-line
change here; no call site is touched because every signature uses
`olmoe_act_t`.

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
intermediate storage. Stubs: returns `OLMOE_OK` after NULL and
`scratch->seq_len >= seq_len` checks. `logits_out` is caller-owned
(commonly `scratch->logits`).

## Topology constant

`OLMOE_N_EXPERTS_PER_TOK` (=8) lives in `engine.h`, not in
`model_layout.inc`: the layout generator only bakes quantities derivable
from the safetensors index, and per-token routing arity is a config field
(`num_experts_per_tok`), not a tensor shape.

## File layout

```
src/olmoe/engine/
    engine.h             public API (olmoe_act_t, status, scratch, 15 ops)
    engine_internal.h    shared internal helpers (safe_array_size)
    engine_embed.c       embed + lm_head
    engine_norm.c        final / input / post / q / k norms
    engine_attn.c        q / k / v / o projections
    engine_mlp.c         mlp router + 3 expert weight ops
    engine_forward.c     olmoe_forward + scratch_init / free
```

Each `.c` is well under the 500-line limit even after the real impls land
(norm/RMSNorm/math stay in their own sub-modules; no god file).

## Tests (`tests/test_engine.c`)

Compiled into the unified `tests/test_main.c` runner. Currently:

  - `test_scratch_init_free_roundtrip`
  - `test_scratch_init_null_returns_err`
  - `test_scratch_free_null_is_safe`
  - `test_null_input_returns_err`
  - `test_empty_seq_returns_ok`
  - `test_forward_stub_returns_ok`
  - `test_forward_oversize_seq_returns_shape`

The dispatcher `test_engine_stubs_pass()` sums the per-check failure
counts, matching the `test_layer.h` pattern. No model dir is required
(the stubs do not load weights), so these run anywhere without the
~13 GiB safetensors shards.

## Out of scope (explicit, deferred to follow-up PRs)

  - Any real computation: matmul, RMSNorm, RoPE, sdpa, top-K,
    softmax, silu.
  - KV-cache, batch dim, incremental decode.
  - FP32 ↔ BF16 conversion utility.
  - SIMD / `immintrin.h`.

## Assumptions baked in (and where they are validated)

  - topology (hidden/inter/vocab/experts/layers): inherited from
    `model_layout.inc`; any drift trips the layer tests first.
  - `num_experts_per_tok = 8`: literal in `engine.h`; validated by the
    first real router impl against `config.json` once the matmul PR lands.
  - seq_len only, batch = 1: documented in the `olmoe_forward` contract;
    a batch dim is a follow-up.