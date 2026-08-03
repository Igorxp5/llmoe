# Layer module (`src/olmoe/layers/`)

Loads an OLMoE-1B-7B model from `model-*.safetensors` shards into a typed
hierarchical in-RAM struct called `olmoe_model_t`. Weights are stored as the
raw BF16 bit pattern in a `uint16_t` (`olmoe_bf16_t`); no numeric
conversions happen at load time.

## Public API

`src/olmoe/layers/layers.h`:

    olmoe_model_t *olmoe_model_load(const char *dir);
    void           olmoe_model_free(olmoe_model_t *model);

`dir` is the directory that contains the shards (`model-00001-of-
00003.safetensors`, `model-00002-of-00003.safetensors`,
`model-00003-of-00003.safetensors`). The loader does **not** read
`model.safetensors.index.json`. All the indices and metadata needed to read
the shards directly are baked into the binary at build time by
`scripts/generate_model_layout.py` and stored in
`src/olmoe/layers/model_layout.inc`.

## Why no runtime JSON

By design the loader only reads raw BF16 bytes from `.safetensors` files.
JSON parsing (the index and the per-shard safetensors header) is done
**once at build time** by `generate_model_layout.py`. The output `.inc`
contains:

  - dim constants: `OLMOE_N_LAYERS`, `OLMOE_N_EXPERTS`, `OLMOE_HIDDEN`,
    `OLMOE_INTER`, `OLMOE_VOCAB`, `OLMOE_N_SHARDS`, `OLMOE_N_TOTAL_TENSORS`.
  - `olmoe_slot_kind_t` enum -- one value per tensor role
    (`EMBED`, `LM_HEAD`, `NORM`, `INPUT_LN`, `POST_LN`, `Q/K/V/O_PROJ`,
    `Q/K_NORM`, `MLP_GATE`, `EXPERT_GATE/UP/DOWN`).
  - `olmoe_tensor_desc_t { kind, layer, expert }` -- per-tensor position
    tag; `layer=-1` / `expert=-1` are sentinels for non-layer / non-expert
    tensors.
  - `OLMOE_SHARD_FILE_NAMES[]` -- sorted shard filenames, in the order the
    loader opens them at runtime.
  - `OLMOE_SHARD_LAYOUT[][N]` -- per-shard tensor desc arrays, in
    sorted-by-name order; within a safetensors shard this equals
    sorted-by-data-offset (verified at build time), so the loader can
    `fread` sequentially without per-tensor seeks.
  - `OLMOE_SHARD_LEN[]` -- number of tensors per shard.
  - `OLMOE_ORACLE_<TENSOR>[16]` -- raw BF16 uint16 samples for a few
    tensors used by `tests/test_layer.c` for value-equality tests.

The same generator is responsible for cross-checking the bytes-per-kind
formula against each tensor's actual `data_offsets[1] - data_offsets[0]`
span, and for asserting there is no inter-tensor padding, so any upstream
shape drift makes the build fail loudly.

## Why hierarchical structs

The OLMoE topology is fixed (16 layers, 64 experts each); shapes are
constant and derivable from dims. The typed struct
(`olmoe_expert_t`/`olmoe_self_attn_t`/`olmoe_layer_t`/`olmoe_model_t`) gives
the future forward pass ergonomic, named access (`m->layers[i].
self_attn.q_proj`) without per-field shape metadata (which would be
redundant with the compile-time constants). BF16 storage is a bare
`uint16_t*` per field, exactly as requested.

## Lifetime and ownership

The model is a single anonymous `mmap`'d region (~12.9 GiB, zero-initialized,
page-aligned), backed by hugetlb 2 MiB pages (`MAP_HUGETLB`) when the operator
reserved huge pages (`vm.nr_hugepages`) and falling back to plain 4 KiB pages
otherwise. Every weight field is a compile-time-sized array inside that
region; the loader `fread`s shard bytes directly into the right field offset.
There are no individual `malloc`s or per-tensor ownership tables.
`olmoe_model_free(model)` does a single `munmap` (NULL-safe).

On successful load the loader calls `mlock(model, sizeof *model)` so the
~13 GiB of weight pages are pinned in RAM and cannot be swapped out, then
`mprotect(PROT_READ)` seals the region read-only: the returned pointer is
`const` and any write to the model faults with SIGSEGV. `olmoe_model_free`
reverses this with `munlock` before `munmap`.

## Loading flow

Per shard (`src/olmoe/layers/layers.c`):

  1. open `dir + "/" + OLMOE_SHARD_FILE_NAMES[shard_idx]`
  2. read the 8-byte LE header-length, `fseek` past the JSON header
  3. walk `OLMOE_SHARD_LAYOUT[shard_idx]`, compute the target field's
     array address via `target_field(m, d)`, and `fread(bytes_for_kind(kind))`
     directly into it — no per-tensor `malloc`
  4. assert EOF after the last tensor (no trailing bytes -> layout matches)

## Oracle source of truth

The baked `OLMOE_ORACLE_*` BF16 uint16 samples produced by
`generate_model_layout.py` are the same data `scripts/inspect_safetensors.py
--dump-values` reads from the shards (the script applies a `bf16_to_f32`
helper for display; the values printed are the f32 interpretation of the
exact bytes baked here). At dev time the generator output was
hand-verified against the inspect dump (e.g. `0x400f` == 2.234375 for
`model.norm.weight[0]`).

## Tests

`tests/test_layer.c` (compiled into the unified `tests/test_main.c`
runner) loads the model once and runs these checks:

  - `test_top_level_tensors_nonzero`
  - `test_every_layer_self_attn_nonzero`
  - `test_every_layer_layernorms_nonzero`
  - `test_every_layer_mlp_gate_nonzero`
  - `test_every_expert_tensor_nonzero`
  - `test_norm_first_values_match_oracle`
  - `test_lm_head_first_values_match_oracle`
  - `test_expert_zero_down_values_match_oracle`
  - `test_embed_lookup_matches_oracle`

The model dir defaults to `models/OLMoE-1B-7B-0924-Instruct` and is
overridable via the `OLMOE_TEST_MODEL_DIR` environment variable.

## Assumptions baked in (and where they are validated)

  - shard presence and order: assumes exactly `OLMOE_N_SHARDS=3` files
    matching `OLMOE_SHARD_FILE_NAMES[]` exist in `dir`. Drift detected at
    open failure in the loader.
  - in-shard tensor order: sorted-by-name equals sorted-by-offset, with no
    inter-tensor padding. Generator refuses to emit if this fails.
  - tensor byte size per role: derived from dim formulas. Generator
    cross-checks each span against the shard header before emitting.
  - post-last-tensor EOF: loader asserts no extra bytes remain in a shard
    after the expected tensors are read.