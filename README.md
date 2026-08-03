# llmoe

An OLMoE inference engine written in C — a personal playground for studying how a modern Mixture-of-Experts (MoE) language model works under the hood.

This is a learning project. It deliberately keeps the model architecture *fixed* and *hardcoded* so the code stays readable and every piece (tokenizer, loader, forward-pass engine, greedy decoder) can be studied in isolation. Weights are stored raw as BF16 and the engine keeps activations in FP32 throughout.

It targets **Linux x86_64 only**, and uses AVX-512 SIMD kernels extensively.

## Features

- **Tokenizer** — HuggingFace `tokenizer.json` baked into the binary at build time as a C byte array (via [IREE](https://github.com/iree-org/iree)), so no JSON parser runs at runtime.
- **Model loader** — reads raw BF16 bytes from `model-*.safetensors` shards directly into a single ~12.9 GiB `calloc`'d block, with no runtime JSON. The shard layout is baked at build time into `model_layout.inc`.
- **Forward engine** — `embed → 16 layers → final_norm → lm_head → logits`, with KV-cache supported incremental decode and MoE top-8 routing. Compute uses AVX-512-BF16 matmul, AVX-512 RMSNorm/softmax/SDPA/argmax kernels.
- **Interactive REPL** — a chat-style prompt loop that tokenizes your input, runs the chat template, prefills, and decodes until EOS.
- **Deterministic** — greedy argmax sampling; output is reproducible across runs.

```
src/
    kernels/          header-only AVX-512 SIMD math kernels
    olmoe/
        engine/       forward-pass engine (15 per-kind ops + orchestrator)
        layers/       safetensors loader -> olmoe_model_t
        tokenizer/    embedded tokenizer (IREE-built)
    main.c            CLI entry point
    repl.c            interactive decode loop
scripts/              codegen + tokenizer oracle + benchmark tooling
tests/                unified test runner
benchmarks/           decode-speed golden + metrics CSV
```

## Requirements

Only Linux (x86_64) is supported. You need:

| tool | version used |
|------|--------------|
| `gcc` | 14.x (with `-fopenmp`, AVX-512) |
| `clang` / `clang++` | 18.x (used to build the IREE runtime) |
| `cmake` | 3.31+ |
| `ninja` | 1.11+ |
| `git` (with `git-lfs`) | 3.7+ |
| `python3` + `venv` | 3.12 |

On Ubuntu / Debian:

```bash
sudo apt update
sudo apt install -y build-essential clang cmake ninja-build git \
    git-lfs python3 python3-venv python3-pip libopenmp-dev grep findutils
```

Then initialize `git-lfs` (large-file storage, needed to fetch the model's safetensors shards):

```bash
git lfs install
```

Install the bundled `tokenizers` Python package (the test-suite's tokenizer oracle; only needed if you rebuild the tokenizer expectations):

```bash
python3 -m venv .venv
.venv/bin/pip install tokenizers
```

## Getting the model

The project currently supports **one** model:

- **`OLMoE-1B-7B-0924-Instruct`** — by AI2 ([huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct](https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct))

This project depends on the 3 shards of `model-*.safetensors` (~13 GiB), a `tokenizer.json`, and `config.json`, all stored via **Git LFS**. The `models/` directory already points at the upstream HuggingFace repo (its `origin` is `https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct`), so you can fetch the weights straight from upstream:

```bash
cd models/OLMoE-1B-7B-0924-Instruct
git lfs install
git pull origin main
git lfs pull
```

If you cloned the `models/` directory but the large blobs are empty (only LFS pointer files present), `git lfs pull` downloads the real shards. Confirm the large files are present with:

```bash
git lfs ls-files -l
```

The `models/OLMoE-1B-7B-0924-Instruct/` path is the default expected by the build (`Makefile`) and tests, so keep it there.

## Getting the code and building

Clone with submodules (the vendor submodules live under `vendor/`):

```bash
git clone --recurse-submodules git@github.com:Igorxp5/llmoe.git
cd llmoe

# if you already cloned without submodules:
git submodule update --init --recursive
```

Then run the test suite (this also validates the toolchain, the codegen scripts, and compiles everything):

```bash
make test
```

`make test` does a lot on first run: it configures and builds the vendored IREE runtime with CMake (takes a while, one time only, stamped at `build/iree-install/.stamp`), generates the embedded tokenizer data and the baked model layout, compiles the project, and runs every test.

You can build the main binary explicitly with:

```bash
make          # produces build/main, build/llmoe and build/test_runner
```

## Running the REPL

```bash
./build/llmoe models/OLMoE-1B-7B-0924-Instruct
```

Type a prompt, press a key, and the model streams its reply (decoded tokens) to `stdout`; debug/timing info goes to `stderr`. On each turn it prints `speed: N tok/s`. Press Ctrl-C or Ctrl-D to exit.

## Running the tests

```bash
make test
```

This builds and runs the unified runner `build/test_runner`, covering:

- tokenizer (round-trip + oracle vs `tokenizers`)
- model loader (oracle value checks against the shards)
- engine forward pass (per-op vs scalar reference, zero-seq / oversize-seq contracts)
- REPL integration (KV-cache incremental decode)
- argmax sampling, norms, matmuls, MoE router vs scalar

The engine/tokenizer tests need **no** model shards and run on any machine; only `test_layer` (loader) requires a `models/OLMoE-*` dir present, overridable via the `OLMOE_TEST_MODEL_DIR` environment variable.

> Per `AGENTS.md`, run the **full** suite (`make test`) after any change — never only the tests in scope.

## Benchmark

Decode speed is benchmarked and its output asserted deterministic against a golden file. To run and record a new row:

```bash
./scripts/benchmark.sh
```

The script filesystem-file fingerprint includes HW fingerprint (CPU name, cores/threads, RAM) and appends `commit, tok/s, ...` to `benchmarks/metrics.csv`. Pass `--sudo-hw` to also read RAM speed/slots (needs root). If the generated text diverges from the committed goldens baseline (`benchmarks/benchmark.expected.txt`), the run fails instead of recording — re-baseline after a legitimate output change with `BENCH_REGEN=1`.

## Last benchmark result

Machine: **AMD Ryzen AI 9 HX 370** (12 physical cores / 24 threads), 32 GiB RAM (2× 5600 MT/s), CPU backend via `./build/llmoe`.

| commit | avg tokens/s | backend | cpu_cores_threads | ram_mb |
|--------|--------------|---------|-------------------|--------|
| `49a9b48` | **27.39** | CPU | 12c/16t | 36864 |

The value is the average of the 8 runs recorded for that commit. Full history lives in `benchmarks/metrics.csv`. The benchmark harness writes the system fingerprint line each run.

## Documentation

Design notes live in `docs/`:

- `engine_module.md` — the forward-pass engine and kernels architecture
- `layer_module.md` — the safetensors loader / model layout, including an operator guide for reserving 2 MiB huge pages (`vm.nr_hugepages`)
- `moe_expert_accumulate_race.md` — notes on the MoE decode accumulation

## License

MIT — see [LICENSE](LICENSE). This applies to the code in this repo only; the vendored IREE runtime and the OLMoE model carry their own upstream licenses.
