# MoE expert-accumulate data race

## Symptom

Repeated runs of `build/main` with the *same* prompt produced a different
number of generated tokens every run (e.g. 229, then 113, then 253). The REPL
samples greedily (`sample_argmax`), so the pipeline is fully deterministic
given identical forward results — varying token counts mean the forward pass
itself was non-deterministic.

## Root cause

Introduced by `aad46a2 perf(engine): collapse expert_accumulate loop with
OpenMP simd`, which changed the MoE dispatch pragma to:

```c
#pragma omp parallel for schedule(static) collapse(2)
for (size_t i = 0; i < seq; ++i) {          /* token                     */
    ...
    for (size_t r = 0; r < OLMOE_N_EXPERTS_PER_TOK; ++r) { /* 8 experts  */
        ...
        expert_accumulate(&L->experts[e], tok, acc, w, ...);
    }
}
```

`collapse(2)` fuses both loops into a `seq * 8` iteration space handed out to
all threads. Iterations with the same token `i` but different expert slot `r`
therefore run **concurrently on different threads**, and every one of them
accumulates into the same row `s->expert_out + i * OLMOE_HIDDEN` via the
non-atomic RMW in `expert_accumulate`:

```c
_mm512_storeu_ps(acc_row + h,
                 _mm512_fmadd_ps(vw, _mm512_loadu_ps(down + h),
                                 _mm512_loadu_ps(acc_row + h)));
```

Load → fmadd → store interleavings across threads drop updates; which updates
are lost depends on thread timing, so `expert_out` differs run-to-run. The
error is amplified worst during **decode** (`seq == 1`): the fused space is
just 8 iterations scattered over all threads, so *every* decode step races.
The tiny numeric drift eventually flips the greedy argmax at some token and
generation diverges (butterfly effect), producing wildly different token
counts.

Everything else was verified deterministic: the IREE tokenizer, the
`calloc`+`fread` loader, the matmul/rmsnorm/rope/sdpa kernels (disjoint
per-element/per-row writes, fixed reduction order), and softmax (fixed in
`47251f6`; additionally it is only ever invoked from inside a parallel region,
so it runs single-threaded).

## Fix (Option B: private accumulators + deterministic reduction)

The MoE block now has two parallelization shapes:

- **Decode** (`seq == 1`): parallelize over the 8 expert slots; each slot
  folds into its own `acc_slots[r]` row (disjoint writers → race-free), then
  the rows are summed into `s->expert_out` in ascending `r` order (fixed
  order → bit-reproducible).
- **Prefill** (`seq > 1`): parallelize over tokens; each token's 8 experts
  fold serially into that token's own row (disjoint across threads).

Keeps 8-way decode parallelism while removing the cross-thread RMW.
