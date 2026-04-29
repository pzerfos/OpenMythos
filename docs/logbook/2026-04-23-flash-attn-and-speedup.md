# Flash Attention Integration & Throughput Investigation

**Date:** 2026-04-23
**Goal:** Integrate upstream Flash Attention 2 support, install on BlueVela, and investigate throughput bottlenecks.

---

## Upstream Merge

Merged 3 upstream commits (`7d78ebe`, `963e112`, `227dbb1`):
- **Flash Attention 2** in GQAttention — conditional import, native GQA handling, fallback to manual attention
- **Causal mask dtype fix** — mask now created in activation dtype (resolves code review item #9)
- Tests moved to `tests/`, new benchmarks and examples added
- Conflict resolution: preserved our FSDP dtype casts in the fallback attention path

## flash-attn Installation

Built `flash-attn 2.8.3` from source on BlueVela login node:
- `CUDA_HOME=/opt/share/cuda-13.0`
- Installed into conda env `openmythos`
- Import verified: `from flash_attn import flash_attn_func` works

## Training with Flash Attention (Job 29260)

4 GPUs, FSDP, micro_batch=1, grad_accum=4, flash-attn enabled.

```
step  4/305175 | loss 12.6406 | gnorm 4.08 | lr 4.50e-07
step 10/305175 | loss 12.6250 | gnorm 4.33 | lr 1.35e-06
step 15/305175 | loss 12.5312 | gnorm 4.42 | lr 2.10e-06
step 18/305175 | loss 12.4688 | gnorm 4.49 | lr 2.55e-06
```

Loss dropping (12.64 → 12.47), training stable.

### Key Finding: Flash Attention Does Not Improve Step Time

| Config | Step time |
|--------|-----------|
| Manual attention (job 26518) | ~16s/step |
| Flash attention (job 29260) | ~16s/step |

**Flash-attn is not the bottleneck.** At seq_len=2048 with micro_batch=1, attention is cheap. The dominant cost is the **MoE dispatch loop** — 256 Python iterations per forward pass (topk=4 x n_experts=64), executed 16 times per step (once per recurrent loop). That's 4,096 Python-level expert dispatches per forward, plus backward.

### Throughput Analysis

- ~16s/step, 32,768 tokens/step → ~2,048 tokens/s
- 10B tokens target → 305,175 steps → **~56 days**
- MoE dispatch loop is the critical path for speedup

---

## Code Review Fixes & MoE Optimization

Applied all actionable items from `docs/code-review-2026-04-23.md` plus the MoE dispatch optimization:

### MoE grouped dispatch (issue #4 — throughput bottleneck)

Replaced the nested Python loop (`for i in range(topk): for eid in range(n_experts):` — 256 iterations) with grouped/batched dispatch:

1. Flatten all topk (token, expert) pairs into `(N*topk,)` tensors
2. `argsort` by expert ID for contiguous grouping
3. `unique_consecutive` to find expert boundaries
4. Run each active expert once on its full batch
5. Scatter results back and sum over topk dim

Reduces from 256 Python iterations to at most `n_active_experts` (<=64, typically fewer), each processing a larger contiguous batch for better GPU utilization.

### Correctness fixes applied

| # | Fix | File(s) |
|---|-----|---------|
| 1 | ACT remainder for non-halted positions | `main.py:898-901` |
| 2 | MoE score renorm `.clamp(min=1e-9)` | `main.py:521` |
| 5 | Remove nonexistent `__init__.py` exports | `__init__.py` |
| 6 | Guard `fused=True` AdamW with CUDA check | both training scripts |
| 7 | LoRAAdapter.B defensive dtype cast | `main.py:625` |
| 8 | loop_index_embedding compute in float32 | `main.py:567-575` |

### Not addressed this session

- **Issue #3 (router_bias):** Load balancing bias is never updated. Deferred — requires design decision on training loop integration vs. documenting as disabled for PoC.
- **Issue #9 (causal mask dtype):** Already fixed by upstream flash-attn merge.
- **Issue #10 (RoPE lazy extension):** Deferred — no current need beyond `max_seq_len`.
- **Issue #11 (amp_ctx variable flow):** Cosmetic, deferred.

### Pre-existing test failures (14 tests)

- 13 from RoPE dimension mismatch in `apply_rope` — test configs produce incompatible `freqs_cis` shapes after the upstream flash-attn merge. Needs test config update.
- 1 from LTI spectral radius boundary: `A.max() == 1.0`, test uses strict `< 1.0`.

All 53 non-pre-existing tests pass, including full-model GQA/MLA forward, generate, KV cache, and depth extrapolation.

---

## MoE Dispatch Speedup Confirmed (job 33050 vs job 31808)

| Run | Step time | Dispatch method |
|-----|-----------|----------------|
| Pre-optimization (job 31808) | ~16.7s | Nested loop (4 × 64 = 256 iterations) |
| Post-optimization (job 33050) | ~2.3s | Grouped dispatch (sort + batch per expert) |

**6.7x speedup** from the MoE dispatch optimization alone. Both runs deadlocked
at ~step 33 due to a separate bug (ACT early exit FSDP deadlock — see
`2026-04-23-act-fsdp-deadlock.md`).

After fixing the ACT deadlock (commit `6c5659c`), ACT early exit restored a
further ~2.3x speedup (steps drop from ~2.3s to ~1.0s once halting kicks in).
**Combined: ~16x total speedup** over the original code.

## Next Steps

> **Update 2026-04-29:** canonical roadmap lives in
> `docs/logbook/2026-04-28-option-b-and-upstream-pr.md`. The near-term
> follow-ups below (#5 tests, #6 router_bias) are now resolved; #8 ACT vs
> depth was resolved by choosing Option B. Entries preserved as historical
> record.

### Completed this session
1. ~~**Benchmark MoE dispatch on GPU**~~ — Done: 6.7x confirmed (see above)
2. ~~**ACT FSDP deadlock**~~ — Diagnosed, fixed, validated (see `2026-04-23-act-fsdp-deadlock.md`)
3. ~~**Comprehensive test suite**~~ — 233 → 262 tests across all modules

### In progress
4. **Monitor job 34019** — 1B token run (ACT enabled, fixed `n_loops=16`),
   4 GPUs on preemptable. Step ~10,000 / 30,518 as of 01:16 UTC, loss 3.57.
   ETA ~5-6 hours to complete. Validates the full pipeline post-fixes.

### Near-term follow-ups
5. **Fix pre-existing test failures** — 14 tests in `tests/test_main.py`:
   13 from RoPE dimension mismatch after upstream flash-attn merge,
   1 from LTI spectral radius float32 boundary
6. **router_bias load balancing** (issue #3) — Implement DeepSeek-V3 style
   periodic bias updates in training loop, or document as disabled for PoC
7. **Scale to 16 GPUs on normal queue** — Once job 34019 validates the
   pipeline, submit a longer run with `-q normal -G grp_granite_` and 16 GPUs
   (~2.5 days for 10B tokens vs ~21 hours for 1B)

### Architecture decisions (informed by upstream findings)
8. **ACT vs depth extrapolation** (issue #5) — Upstream empirical work
   ([kyegomez/OpenMythos#28](https://github.com/kyegomez/OpenMythos/issues/28),
   13 ablation runs) conclusively shows ACT is the primary mechanism binding
   the model to its training loop count. Disabling ACT + random loop sampling
   is the only recipe that produces monotonic depth-scaling. Three options:
   - **A.** Keep ACT for adaptive compute, accept no depth extrapolation
     (current PoC approach)
   - **B.** Disable ACT, return final `h` directly, train with stochastic
     depth sampling — enables depth extrapolation but loses per-token
     adaptive compute
   - **C.** Replace ACT weighted sum with a soft attention over loop outputs
     trained jointly with stochastic depth — potentially gets both, untested
9. **RoPE lazy extension** (issue #3) — Not urgent until `seq_len > max_seq_len`
   is needed, but a correctness gap worth closing
