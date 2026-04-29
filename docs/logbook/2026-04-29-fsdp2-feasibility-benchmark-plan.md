# FSDP2 Feasibility Benchmark — Plan

**Date:** 2026-04-29
**Status:** Planned. Execute next session.
**Goal:** Determine empirically whether FSDP2 + `torch.compile` produces a real speedup on OpenMythos given our dynamic-shape hotspots (variable `n_loops` from stochastic_depth, variable per-expert batch sizes from MoE), before committing to a full FSDP1 → FSDP2 migration.

---

## Prerequisites (verified 2026-04-29)

BlueVela conda env `openmythos`:

- `torch 2.11.0+cu130`
- `torch.compile` available
- `torch.distributed.fsdp.fully_shard` importable (FSDP2 API)

All prerequisites for the benchmark are already present. No env upgrade required.

## Expected benefits (what we are testing for)

If the benchmark confirms these, FSDP2 migration is worth scheduling before the next scale-up:

- **`torch.compile` survives dynamic shapes.** Target: 20–40% step-time reduction on the recurrent block + MoE dispatch without recompilation storms.
- **Lower peak memory** than FSDP1 (no FlatParameter allocation overhead). Could enable bumping `micro_batch` above 1.
- **Cleaner state_dict** — separate from the benchmark, but a useful secondary check via save/load round-trip.

Risks the benchmark is designed to surface:

- Variable `n_loops` from stochastic_depth triggers per-shape recompilation on every step → `p95 >> median`, and the full-run time is *worse* than FSDP1.
- Variable per-expert batch sizes from grouped MoE dispatch force recompilation inside each expert call.
- Existing checkpoints (e.g. `step_0121000.pt`, saved under FSDP1) fail to load into FSDP2.

## Benchmark design

### Scope

Standalone script at `evaluations/bench_fsdp_modes.py`, *not* integrated into the training pipeline. Uses random synthetic tokens — no dataset loading, no ClearML, no checkpoint save/load. Keeps the comparison clean.

1B config (matches current training run). 8 GPUs on a single BlueVela node, preemptable queue.

### Test matrix

`{fsdp1, fsdp2, fsdp2_compile} × {fixed_n_loops, stochastic_n_loops}` = 6 runs.

- `fsdp1`: current production path (`FSDP(model, ...)` with the existing `MixedPrecision` policy from `training/1b_poc_fineweb.py`).
- `fsdp2`: `fully_shard(model, mesh=..., mp_policy=...)` applied per-submodule, no compile. Baseline for FSDP2 without compile tax.
- `fsdp2_compile`: same as above, then `torch.compile(model, dynamic=True)` to hint at dynamic shapes.
- `fixed_n_loops=16`: matches the ACT regime (constant depth).
- `stochastic_n_loops`: matches the current production regime (`random.randint(1, 32)` per step, broadcast from rank 0 — same as training).

### Measurements per run

- 10 warmup steps (excluded from stats — let `torch.compile` finish its initial traces).
- 50 timed forward+backward+optimizer-step passes.
- Report **median, p5, p95** step time, plus **peak allocated GPU memory** and total recompilation count from `torch._dynamo.utils` if available.
- Synthetic data: `torch.randint(0, vocab_size, (B, T))` with the production `B=1`, `T=2048`.

### Pass/fail criteria

- **Green light for full migration:** `fsdp2_compile × stochastic_n_loops` median ≥15% faster than `fsdp1 × stochastic_n_loops` median, with `p95 / median` ≤ 1.5 (no recompilation storms).
- **Yellow (migrate but skip compile):** `fsdp2` beats `fsdp1` on memory and matches on speed, but `fsdp2_compile × stochastic_n_loops` shows `p95 / median > 2`. Plan: migrate to FSDP2 for the cleaner state_dict and memory, leave compile disabled.
- **Red (stay on FSDP1):** `fsdp2_compile × fixed_n_loops` isn't materially faster than FSDP1, or FSDP2 shows unexpected OOMs / correctness issues.

## Deliverables

- Branch: `bench/fsdp2-feasibility`.
- `evaluations/bench_fsdp_modes.py` — single script, `--mode` + `--n-loops-mode` CLI flags.
- `deploy/bluevela/bsub_bench_fsdp.sh` — submits one job that runs all 6 modes sequentially (~30 min wall clock).
- A follow-up logbook entry recording results and the migrate-or-not decision.

## Explicitly out of scope for the feasibility phase

- Checkpoint save/load refactor (FSDP2 uses `get_state_dict` / `set_state_dict` helpers; too much surface for a feasibility test).
- ClearML integration in the benchmark.
- Multi-node (blaunch) paths.
- Production training script changes. Anything that affects `training/1b_poc_fineweb.py` waits until after the decision.

## Time estimate

- Write benchmark + bsub wrapper: 20–30 min.
- BlueVela wall clock: ~30 min (6 modes × ~5 min each sequentially).
- Results analysis + logbook follow-up: ~10 min.

---

## Other still-open items from 2026-04-28 roadmap

1. **lm-eval-harness integration** — best deferred until the 10B run (job 67208) completes.
2. **Study Gated DeltaNet** — research direction, low urgency.
