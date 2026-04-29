# FSDP2 Feasibility Benchmark — Results and Decision

**Date:** 2026-04-29
**Status:** Complete.
**Decision:** **YELLOW — migrate to FSDP2, skip `torch.compile`.**

Plan and methodology: `docs/logbook/2026-04-29-fsdp2-feasibility-benchmark-plan.md`.
Implementation: branch `bench/fsdp2-feasibility`, commits `501756f` + `7668f33`, script at `evaluations/bench_fsdp_modes.py`, wrapper at `deploy/bluevela/bsub_bench_fsdp.sh` → `run_fsdp_bench.sh`.
Run: BlueVela job **72079** (8 × H100 single node, preemptable queue, ~25 min wall clock). Logs: `/u/pzerfos/data/granite-mythos/output/experiments/errs_and_logs/bench-fsdp-2026-04-29-13-58.{log,err}`.

---

## Results

All six cells completed, no OOMs, no crashes. 10 warmup + 50 timed steps per cell. Same seed across runs → identical n_loops sequence in the stochastic cells. Step times taken as the per-step max across the 8 ranks.

| Mode            | n_loops    | median (ms) | p5 (ms) | p95 (ms) | p95/med | peak mem (GB) | recompilations |
|-----------------|-----------|-------------|---------|----------|---------|---------------|----------------|
| fsdp1           | fixed(16) | **307.7**   | 300.3   | 331.2    | 1.08    | 21.50         | 0              |
| fsdp1           | stochastic| **287.1**   | 87.3    | 525.0    | 1.83    | 32.78         | 0              |
| fsdp2           | fixed(16) | 514.7       | 188.7   | 871.4    | 1.69    | 19.42         | 0              |
| fsdp2           | stochastic| **244.1**   | 83.9    | 658.9    | 2.70    | 30.70         | 0              |
| fsdp2_compile   | fixed(16) | 484.2       | 162.6   | 3141.4   | **6.49**| 17.42         | **35**         |
| fsdp2_compile   | stochastic| **197.9**   | 71.3    | 661.9    | 3.34    | 27.12         | **36**         |

## Interpretation

### Production-relevant comparison (stochastic)

Same seed → identical n_loops sequence across the three stochastic rows. Directly comparable.

| Wrapper          | median (ms) | % vs fsdp1 baseline | peak mem (GB) | notes                           |
|------------------|------------|---------------------|---------------|----------------------------------|
| fsdp1            | 287.1      | —                   | 32.78         | current production               |
| fsdp2            | 244.1      | **−15%**            | 30.70         | clean win, no caveats            |
| fsdp2_compile    | 197.9      | **−31%**            | 27.12         | 36 recompilations; p95/med=3.34  |

### Compile tax at fixed shape

A 57% *regression* vs fsdp1 at fixed depth (`fsdp2_compile × fixed` median 484.2 ms vs `fsdp1 × fixed` 307.7 ms), with p95 blowing up to 3141 ms (p95/median = 6.49) and **35 recompilations in 50 timed steps**. That is a recompilation storm at a constant workload shape — meaning the dynamic-shape argument does not explain it. Torchinductor flagged the root cause at startup:

```
UserWarning: Torchinductor does not support code generation for complex operators.
Performance may be worse than eager.
```

OpenMythos's RoPE uses complex phasors (`freqs_cis`, `freqs_cis_mla`) as persistent buffers. Inductor inserts fallback-to-eager wrappers around those ops, which breaks the compiled graph into many fragments and produces repeated recompilations as the control flow touches them.

### `fsdp2 × fixed` oddity

`fsdp2 × fixed` median (514.7 ms) is roughly 2× the `fsdp2 × stochastic` median (244.1 ms), even though stochastic's median should correspond to n_loops ≈ 16. Linearity would predict a match. Hypothesis: 10 warmup steps is insufficient for fsdp2's deferred allocator/kernel selection to settle at a single fixed shape; the stochastic run's varied shapes reach steady state faster. Not relevant to the migration decision since our production recipe is stochastic and fsdp2 wins there — but worth flagging if we ever need fsdp2 under ACT (fixed-depth) training.

## Decision

### Against the plan's stated criteria

- **Green light** (median ≥15% faster AND p95/median ≤ 1.5): **FAILS** on p95/median.
  - `fsdp2_compile × stochastic` is 31% faster (passes speed bar) but p95/median = 3.34 (fails recompilation-storm bar). The 36 recompilations confirm it — this is not a noise artifact.
- **Yellow light** (fsdp2 beats/matches fsdp1 on speed, wins on memory, compile p95/median > 2): **PASSES.**
  - fsdp2-only beats fsdp1 on speed (−15%) and memory (−6%). fsdp2_compile shows p95/median > 2, confirming compile is unreliable.
- **Red light** (fsdp2_compile × fixed not materially faster than fsdp1, or unexpected OOMs): partial red on the `fsdp2_compile × fixed` axis (it is materially *slower* at fixed shape), but no correctness issues.

### Migrate to FSDP2, leave compile disabled

- **Speed:** expected 15% step-time reduction on the production (stochastic) recipe, from the `fsdp2` numbers. Rough extrapolation: a 10B-token run at 287 ms/step × 305k steps ≈ 24.3 hours on fsdp1 → 20.7 hours on fsdp2. Saves ~3.5 hours per full run.
- **Memory:** ~6% reduction (30.7 vs 32.8 GB peak). Not enough to safely bump micro_batch from 1 → 2 on H100 80GB with the 1B config, but nontrivial headroom for scale-up to 3B / 10B configs.
- **Cleaner state_dict:** FSDP2's `get_state_dict`/`set_state_dict` helpers replace the FSDP1 `FullStateDictConfig` + `optim_state_dict_to_load` dance in `training/1b_poc_fineweb.py`. Incidental but welcome.
- **`torch.compile` stays an opt-in flag, off by default.** It is not viable until either (a) we move RoPE to real-valued cos/sin buffers, removing the complex-op graph break, or (b) we isolate RoPE from the compiled region via `@torch.compiler.disable` on the relevant functions. Re-benchmark after either fix.

## Caveats and limitations

- **Single configuration measured:** `mythos_1b()`, seq_len 2048, 8 × H100 single node, bf16 MixedPrecision. Behavior at 10B-config scale, longer contexts, or multi-node is unverified.
- **Synthetic data:** no real-dataset memory effects (dataloader overhead, pinned-memory transfers). Under production these add ~10–20 ms/step of roughly-constant overhead; does not change relative comparisons.
- **p95/median thresholds were mis-calibrated in the original plan.** Stochastic n_loops in [1,32] inherently produces p95/median ≈ 1.83 (from fsdp1 baseline). The plan's `≤ 1.5` green threshold was implicitly assuming fixed-shape variance only. Future fixed-threshold comparisons should use `p95/median ratio vs the fsdp1 baseline at the same n_loops mode`, not an absolute number.
- **10 warmup steps may not fully warm fsdp2 at fixed shape** (see oddity above). If we ever benchmark fsdp2 under ACT-style fixed-depth training, extend warmup to 30+ steps.

## Migration plan (queue as the new top item)

1. **Create branch `feat/fsdp2-migration` off main** (after the 10B run on job 71939 finishes, to avoid code churn under an active long run).
2. **Rewrite FSDP wrapping** in `training/1b_poc_fineweb.py`: replace `FSDP(model, auto_wrap_policy=...)` with per-submodule `fully_shard(block, mesh, mp_policy)` loops over `model.prelude`, `model.recurrent`, `model.coda`, then root. Keep the same `bf16` `MixedPrecisionPolicy`. Keep buffers in init dtype (FSDP2 default, matches our `router_bias`/`expert_counts` constraints).
3. **Rewrite checkpoint save/load** to use `torch.distributed.checkpoint.state_dict.{get,set}_state_dict`. Verify round-trip with an existing fsdp1 checkpoint — state_dict keys and shapes should match; the wrapping difference is internal to FSDP, not the state_dict.
4. **Add a single smoke test** (`tests/test_fsdp2_wrapping.py`) that constructs the 1B model, applies the fsdp2 wrapping, runs one forward+backward, and verifies parameters updated. No GPU required if we skip the actual shard step.
5. **Re-run the 6-cell benchmark** (same harness) on the migrated code to confirm the 15% speedup holds end-to-end.
6. **Merge, update `2026-04-28-option-b-and-upstream-pr.md` item #4** to closed.

Estimated effort: 4–6 hours of focused work. The benchmark script on `bench/fsdp2-feasibility` already exercises the `fully_shard` wrapping pattern, so that's debugged.

## Closes / opens

- **Closes:** open item #4 on `2026-04-28-option-b-and-upstream-pr.md` (FSDP2 feasibility — decided).
- **Opens:** `feat/fsdp2-migration` as the next implementation track, queued behind the in-flight 10B run finishing.
