# NoPE Ablation — Queued for Launch on or after 2026-05-01

**Date:** 2026-04-29
**Status:** ~~Code prepared on branch `feat/nope-ablation`. Launch held pending BlueVela load reduction.~~
**Superseded by:** `docs/logbook/2026-04-29-nope-ablation-launched.md` — BlueVela had free slots the same night, so we launched on 2026-04-29 at 01:04 UTC instead of waiting for 2026-05-01. Jobs: 75472 (partial), 75479 (baseline), 75480 (scoped) on preemptable, 4 GPUs each.

---

## Summary

Ablation study evaluating whether NoPE (no positional embedding) is feasible
and beneficial for OpenMythos's recurrent-depth architecture. Three variants:

- **Baseline** (full RoPE) — `mythos_1b()`
- **Scoped NoPE** — RoPE in prelude/coda, NoPE inside the recurrent block; `mythos_1b_scoped_nope()`
- **Partial NoPE (MLA)** — `qk_rope_head_dim=0`, budget reassigned to `qk_nope_head_dim`; `mythos_1b_partial_nope()`

Each variant: 1 B tokens, 4 × H100 single node, preemptable queue, ~20 h wall
clock. Three jobs in parallel on separate nodes → ~20 h total, ~240 GPU-hours
compute.

Design rationale and pre-registered decision thresholds:
`docs/superpowers/specs/2026-04-29-nope-for-recurrent-depth-design.md`.
Implementation plan:
`docs/superpowers/plans/2026-04-29-nope-ablation.md`.

## Why this is queued, not running

BlueVela is currently overloaded: job 71939 (the ongoing 10 B run) occupies
one preemptable slot, and the general preemptable queue depth is high. Do
not submit until:

1. `bjobs -u pzerfos` shows no conflicting pzerfos jobs pending from earlier
   decisions, AND
2. A spot check of overall queue utilization on BlueVela indicates submitting
   three 4-GPU jobs won't preempt other users' critical work.

Earliest reasonable launch: **on or after 2026-05-01**.

## Launch procedure (when load permits)

```bash
# On BlueVela, from a dedicated worktree so the main checkout stays on main:
ssh pzerfos@login4.bluevela.rmf.ibm.com
cd /u/pzerfos/OpenMythos
git fetch origin feat/nope-ablation
git worktree add /u/pzerfos/OpenMythos-nope feat/nope-ablation
cd /u/pzerfos/OpenMythos-nope

# Submit all three variants as independent 4-GPU jobs:
bash deploy/bluevela/bsub_nope_ablation.sh all
```

Expected: three PEND jobs named `pz-mythos-nope-{baseline,scoped,partial}`.

The variant is selected via a `--variant` CLI flag on the training script
(default "baseline"), plumbed through `deploy/bluevela/run_nope_ablation.sh`
as a positional arg. No new environment variables are introduced.

## Post-training eval

```bash
# Run per-variant on the final checkpoint:
python evaluations/eval_length_gen.py \
    --checkpoint /proj/checkpoints/pzerfos/openmythos/nope-ablation/scoped/step_NNNNN.pt \
    --variant scoped
python evaluations/eval_depth_gen.py \
    --checkpoint /proj/checkpoints/pzerfos/openmythos/nope-ablation/scoped/step_NNNNN.pt \
    --variant scoped
```

Grep `RESULT` in the log output for the tabulated NLL / PPL at each (seq_len, n_loops) point.

## Decision matrix reference

See spec §7. Summary:

- **Scoped wins** on all three axes → migrate production to Scoped NoPE.
- **MLA partial wins** on parity + comparable extrapolation → adopt the one-line `qk_rope_head_dim=0` config flip.
- **Mixed or regression** → keep full RoPE as default; publish as ablation-negative.
