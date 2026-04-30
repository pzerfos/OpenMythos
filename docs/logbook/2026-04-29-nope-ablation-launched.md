# NoPE Ablation — Launched 2026-04-29 (Earlier Than Queued 2026-05-01)

**Date:** 2026-04-29
**Status:** 3 variants running on BlueVela. Expected completion ~21:00–21:30 UTC on 2026-04-30.
**Prior logbook:** `docs/logbook/2026-04-29-nope-ablation-queued.md` (launch was held for 2026-05-01 there; BlueVela had space tonight, so we launched early).

---

## Summary

The 3-variant NoPE ablation was submitted on 2026-04-29 at ~01:04 UTC, roughly 36 hours ahead of the queued earliest-launch date. BlueVela preemptable queue had compute available (724 RUN + 74 PEND at submission time), so waiting was unnecessary.

First submission round hit two independent bugs and was resubmitted after quick fixes. All three variants are now training cleanly.

## Running jobs (final state after fixes)

| Job | Variant | Start (UTC) | Node | Ckpt dir | ClearML task |
|-----|---------|-------------|------|----------|--------------|
| 75472 | **partial** (MLA `qk_rope_head_dim=0`) | 2026-04-30 01:07 | p6-r16-n4 | `/proj/checkpoints/pzerfos/openmythos/checkpoints/nope-ablation/partial/` | `granite-mythos / nope-ablation/partial` |
| 75479 | **baseline** (full RoPE) | 2026-04-30 01:11 → 01:20 RUN | p3-r28-n4 | `.../nope-ablation/baseline/` | `granite-mythos / nope-ablation/baseline` |
| 75480 | **scoped** (NoPE inside recurrent block) | 2026-04-30 01:11 → 01:20 RUN | p6-r27-n2 | `.../nope-ablation/scoped/` | `granite-mythos / nope-ablation/scoped` |

Each run is 4 × H100, single node, preemptable queue. Target 1 B tokens (~30,517 optimizer steps at global batch 32,768). Expected wall clock ~20 h per variant → earliest completion around 21:00–21:30 UTC on 2026-04-30.

The in-flight **10 B production run (job 71939)** continues unaffected on p5-r08-n1. As of the launch time it was at ~step 185,379/305,175 (6.1 B tokens seen, loss 3.08, gnorm 0.96, stochastic_depth + router_bias @ 1e-3).

## First-submission bugs and fixes (commit `897c9e5`)

### Bug 1 — baseline loaded the production 10B checkpoint instead of training from scratch

**Symptom:** `--variant baseline` run exited with "Training complete" at step 184,000 seconds after submission, having loaded an 18 GB checkpoint from the 10B production run's directory. No ablation training actually occurred.

**Root cause:** the training script's ckpt_dir logic gated the isolation subdir on `variant != "baseline"`, so baseline fell through to the default `checkpoints/` path — which is exactly where the 10B production run's checkpoints live. Auto-resume then picked up `step_0184000.pt`.

**Fix:** introduce `ablation_mode = cli_args.variant is not None`. The presence of the `--variant` CLI flag (not the value) is what signals "submitted by `bsub_nope_ablation.sh`". All three variants, including `baseline`, now get `checkpoints/nope-ablation/<variant>/`. Production submissions via `bsub_1b_10b.sh` (which omit `--variant`) continue to use `checkpoints/` unchanged.

### Bug 2 — torchrun default port 29500 collision (EADDRINUSE)

**Symptom:** scoped-variant job died at `torchrun` startup with `torch.distributed.DistNetworkError: address already in use, port 29500`.

**Root cause:** torchrun's default master-port is 29500. The scoped job landed on a node where a previous torchrun had just exited; the socket was still in TIME_WAIT. Could also happen under two jobs from different users landing close in time.

**Fix:** derive `--master_port` in `deploy/bluevela/run_nope_ablation.sh` as `29500 + (LSB_JOBID % 1000)`, giving each LSF job a distinct port. Falls back to `$$` (PID) when `LSB_JOBID` is unset (local sanity checks).

### Separate, non-bug issue — editable-install path

`open_mythos` is installed in editable mode in BlueVela's conda env, pointing at `/u/pzerfos/OpenMythos` (the main checkout). The dedicated NoPE worktree at `/u/pzerfos/OpenMythos-nope` was ignored by Python's import resolver — imports from `open_mythos.variants` resolved to the old-main checkout at commit `5570b5f`, which predates the NoPE merge and raised `ImportError: cannot import name 'mythos_1b_partial_nope'`.

**Resolution:** `git pull` on `/u/pzerfos/OpenMythos` to bring main up to `4a25308` (the merged + post-merge state). This is backward-compatible with the 10B production run's auto-resume path: the new default `variant = "baseline"` (without the CLI flag) uses `mythos_1b()`, identical to the old default; all `pe_mode_*` default to `"rope"`; `state_dict` shapes unchanged.

**Lesson:** the "separate worktree per experiment" pattern used for the FSDP2 bench (`/u/pzerfos/OpenMythos-bench`) does not isolate Python imports when the package is installed editably from a single canonical checkout. For Python-package experiments, either (a) pull main on the canonical checkout when the new code is backward-compatible, or (b) make the worktree its own conda env / venv. Option (a) worked here because the NoPE changes are purely additive.

## First-40-step sanity check (each variant)

All three variants log `mode=stochastic_depth n_loops=<varies>`, freshly-initialized loss in the ~12.3–12.5 range (close to `log(vocab_size) = log(32000) ≈ 10.37`; the gap reflects early-training noise before the loss settles toward the entropy floor). gnorm 4–12 range, no NaN/Inf. Same per-step `n_loops` sequence across variants (seeded identically, broadcast from rank 0).

Example step-42 lines:
- partial: `step 42/30517 | loss 12.11 | gnorm 9.56 | n_loops=22`
- baseline (at step 53 when partial was at 42): `loss 11.20 | gnorm 6.24 | n_loops=2`
- scoped (at step 51): `loss 11.44 | gnorm 11.09 | n_loops=32`

Wall-clock offset: partial started ~13 min earlier than baseline/scoped due to the resubmission cycle, so it remains a few steps behind the other two despite the earlier start (baseline/scoped had a cleaner startup path without the bug-retry overhead, and n_loops distribution can shift step timing).

## Follow-up when training completes

1. Run `evaluations/eval_length_gen.py --variant <v>` on each variant's final checkpoint — seq_len sweep {2048, 4096, 8192, 16384}.
2. Run `evaluations/eval_depth_gen.py --variant <v>` — n_loops sweep {16, 32, 48, 64, 96}.
3. Tabulate training-loss parity via ClearML scalars (EMA of last ~1,000 steps per variant).
4. Apply the pre-registered decision matrix from the spec §7 — Green / Partial-win / Mixed / Red.
5. Write up results in `docs/logbook/2026-04-30-nope-ablation-results.md` mirroring the FSDP2 feasibility-results format.

## Changes to the merged PR

The fixes in commit `897c9e5` were pushed to `main` after PR #10 was already merged. They are NOT in the merged PR diff but they ARE on the branch-that-was-PR-#10's tip at merge-time — meaning the merged code had the two bugs. The `897c9e5` fix commit is a standalone post-merge fix on main. If we ever cherry-pick or re-merge this work elsewhere, include `897c9e5` alongside the PR #10 commits.
