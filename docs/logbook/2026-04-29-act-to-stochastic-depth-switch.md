# Mid-Training Switch: ACT → stochastic_depth at step 121,000

**Date:** 2026-04-29
**Goal:** Switch the in-flight 10B-token run from ACT mode to stochastic_depth (Option B) so the final checkpoint gains depth-extrapolation capability.

---

## Transition Point

| Item | Value |
|------|-------|
| Transition step | **121,000** (out of 305,175) |
| Tokens seen at switch | ~3.97B |
| Last ACT-mode checkpoint | `step_0121000.pt` |
| Preserved copy | `/proj/checkpoints/pzerfos/openmythos/preserved/step_0121000.act_final.pt` |
| ACT-mode job | `56429` (killed after checkpoint save at 01:10 UTC 2026-04-29) |
| stochastic_depth job | `67208` (submitted 01:31 UTC, resumed at 01:32:12 from `step_0121000.pt`) |
| BlueVela repo HEAD at resubmit | `caf87ad` (includes PR #7 stochastic-depth merge `87d5fa6`) |
| Resume loss (ACT, step 120,751) | ~3.1 |
| First stochastic_depth steps (121001–121021) | loss 3.1–4.8 depending on sampled n_loops |

## Procedure Used

1. Waited for scheduled checkpoint at step 121,000 (cadence: every 1,000 steps ≈ 19 min).
2. `bkill 56429` immediately after `step_0121000.pt` was written.
3. `git pull` on BlueVela (`/u/pzerfos/OpenMythos`) to pick up the stochastic_depth feature. Default `recurrent_mode` is now `stochastic_depth`; `n_loops` sampled uniformly from `[1, 32]` per optimizer step (broadcast from rank 0 to prevent FSDP collective-ordering deadlock).
4. Copied `step_0121000.pt` (18.5 GB) to `preserved/step_0121000.act_final.pt` so the `keep_last=3` rotation can't delete it.
5. Resubmitted via `bash deploy/bluevela/bsub_1b_10b.sh` → job 67208 (8 GPUs, `p6-r02-n1`, preemptable).

## First-20-Step Behavior After Switch

Loss correlates directly with the sampled `n_loops`:
- `n_loops ≥ 20`: loss 3.1–3.9 (matches pre-switch baseline)
- `n_loops ≈ 14–16`: loss 3.5–3.7
- `n_loops ≤ 10`: loss 4.1–4.8 (not yet adapted to shallow depths)

`gnorm` spiked to 10.4 on step 121,002 (first full-size backward through a shallow-depth sample at `n_loops=11`), then settled to 1.0–4.1 by step 121,015. No NaN/Inf, no crashes. Mode switch is clean — the deep-depth performance was preserved, and the shallow-depth gap is exactly what stochastic depth is meant to close over the remaining ~184k steps.

## Why Switch Now vs. Finish ACT First

The 1B-token evaluation on 2026-04-24 confirmed the upstream finding (kyegomez/OpenMythos#28): ACT binds the final model to its trained depth. An ACT-trained 10B checkpoint would show the same flat-beyond-16, cliff-below-16 curve we observed at 1B. Switching mid-run keeps the ~3.97B tokens of ACT pre-training (which produced a usable LM) and spends the remaining ~6B tokens teaching the model to produce valid hidden states at every depth in `[1, 32]`.

## Notes

- ClearML task `c420fc0ca4324591b0904ebee2b81797` (project `granite-mythos / 1b-10b-tokens`) continues with the new job — scalar series will show a visible discontinuity at step 121,000.
- The `ACTHalting` module is still instantiated inside `RecurrentBlock`; under `bypass_act=True` its weights simply receive no gradient, so the state_dict shape is unchanged and checkpoints are cross-mode compatible in both directions.
- If we ever want to recover the ACT-mode behavior from this run, load `preserved/step_0121000.act_final.pt` with `recurrent_mode="act"`.
