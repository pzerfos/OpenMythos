# Mid-Training Switch: enable `router_bias_update_rate=1e-3` at step 156,000

**Date:** 2026-04-29
**Goal:** Activate DeepSeek-V3 aux-loss-free MoE load balancing on the in-flight 10B-token run so the final checkpoint benefits from a balanced expert allocation, without starting a fresh run.

---

## Training timeline so far

| Steps | Regime | Job | Notes |
|-------|--------|-----|-------|
| 0 – 121,000 | ACT | 56429 | Original recipe; binds model to trained depth |
| 121,001 – 156,000 | stochastic_depth, `router_bias_update_rate=0.0` | 67208 | Option B switch (see `2026-04-29-act-to-stochastic-depth-switch.md`) |
| 156,001 – | stochastic_depth + `router_bias_update_rate=1e-3` | **71939** | DeepSeek-V3 Algorithm 1 now live |

## Transition Point

| Item | Value |
|------|-------|
| Transition step | **156,000** (out of 305,175) |
| Tokens seen at switch | ~5.1B |
| Last pre-update checkpoint | `step_0156000.pt` |
| Preserved copy | `/proj/checkpoints/pzerfos/openmythos/preserved/step_0156000.stochastic_depth-pre_router_bias.pt` (18.5 GB) |
| Prior job | `67208` (killed 13:24 UTC 2026-04-29 after step_0156000.pt was written) |
| New job | `71939` (submitted 13:25 UTC, resumed at 13:26:09 from `step_0156000.pt`) |
| BlueVela repo HEAD at resubmit | `5570b5f` (`caf87ad..5570b5f` fast-forward — picks up PR #8/#9 router-bias feature plus the rate flip) |

## Checkpoint compatibility check (done before the switch)

The `step_0156000.pt` state_dict was probed before killing the job to confirm the resume would load cleanly under the new code:

- `register_buffer("router_bias", torch.zeros(cfg.n_experts))` has existed in `MoEFFN.__init__` since commit `7916266`, long before PR #8/#9. So the old checkpoint already contained `recurrent.block.ffn.router_bias` as a zero-initialized buffer. PR #8/#9 added the *update logic* (`update_router_biases`, training-loop call, ClearML scalars) and a non-persistent `expert_counts` buffer — **no new persistent state**.
- Diffed `register_buffer` / `nn.Parameter` calls between BlueVela (`caf87ad`) and the local post-PR-#9 tree: identical persistent key set.
- Conclusion: `load_state_dict(..., strict=True)` succeeds as-is, no shim needed. Confirmed by the resume log line `SUCCESS | __main__:main:560 - Resumed at step 156000`.

## Procedure Used

1. Set a file-existence monitor polling BlueVela every 60 s for `step_0156000.pt`.
2. When the checkpoint appeared at 13:23:51 UTC (written atomically — `save_checkpoint` writes to `.tmp` then `os.replace`):
3. `cp` the checkpoint to `preserved/step_0156000.stochastic_depth-pre_router_bias.pt` so the `keep_last=3` rotation cannot delete it.
4. `bkill 67208` → EXIT.
5. Local edit: `training/1b_poc_fineweb.py:424` `router_bias_update_rate = 0.0` → `1e-3`. Commit `5570b5f`, push `origin/main`.
6. On BlueVela: `git pull --ff-only` (fast-forward `caf87ad..5570b5f`), then `bash deploy/bluevela/bsub_1b_10b.sh` → job **71939**.
7. Second monitor confirmed 71939 → RUN at 13:25:42 UTC and clean resume at 13:26:09 UTC.

Total downtime: ~2 minutes between kill and resume start.

## Why enable now vs. wait for a fresh run

The 2026-04-28 roadmap (`2026-04-28-option-b-and-upstream-pr.md`) had router_bias flagged to enable "at the next scale-up from a balanced starting point". Enabling mid-run instead, because:

- Starting a fresh 10B-token run purely to enable router_bias would discard ~5.1B tokens of useful training. The cost of discarding is strictly larger than the cost of a transient loss bump as `router_bias` drifts from zero toward a balanced configuration.
- `router_bias` starts at exactly zero in the preserved checkpoint, so the first update steps are operating from the paper's stated starting point.
- With DeepSeek-V3's `rate=1e-3` the update is small per step; expect `|router_bias|` to grow by at most `rate × n_steps` in the worst case, i.e. O(1) over the remaining ~149k steps. Safe margin against saturating the gating logits.
- ~6B tokens remain — enough window for the balance to converge and for the Coda to adapt.

## What to watch in ClearML

New scalars (only emitted when `router_bias_update_rate > 0.0`, which is now true):

- `router_imbalance_max_over_mean` — max expert count ÷ mean expert count. DeepSeek-V3 target: approach 1.0.
- `router_imbalance_stddev_over_mean` — coefficient of variation of per-expert load. Lower is more balanced.
- `router_imbalance_ratio` — alternate formulation (max/min or similar, per the implementation).

Expected trajectory: spike at step ~156,005 (first window of counts collected under a still-zero bias), then steady decline over the next few thousand steps.

Loss behavior: a small transient bump (~0.1–0.3) is possible as the Coda adapts to the shifted expert-selection distribution. If the bump exceeds ~0.5 or `gnorm` spikes above 10 for more than a few consecutive steps, fall back by:

1. `bkill 71939`
2. Edit `router_bias_update_rate` back to `0.0`, commit, push, pull, resubmit.
3. Loading will pick up whatever `router_bias` values are in the latest checkpoint — setting rate=0 freezes them rather than resetting; that's fine for a fallback. If a harder reset is needed, resume from `preserved/step_0156000.stochastic_depth-pre_router_bias.pt`.

## Notes

- ClearML task `c420fc0ca4324591b0904ebee2b81797` continues — expect the router_imbalance series to start mid-run; loss/gnorm series continue without reset.
- `step_0121000.act_final.pt` (preserved 2026-04-29) and `step_0156000.stochastic_depth-pre_router_bias.pt` (preserved today) are the two "regime-change anchor" checkpoints. Useful if we ever want to ablate the effect of either switch in isolation.
- Closes the "router_bias enablement" open item from the 2026-04-28 roadmap ahead of the originally planned "next scale-up" trigger.
