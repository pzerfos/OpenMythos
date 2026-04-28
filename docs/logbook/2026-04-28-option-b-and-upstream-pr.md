# Option B Implementation, ClearML Fix, and Upstream PR Update

**Date:** 2026-04-28
**Goal:** Close out the 10B scale-up dependencies (ClearML connectivity, artifact-upload bug), design and implement the stochastic-depth training recipe (Option B), and push accumulated improvements upstream.

---

## 1. ClearML Reachable from Compute Nodes

The ClearML firewall to `clearml-ext.sl.res.ibm.com:8008` was opened for BlueVela compute nodes on 2026-04-27. Confirmed via job 56069 — `Task.init()` succeeded in 9.4s from a compute node. Training metrics for the 10B run are now live on the ClearML dashboard at project `granite-mythos / 1b-10b-tokens`.

Earlier defensive SIGALRM timeout in `init_clearml()` is retained as a safety net — it no longer fires but is kept in case the network goes down mid-run.

## 2. 10B Scale-Up Shakedown

Three jobs were needed to get the 10B run stable:

| Job | Outcome | Issue |
|-----|---------|-------|
| 56133 | EXIT 10s | 2-node `blaunch` failure: `info:: line 5: p4-r23-n3: command not found` (blaunch multi-node env-var prefix syntax breaks on the second host) |
| 56153 | Killed step ~32,400 | ClearML 18GB checkpoint artifact upload blocked the background scalar-reporting thread for ~10 min per ckpt; second ClearML task spuriously created during upload stall |
| **56429** | **Running** | Healthy — 8 GPUs on a single node, no multi-node complexity, no artifact uploads |

**Fixes applied:**
- Dropped multi-node `blaunch` for now — using **8 GPUs on a single node** (`deploy/bluevela/bsub_1b_10b.sh`). `grad_accum` auto-adjusts to 2; global batch stays at 32,768 tokens/step; step numbering continuous with the prior 4-GPU checkpoint.
- Removed `register_clearml_artifact()` from `save_checkpoint()` (commit `20c753a`). Checkpoints live on the shared filesystem at `/proj/checkpoints/pzerfos/openmythos/checkpoints`; no reason to duplicate to the ClearML file server (which has limited storage anyway).

**Status at ~04:22 UTC 2026-04-28:** step 54,952 / 305,175 (18.0%), loss ~3.3, gnorm ~0.7, 1.8B tokens seen. ~7.75 hours uninterrupted on `p6-r26-n3`.

## 3. Stochastic-Depth Training (Option B) — Designed, Implemented, Merged

The 1B PoC evaluation (2026-04-24) confirmed the upstream ablation finding that ACT binds the model to its trained depth. Option B — disable ACT + sample `n_loops` uniformly per step — is the only recipe shown to produce monotonic PPL-vs-depth curves.

### Design (spec: `docs/superpowers/specs/2026-04-27-stochastic-depth-training-design.md`)

- Runtime toggle via local variables in the training script: `recurrent_mode` (`"act"` or `"stochastic_depth"`), `stochastic_depth_min = 1`, `stochastic_depth_max = 32`.
- New `bypass_act: bool = False` parameter plumbed through `OpenMythos.forward` → `RecurrentBlock.forward`. When `True`, skip the ACT weighted-sum accumulation and halting-driven early exit; return the final hidden state directly.
- `ACTHalting` module stays in the architecture (no state_dict shape change) — its weights simply receive no gradient when bypassed.
- **Checkpoints are compatible across modes.** The 1B ACT-trained checkpoint can resume under stochastic_depth and vice versa.
- FSDP safety: `n_loops` sampled once per optimizer step on rank 0 and broadcast to all ranks, preventing collective-ordering deadlock (same bug class as the ACT early-exit deadlock fixed in `6c5659c`).

### Implementation (6 tasks, 7 commits, subagent-driven)

Executed as a feature branch `feat/stochastic-depth-training` following the subagent-driven workflow: per-task implementer + spec-compliance reviewer + code-quality reviewer, then a final holistic reviewer.

Commits on the branch (merge commit `87d5fa6`):
- `fcf00d9` feat(model): add bypass_act flag to RecurrentBlock.forward
- `ceab6b2` feat(model): plumb bypass_act through OpenMythos.forward
- `c03eba7` + `58a5ded` test: state_dict cross-mode round-trip
- `1a82955` + `4db500f` feat(training): stochastic-depth mode + rank 0 broadcast fix
- `4714f68` test: smoke test one training step in each mode

Notable reviewer catches during implementation:
- **Restored the FSDP deadlock rationale comment** that got dropped in the first pass — that comment is load-bearing (without it a future reader might "optimize away" the unconditional all-reduce).
- **Caught the per-rank RNG desync bug** before landing: first pass called `random.randint` independently on each rank, which would have deadlocked multi-rank FSDP training immediately. Fixed by broadcasting from rank 0.
- **Moved sampling outside the micro-step loop** so all grad_accum micro-steps within one optimizer step train at the same depth and the ClearML scalar reports the actual depth used.

### Tests

6 new tests in `tests/test_stochastic_depth.py`, all passing:
- bypass_act=True produces a different output from bypass_act=False
- bypass_act=True runs all `n_loops` iterations (no early exit)
- bypass_act=True output matches a manual unrolled iteration returning final `h`
- `OpenMythos.forward(bypass_act=True)` propagates correctly
- state_dict round-trip across modes (strict load + finiteness)
- one forward+backward+optimizer step works in each mode

Full suite: 311 passing, 14 pre-existing failures unchanged.

### Merge to main

Merged via PR #7 on IBM fork (merge commit `87d5fa6`), mirrored to public fork. **Default is `recurrent_mode="stochastic_depth"`** — the next time the 10B job preempts and resubmits, it will switch modes. Expect a transient loss spike of ~0.3–0.5 for a few hundred steps while the Coda re-adapts. Set `recurrent_mode = "act"` on the BlueVela checkout before relaunch to keep the current run on ACT.

## 4. Upstream PR #56 Updated

PR #56 to `kyegomez/OpenMythos` previously contained only the FSDP dtype fix (1 commit). Updated with 4 logically-grouped commits covering everything we've built that is generally useful:

1. `fix(model): FSDP mixed precision dtype compatibility` (original, preserved)
2. `feat(model): MoE grouped dispatch + ACT FSDP deadlock fix + code review fixes + tests`
3. `feat: stochastic-depth training (Option B)`
4. `feat(training): add 1B FineWeb-Edu training script with stochastic-depth toggle`

All BlueVela/IBM-specific content was excluded: `deploy/bluevela/*`, the BlueVela-specific logbooks, the original `training/1b_poc_fineweb.py` (replaced by a cleaned-up `training/1b_fine_web_edu.py`), `evaluations/eval_checkpoint.py` (has IBM paths — can be generalized later), and internal architecture comparison docs.

Generic cleanups applied to the upstream version of the training script:
- `OUTPUT_DIR` default → `./output/experiments`
- `DATASET_PATH` default → `./data/fineweb-edu`
- `CLEARML_PROJECT` default → `openmythos`
- `EXPERIMENT_NAME` default → `1b-fine-web-edu`
- Docstring rewritten to remove BlueVela references

311 tests pass on the PR branch. Force-pushed to `public/fix/fsdp-mixed-precision-dtype`; PR title and body updated to reflect expanded scope.

**PR:** https://github.com/kyegomez/OpenMythos/pull/56

---

## Current Status Snapshot

- **10B training (job 56429):** running, step ~54,952 / 305,175 (~18% complete), loss 3.3, ~1.1s/step. ETA ~77h if uninterrupted. Currently in ACT mode (the BlueVela checkout predates the stochastic-depth merge).
- **Code:** main fork has full stochastic-depth feature merged. Upstream PR #56 updated.
- **ClearML:** live at `granite-mythos / 1b-10b-tokens`, no more artifact-upload blocking.

---

## Remaining Open Items

All ACT-architecture uncertainty is resolved (Option B chosen and built). Remaining items from prior logbooks that are still open:

1. **lm-eval-harness integration** — benchmark comparison (HellaSwag, ARC, MMLU) with published models of similar size. Not yet started.
2. **Fix 14 pre-existing test failures** — 13 RoPE dimension mismatches after upstream flash-attn merge + 1 LTI spectral-radius strict-inequality boundary. Deferred; do not block current training.
3. **`router_bias` load balancing** (pzerfos/OpenMythos#3) — bias is initialized but never updated during training. Expert utilization imbalance may grow with longer training. Deferred.
4. **FSDP1 → FSDP2 migration** (pzerfos/OpenMythos#6) — lower memory, `torch.compile` support. Worth doing before a larger-scale run.
5. **Study Gated DeltaNet** — Qwen3.6's hybrid linear+full attention for long-context efficiency. Future architecture direction.
6. **Decide whether to switch the 10B run mid-training to Option B** — the next preemption will do this automatically unless `recurrent_mode = "act"` is set on the BlueVela checkout. Open question: do we want to continue the ACT 10B run to completion, or switch to Option B now to get a depth-extrapolation-capable checkpoint?

---

## Closed Items (from prior logbooks)

- ~~ACT architecture decision~~ — resolved: implemented Option B (stochastic depth) as a switchable recipe, with Option A (ACT) preserved as default-off alternative.
- ~~ClearML connectivity from compute nodes~~ — resolved 2026-04-27 (firewall opened).
- ~~Multi-node FSDP via `blaunch`~~ — deferred; 8 GPUs on a single node is sufficient for 1B PoC.
- ~~Code review items from `docs/code-review-2026-04-23.md`~~ — all actionable items landed in commit `65cd807` and follow-ups; the ACT deadlock item is resolved in `6c5659c`.
- ~~ClearML checkpoint artifact uploads blocking scalar reporting~~ — resolved `20c753a`.
