# NoPE (No Positional Embeddings) for Recurrent-Depth — Design Spec

**Date:** 2026-04-29
**Status:** Spec — prepared for launch in ~2 days (BlueVela currently overloaded; do not submit immediately).
**Author:** brainstorming session.
**Context:** Explore whether replacing RoPE with NoPE is (a) feasible at OpenMythos's 1B scale, (b) simplifies the design, and (c) trains as well or better — with specific attention to the recurrent-depth looped structure, which is unstudied in the NoPE literature to date.

---

## 1. Motivation & literature grounding

Rotary Positional Embeddings (RoPE) have been the default positional encoding in OpenMythos, inherited from the upstream recurrent-depth line (Geiping's Huginn, Kyegomez's OpenMythos). They introduce two practical issues:

1. **Extrapolation beyond the pre-computed `max_seq_len=4096`** requires regenerating `freqs_cis`; beyond that range, RoPE is the known cause of length-generalization degradation (see Kazemnejad et al., "The Impact of Positional Encoding on Length Generalization in Transformers", NeurIPS 2023 — arxiv 2305.19466).
2. **Complex-valued `freqs_cis` / `freqs_cis_mla` buffers** caused 36 `torch.compile` recompilations and the `"Torchinductor does not support code generation for complex operators"` fallback in the FSDP2 feasibility benchmark (see `docs/logbook/2026-04-29-fsdp2-feasibility-results.md`).

NoPE — removing positional encoding entirely and relying on the causal mask to provide position — has been shown by Kazemnejad to match or outperform RoPE on length generalization in small (~250M) decoder-only models. Subsequent work ("Rope to NoPE and Back Again", Yang et al. 2025, arxiv 2501.18795; the HuggingFace "Smol Training Playbook", Oct 2025) has explored hybrid strategies at larger scale.

**The gap this study closes:** no published work evaluates NoPE inside a *recurrent-depth* transformer, where the same attention layer is re-applied T times per forward pass. "Mechanistic Dynamics of Looped Transformers" (2026) explicitly calls out this gap: *"Role of positional encoding across recurrences: No analysis of how RoPE [behaves across recurrences]"*.

## 2. Scope & non-goals

### In scope

- Add a per-site `pe_mode` axis to `MythosConfig` so we can selectively disable RoPE in the prelude, coda, or recurrent block.
- Two novel variants in addition to the full-RoPE baseline:
  - **Scoped NoPE:** RoPE in prelude + coda; NoPE in the recurrent block.
  - **Partial NoPE (MLA):** set `qk_rope_head_dim=0`, transfer the budget to `qk_nope_head_dim`. Config-only, no code changes.
- Three 1 B-token training runs and three post-training generalization sweeps.

### Out of scope

- **Hybrid RNoPE** (per-layer alternation, Yang 2025). Rejected during brainstorming because it has no clean analog for a single-block recurrent architecture.
- **Pure NoPE** (drop RoPE everywhere). Rejected because the potential payoff is not enough larger than Scoped NoPE to justify the higher risk at 1B scale.
- **RoPE → real-valued cos/sin refactor.** Would unblock `torch.compile` without changing semantics, but orthogonal to this study. Potential follow-up if baseline RoPE is chosen.
- **Bidirectional attention / MLM pretraining.** NoPE fundamentally requires causal masking to provide its positional signal (Kazemnejad §4); extending this study to encoder-style models would not share conclusions. Hard boundary.
- **End-to-end `torch.compile` unblock test.** Interesting but was explicitly deprioritized during brainstorming; can be added after the feasibility result.

## 3. Invariants preserved

1. **Causality is preserved** in all three variants. The causal mask is applied in the attention kernel regardless of RoPE presence; NoPE depends on this mask for its implicit position signal (Kazemnejad §4).
2. **Checkpoint cross-mode loading** with `strict=True`. `freqs_cis` and `freqs_cis_mla` buffers remain registered on `OpenMythos` even under fully-NoPE configurations, so state_dict keys are identical across variants. Under NoPE the buffers exist but are never read.
3. **KV cache semantics** — under RoPE the cached K is post-rotation; under NoPE the cached K is raw. Either is correct so long as prefill and decode use the same mode. A unit test enforces `prefill_then_decode == one_shot_prefill` for each mode.
4. **FSDP collective-ordering invariant** (see `docs/logbook/2026-04-23-act-fsdp-deadlock.md`) — no rank-local early exits introduced. No new `all_reduce` calls introduced.
5. **Loop-index sinusoidal embeddings** inside `RecurrentBlock` are depth encodings, orthogonal to sequence-position encoding. They remain in all variants.

## 4. Architecture changes

### New `MythosConfig` fields

```python
pe_mode_prelude:   Literal["rope", "nope"] = "rope"
pe_mode_coda:      Literal["rope", "nope"] = "rope"
pe_mode_recurrent: Literal["rope", "nope"] = "rope"
```

All default to `"rope"`, so existing configurations and checkpoints are unchanged.

### Attention plumbing

- `MLAttention.forward` and `GQAttention.forward` gain a `pe_mode: str` parameter.
  - `pe_mode == "rope"` — existing behavior.
  - `pe_mode == "nope"` — skip `apply_rotary_emb`. All head dims treated as content. For MLA, the (already-present) content/position split has the position slice run through the same Q/K linear without rotation.
- `TransformerBlock.forward` forwards `pe_mode` to its inner attention.
- `RecurrentBlock.__init__` stores `cfg.pe_mode_recurrent`; its forward passes that to the inner `TransformerBlock`.
- `OpenMythos.__init__` passes `cfg.pe_mode_prelude` into prelude `TransformerBlock` constructors, `cfg.pe_mode_coda` into coda constructors.

### Partial NoPE via MLA config

Exclusively a config change: `qk_rope_head_dim=0`, `qk_nope_head_dim=96`. No new code path; MLA's existing content/position head split handles a zero-sized rope slice correctly. A single smoke-test forward validates this.

### Variant config helpers in `open_mythos/variants.py`

```python
def mythos_1b_scoped_nope()  -> MythosConfig:  # base + pe_mode_recurrent="nope"
def mythos_1b_partial_nope() -> MythosConfig:  # base + qk_rope_head_dim=0, qk_nope_head_dim=96
```

### Unit tests (`tests/test_nope.py`)

- RoPE-vs-NoPE forward equivalence at position 0 (RoPE identity).
- State_dict round-trip across all three variants (strict=True), confirming checkpoint compatibility.
- KV-cache correctness under each variant: `prefill(seq) == prefill(seq[:k]) then decode(seq[k:])`.
- Partial-NoPE MLA forward smoke test (`qk_rope_head_dim=0`, ensure shapes and finiteness).

All tests CPU-only, small dims, following existing `tests/` conventions.

## 5. Training setup

### Compute profile

- Per experiment: **4 × H100, single node, preemptable queue**.
- Three jobs submitted simultaneously on separate nodes (no shared state, no cross-run comms).
- `grad_accum` auto-adjusts to 8 (from the 8-GPU baseline's 4) to keep the global batch at 32,768 tokens/step.

### Shared across all three variants

- Dataset: FineWeb-Edu 100B sample at `/proj/datasets/pzerfos/fineweb-edu-100B/sample/100BT`. Same shard ordering.
- Base config: `mythos_1b()` — dim=2048, 64 experts, max_loop_iters=16, MLA.
- `recurrent_mode = "stochastic_depth"`, `n_loops ∈ [1, 32]`.
- `router_bias_update_rate = 1e-3`.
- Optimizer, LR schedule, warmup, weight decay: unchanged from `training/1b_poc_fineweb.py`.
- Seed for data shuffling and stochastic `n_loops` sampling: fixed and identical across variants. This ensures the three runs see the same data in the same order and the same `n_loops` sequence.

### Per-variant differences (the only differences)

| Variant | `pe_mode_{prelude, coda, recurrent}` | `qk_rope_head_dim` | `qk_nope_head_dim` |
|---|---|---|---|
| **Baseline (RoPE)**      | rope / rope / rope | 32 | 64 |
| **Scoped NoPE**          | rope / rope / nope | 32 | 64 |
| **Partial NoPE (MLA)**   | rope / rope / rope | 0  | 96 |

### Token budget

**1 B tokens per variant** (≈ 30,500 optimizer steps at global batch 32,768). Expected wall clock at 4 GPUs: ~20 h per variant (step time ~2.4 s, roughly 2× the production 8-GPU step time since `grad_accum` doubles). Three jobs in parallel on separate nodes → ~20 h total wall clock, ~240 GPU-hours total compute (4 GPUs × 20 h × 3 variants).

### Logging

- ClearML project: `granite-mythos`. Task names: `nope-ablation/baseline`, `nope-ablation/scoped`, `nope-ablation/partial-mla`. Separate tasks so loss curves overlay in the UI.
- Checkpoint dirs: `/proj/checkpoints/pzerfos/openmythos/nope-ablation/{baseline,scoped,partial-mla}/`.
- `keep_last=3` applies per variant.

### Submission

- `training/1b_poc_fineweb.py` carries a `variant` local (default `"baseline"`) near the other run-knob locals (`recurrent_mode`, `router_bias_update_rate`, etc.) and accepts a `--variant {baseline,scoped,partial}` CLI flag that overrides the local. The variant picks the config builder from `variants.py` and overrides the ClearML task name + checkpoint dir. No new environment variables are introduced.
- `deploy/bluevela/bsub_nope_ablation.sh` submits all three variants as separate bsub jobs. Each 4 GPUs, one node, preemptable.

### Launch timing

**Do not submit on 2026-04-29 or 2026-04-30.** BlueVela is overloaded; existing job 71939 (10 B training) occupies one preemptable slot and must not be displaced. Target launch: **on or after 2026-05-01**, pending a visual check of `bjobs -u pzerfos` and the general preemptable queue depth. When the 10 B run completes, that also frees a slot.

## 6. Evaluation protocol

All three evaluations run post-training on the final checkpoint of each variant. Held-out data: 1 k sequences from a FineWeb-Edu shard file not used in training.

### 6.1 Training loss parity (via ClearML scalars, no separate run)

Pre-registered decision thresholds, read off the EMA of the last ~1,000 training-loss values:

- **Parity:** NoPE variant within **±0.05 nats** of baseline at step 30,500.
- **Better:** NoPE variant ≥0.02 nats lower than baseline.
- **Regression:** NoPE variant >0.05 nats higher than baseline.

Also monitor `gnorm` trajectories and `router_imbalance_*` scalars; unexpected divergence from baseline on either is a side-finding worth noting.

### 6.2 Length generalization

- Held-out sequences at lengths `{2048, 4096, 8192, 16384}`.
- For length > `max_seq_len=4096` the baseline/partial variants require regenerating `freqs_cis` at the longer length; Scoped NoPE requires no adjustment for the recurrent block (its inner attention is NoPE) but still needs longer `freqs_cis` for prelude/coda RoPE.
- **Criterion:** Scoped NoPE shows ≥5% relative-PPL-degradation improvement over baseline at 4096 and 8192 → **reproduces Kazemnejad at recurrent-depth scale**, novel contribution.

### 6.3 Depth generalization (novel axis)

- Held-out sequences, fixed seq_len=2048.
- Sweep `n_loops ∈ {16, 32, 48, 64, 96}` with `bypass_act=True`. 32 is the training maximum; {48, 64, 96} are pure extrapolation.
- **Criterion:** Scoped NoPE shows a flatter / monotonically-improving PPL-vs-`n_loops` curve relative to baseline beyond `n_loops=32` → **novel evidence that NoPE aids depth-extrapolation in recurrent-depth transformers**. Publication-worthy.
- **Expected mechanism check:** baseline RoPE's complex phasors can accumulate numerical drift past 32 iterations; Scoped NoPE's recurrent block avoids this. If baseline degrades sharply past `n_loops=48` and Scoped doesn't, we have mechanistic evidence.

Implemented as `evaluations/eval_length_gen.py` and `evaluations/eval_depth_gen.py`, each cloning `evaluations/eval_checkpoint.py`'s scaffolding (distributed init, checkpoint load, loguru logging).

## 7. Decision matrix (pre-registered)

Read the three eval axes together, then pick from:

| Outcome | §6.1 parity | §6.2 length-gen | §6.3 depth-gen | Action |
|---|---|---|---|---|
| **Green — Scoped wins** | Parity or better | Better than baseline | Better than baseline | Migrate production to Scoped NoPE after the next full run cycle |
| **Partial win — MLA wins** | Parity | Comparable | Comparable or better | Adopt `qk_rope_head_dim=0` as a one-line config flip; keep the `pe_mode` flag for future experiments |
| **Mixed** | Parity on one, regression on another | Mixed | Mixed | Keep full RoPE as default; publish the ablation as a negative / neutral result; revisit at larger scale if resources permit |
| **Red — NoPE breaks** | Regression > 0.05 nats on any variant | N/A | N/A | Keep full RoPE; log as a recurrent-depth-specific negative result for the literature |

## 8. Risks & mitigations

1. **1 B tokens may be underpowered for fine-grained depth-gen claims.** — Decision matrix leans conservative on deltas; only pre-registered thresholds are claimed.
2. **Stochastic-depth variance makes per-step losses noisy.** — Compare via EMA over last ~1,000 steps, not single-step loss. Same seed → same `n_loops` sequence across variants.
3. **MLA with `qk_rope_head_dim=0` may interact with MLA latent compression in non-obvious ways.** — Smoke-test unit test on the forward pass before a full run (covered in §4 tests).
4. **Scoped NoPE at very shallow `n_loops` (1 or 2) may not develop the implicit-position signal adequately.** — This is part of what we're measuring; flag rather than suppress in the writeup.
5. **BlueVela preemption during a 20 h run** — each variant's training script already auto-resumes from the latest checkpoint in its directory, and `keep_last=3` protects against premature rotation. Resubmit on preemption.

## 9. Artifact checklist

Prep-now deliverables (before launch):

- [ ] `open_mythos/main.py` — `pe_mode` plumbing through `MLAttention` / `GQAttention` / `TransformerBlock` / `RecurrentBlock`.
- [ ] `open_mythos/variants.py` — `mythos_1b_scoped_nope()`, `mythos_1b_partial_nope()`.
- [ ] `tests/test_nope.py` — equivalence-at-position-0, state_dict round-trip, KV-cache correctness, partial-NoPE MLA smoke test.
- [ ] `training/1b_poc_fineweb.py` — local `variant` default + `--variant` CLI flag override; variant-aware ClearML task name and checkpoint dir.
- [ ] `deploy/bluevela/bsub_nope_ablation.sh` — submits all three variants as separate bsub jobs.
- [ ] `evaluations/eval_length_gen.py` — length-generalization sweep.
- [ ] `evaluations/eval_depth_gen.py` — depth-generalization sweep.
- [ ] `docs/logbook/2026-04-29-nope-ablation-queued.md` — companion logbook marking this as queued for launch on or after 2026-05-01.
- [ ] Update `docs/logbook/2026-04-28-option-b-and-upstream-pr.md` to add NoPE ablation as a new open item.

All code artifacts land on a dedicated branch (`feat/nope-ablation`). The three training runs differ only in the `--variant` CLI flag passed at submission time; no code changes between runs.

## 10. Open questions (to resolve during the implementation-plan phase)

- Whether to add the real-valued RoPE refactor (cos/sin split) as a *within-study* extra variant, or defer to a follow-up. Currently deferred; revisit if any NoPE variant shows a clear training-loss regression and we need to isolate whether RoPE's numerical behavior vs its inductive bias is the load-bearing part.
- Whether to log per-layer attention-score entropy during training for each variant. Cheap to add and would provide mechanistic evidence for the NoPE-builds-implicit-position hypothesis.
- Whether to gate the go/no-go on depth-gen at `n_loops=96` specifically, versus letting the full curve shape drive the decision.
