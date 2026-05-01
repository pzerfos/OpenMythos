# NoPE Ablation — Results and Decision

**Date:** 2026-05-01
**Status:** Complete.
**Decision:** **RED — keep full RoPE; log as a recurrent-depth-specific negative result.**

Spec and plan: `docs/superpowers/specs/2026-04-29-nope-for-recurrent-depth-design.md`, `docs/superpowers/plans/2026-04-29-nope-ablation.md`.
Training runs (three variants × 1 B tokens × 4 × H100, preemptable queue): `docs/logbook/2026-04-29-nope-ablation-launched.md`. Jobs 75472/75479/75480, all completed to step 30,517.
Post-training evals: BlueVela jobs **85192–85197** (depth-gen) and **85210–85212** (length-gen, resubmitted after a `freqs_cis` shape-mismatch fix — commits `d41254f` + `2bcc9a8`). Each eval is a 1 × H100, preemptable, ~1–7 min wall-clock job over 100 held-out FineWeb-Edu sequences from shard `013_00009.parquet` (not reached during the 1 B-token training).

Submission wrapper landed as `deploy/bluevela/bsub_nope_eval.sh` (parameterized `VARIANT ∈ {baseline, scoped, partial}` × `EVAL ∈ {length, depth}`).

---

## Results

### §6.1 Training-loss parity (EMA over final 1,000 optimizer steps, step 29,518 → 30,517)

Same RNG-broadcast-from-rank-0 recipe → identical `n_loops` sequence across variants per §5.

| Variant  | steps | mean loss | stddev | mean gnorm | mean n_loops |
|---------|-------|-----------|--------|------------|--------------|
| baseline | 1000  | **3.2859** | 0.201  | 0.662      | 15.91        |
| scoped   | 1000  | 3.4099     | 0.178  | 0.711      | 16.60        |
| partial  | 1000  | 3.4542     | 0.245  | 1.043      | 16.02        |

**Δ vs baseline:** scoped **+0.1239 nats**, partial **+0.1683 nats**. Pre-registered thresholds were ±0.05 nats for parity, +0.05 nats for regression → **both NoPE variants exceed the regression threshold by 2.5–3.4×**.

(The mean `n_loops` is slightly off between variants despite the seed-sync recipe; three preemption-and-resume cycles across the two resubmission rounds desynced the RNG state. Over 1,000 draws the Monte-Carlo effect on mean loss is small compared to the measured deltas.)

### §6.2 Length generalization (n_loops=16, 100 held-out sequences)

NLL at eval length:

| seq_len | baseline | scoped | partial |
|---------|---------|--------|---------|
| 2048    | 3.3281  | 3.4915 | 3.4247  |
| 4096    | 3.4056  | 3.5620 | 3.4579  |
| 8192    | 3.4238  | 3.5780 | 3.4958  |
| 16384   | 3.4265  | 3.5804 | 3.5123  |

Relative PPL degradation vs the 2048 reference:

| seq_len  | baseline    | scoped     | partial    |
|---------|-------------|------------|------------|
| 4096    | +8.07 %     | +7.34 %    | **+3.39 %** |
| 8192    | +10.08 %    | +9.05 %    | **+7.39 %** |
| 16384   | +10.36 %    | +9.35 %    | **+9.15 %** |

§6.2 criterion was "Scoped NoPE shows ≥5 % relative-PPL-degradation improvement over baseline at 4096 and 8192." Scoped's advantage over baseline is 0.73 pp @ 4096 and 1.03 pp @ 8192 — **well below the 5 pp bar**.

Partial NoPE shows the *best* relative length-gen (3.4 pp better than baseline @ 4096), but starts from a higher absolute NLL and stays worse than baseline at every length. Baseline's curve also flattens past 8192, suggesting it saturates — removing headroom for a length-gen win on this held-out shard.

### §6.3 Depth generalization (seq_len=2048, 100 held-out sequences)

n_loops sweep with `bypass_act=True`. Training max was 32; {48, 64, 96} are pure extrapolation.

| n_loops | baseline | scoped | partial |
|---------|---------|--------|---------|
| 16      | 3.3281  | 3.4915 | 3.4247  |
| 32 (trained max) | 3.3285 | 3.4919 | 3.4248 |
| 48      | 3.3307  | 3.4957 | 3.4258  |
| 64      | 3.3327  | 3.4974 | 3.4263  |
| 96      | 3.3338  | 3.4982 | 3.4274  |

**Δ across 32 → 96:** baseline +0.0053, scoped +0.0063, partial +0.0026 nats. All three curves are essentially flat over 3× the training depth; no variant shows the sharp RoPE-phasor-drift degradation the spec hypothesized.

§6.3 criterion was "Scoped NoPE shows a flatter / monotonically-improving PPL-vs-n_loops curve relative to baseline beyond n_loops=32." Scoped is **not** flatter than baseline (they are within ±0.001 nats of each other at every depth). The expected mechanism (baseline RoPE drift past the training horizon) **didn't materialize**; at 1 B tokens under stochastic-depth training with `n_loops ∈ [1, 32]`, baseline RoPE extrapolates cleanly to 96 loops.

Partial NoPE has the smallest Δ (+0.0026), but also starts from a worse NLL — consistent with §6.2.

## Interpretation

- **§6.1 alone triggers Red.** Both variants regress on training loss by ≥2.5× the pre-registered threshold. Per §7 of the spec, this is sufficient for the Red decision; the other two axes cannot rescue it.
- **§6.2 would have failed independently** even without §6.1: scoped missed the 5 pp length-gen bar by ~4×.
- **§6.3 would have failed independently**: no flatness advantage for scoped beyond training depth. The interesting observation here is the null result for the baseline — RoPE phasor drift past `n_loops=32` appears not to be a real problem at this training recipe (stochastic depth over [1, 32] for 1 B tokens), which calls the *motivation* of the scoped variant into question at this scale.

## Decision

**Red — keep full RoPE as default.** No production migration. No one-line config flip to `qk_rope_head_dim=0`. Close `pzerfos/OpenMythos#?` (NoPE feature) as a negative result; keep the `pe_mode` plumbing in place so future experiments (e.g., at larger scale, or with different training depths) don't need to re-implement the scaffolding.

Action items:
1. Don't disturb `training/1b_poc_fineweb.py`'s default (`variant = "baseline"`, all `pe_mode = "rope"`) — the 10 B production run and anything after uses full RoPE.
2. Archive the three NoPE checkpoints under `/proj/checkpoints/pzerfos/openmythos/checkpoints/nope-ablation/` — they cost compute to produce and are useful as reference models for any follow-up.
3. Annotate the PR #10 description on `github.ibm.com/pzerfos/OpenMythos/pull/10` with a post-merge-results pointer to this file, next to the existing post-merge-addendum for `897c9e5`.
4. No upstream PR of the NoPE feature to `kyegomez/OpenMythos`. A negative result at 1 B tokens isn't interesting enough to push, and the plumbing is non-trivial.

## Side-findings worth noting for the literature / follow-up

1. **Baseline RoPE does not break past training depth at 1 B tokens.** The spec's hypothesized failure mode (complex-phasor drift past n_loops > 32) produces a +0.005 nats regression over 3× the training depth — below the noise floor on a 100-sequence held-out evaluation. This *weakens the Kazemnejad-style motivation* for NoPE inside a recurrent-depth transformer at this training recipe and token budget. Whether larger scales (10 B+ tokens, deeper training-depth distributions) surface the effect is open.
2. **Partial NoPE (MLA `qk_rope_head_dim=0`) has the best relative length-gen.** It beats baseline by 3.4 pp @ 4096 and 1.2 pp @ 16384, but loses on absolute NLL everywhere. The training-loss regression (+0.17 nats) dominates, but the relative length-gen slope is a real (if small) effect worth a line in any future literature write-up.
3. **Partial NoPE's training gnorm was 1.57× baseline's** (1.04 vs 0.66, EMA over final 1,000 steps). Could indicate optimization difficulty with a zero-dim RoPE head in MLA's compressed-latent attention. Not revisiting unless someone re-opens partial NoPE.
4. **`eval_length_gen.py` had a latent shape-mismatch bug at load time.** The training script overrides `cfg.max_seq_len = 2048` but the variants default to 4096, so constructing the model from the variant config produced `freqs_cis` buffers at `[4096, *]` that couldn't strict-load from a `[2048, *]` checkpoint. Fixed in `2bcc9a8` by reading the training `max_seq_len` from the checkpoint's buffer shape before model construction. `eval_depth_gen.py` is unaffected because it uses `--seq-len 2048` (matching the checkpoint) and doesn't resize.

## Caveats and limitations

- **1 B tokens per variant may be underpowered** for fine-grained depth-gen claims. The spec warned of this in §8.1; the null-result observation for baseline RoPE at extrapolated depths may be a statement about training volume rather than RoPE itself.
- **Held-out set is 100 sequences from one parquet shard** (`013_00009.parquet`). Stable enough to resolve 0.1-nat deltas but marginal for the 0.005-nat inter-depth deltas in §6.3. A 1,000-sequence eval would tighten the §6.3 curves, but wouldn't change the Red verdict (§6.1 already triggers it).
- **Stochastic-depth training recipe** — results do not generalize to ACT training. If anyone re-runs NoPE under ACT (default-off but supported), re-evaluate.
- **Same seed ≠ same n_loops sequence after restarts.** Preemption-and-resume desyncs the per-step RNG state. For variance-sensitive ablations, checkpoint the RNG state alongside the model state. Not fixing for this run since the deltas we measure are an order of magnitude larger than the expected RNG-desync contribution.

## Closing the loop on the roadmap

Open items from `docs/logbook/2026-04-28-option-b-and-upstream-pr.md` that this entry resolves:

7. ~~**NoPE ablation study**~~ — **Red, as above.** `pe_mode` plumbing stays in the codebase as inactive infrastructure for future experiments.

Still-open (numbering matches `docs/logbook/2026-04-28-option-b-and-upstream-pr.md` §Remaining Open Items):

1. lm-eval-harness integration — not yet started.
4. FSDP1 → FSDP2 migration — feasibility-positive (~15 % speed-up, ~6 % peak-mem reduction on `stochastic_depth`); scheduling open.
5. Gated DeltaNet study — future architecture direction.
