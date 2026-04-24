# 1B PoC Evaluation & Architecture Analysis

**Date:** 2026-04-24
**Goal:** Complete the 1B-token training run, evaluate the checkpoint, and compare the OpenMythos architecture against Qwen3.6.

---

## Training Run Completed

Job 34019 (submitted 2026-04-23 21:33 UTC) ran uninterrupted for ~16 hours on
the preemptable queue (`p5-r06-n1`, 4 GPUs) — the longest sustained run of the
PoC campaign.

| Metric | Value |
|--------|-------|
| Final step | 31,225 (target was 30,518 for 1B tokens) |
| Tokens seen | 1.02B |
| Final loss | ~3.3-3.5 (point, not smoothed) |
| Gradient norm | 0.65-0.70 |
| Learning rate | 2.94e-04 (cosine decay, barely past peak) |
| Step time | ~2.1s |
| Last checkpoint | `step_0031000.pt` (18GB) |

Loss evolution over the full run (sampled every 500-1000 steps from job logs):

| Step | Loss | Tokens | Notes |
|-----:|-----:|-------:|-------|
| 1 | 12.66 | 0.0B | Near random (ln(200K) = 12.2) |
| 31 | 11.91 | 0.0B | Convergence confirmed |
| 107 | 8.50 | 0.0B | Rapid early drop |
| 1,001 | 5.54 | 0.0B | Checkpoint resume (job 34019) |
| 2,000 | 4.93 | 0.1B | End of warmup, LR at peak 3e-4 |
| 5,000 | 4.16 | 0.2B | |
| 10,000 | 3.57 | 0.3B | |
| 15,000 | 3.57 | 0.5B | |
| 20,000 | 3.54 | 0.7B | |
| 25,000 | 3.26 | 0.8B | |
| 28,929 | 3.54 | 0.9B | |
| 31,225 | 3.43 | 1.0B | Final (killed manually) |

The 1B-token PoC target is met. Training was stable throughout: no NaN/Inf, no
preemptions, no deadlocks after the ACT all-reduce fix.

---

## Checkpoint Evaluation

### Setup

Created `evaluations/eval_checkpoint.py` — standalone single-GPU evaluation
script that loads a checkpoint and runs:
1. Qualitative generation from 10 diverse prompts (factual, reasoning, creative,
   technical, math)
2. Depth extrapolation sweep: eval loss/PPL at n_loops = 1, 2, 4, 8, 12, 16, 24, 32

Submitted as BlueVela LSF job (`deploy/bluevela/bsub_eval.sh`, 1 GPU, preemptable).

### Bug: temperature=0.0 crashes model.generate()

`model.generate()` divides logits by temperature before softmax (`main.py:1130`).
With `temperature=0.0`, this produces inf, which propagates to NaN in softmax
and crashes `torch.multinomial`. Fixed in the eval script by using
`temperature=0.01` for near-greedy decode.

### Generation Samples (n_loops=16, temperature=0.8)

Model checkpoint: `step_0031000.pt` (1.408B total params, ~352M active/token)

| Quality | Observation |
|---------|-------------|
| Fluency | Excellent — grammatically correct, natural English |
| Coherence | Good within ~50 tokens, degrades over longer sequences |
| Factual accuracy | Poor — confuses concepts across domains |
| Math/logic | Very poor — "1+1" produces nonsense |
| Tone | Strongly educational/expository (matches FineWeb-Edu) |
| Repetition | Several samples degenerate into repetitive loops after ~150 tokens |

Near-greedy decode (temperature=0.01) immediately falls into a repetition loop
("The theory is based on the assumption that..." repeated verbatim).

This is expected for a 1B-active-param model trained on only 1B tokens — coherent
language modeling but insufficient data for reliable knowledge or reasoning.

### Depth Extrapolation Sweep

Eval dataset: last 2 parquet files from FineWeb-Edu (500K tokens held out).

| n_loops | Loss | PPL | vs trained (16) |
|--------:|-----:|----:|:----------------|
| 1 | 4.828 | 124.95 | +1.457 |
| 2 | 4.475 | 87.80 | +1.105 |
| 4 | 4.193 | 66.23 | +0.823 |
| 8 | 4.257 | 70.59 | +0.886 |
| 12 | 4.082 | 59.28 | +0.712 |
| **16** | **3.371** | **29.09** | **0.000 (trained)** |
| 24 | 3.374 | 29.18 | +0.003 |
| 32 | 3.374 | 29.18 | +0.003 |

**Key findings:**

1. **No depth extrapolation.** n_loops=24 and 32 give identical loss to 16
   (+0.003, within noise). The model gains nothing from extra iterations.

2. **No depth collapse.** Unlike the upstream U-shape finding where PPL increases
   at unseen depths, our model flatlines beyond 16. ACT halting zeros out
   contributions from extra iterations — harmless but useless.

3. **Sharp cliff at trained depth.** The 12 -> 16 jump is massive (PPL 59 -> 29).
   The model strongly depends on all 16 iterations and doesn't degrade
   gracefully with fewer loops.

4. **Non-monotonic below trained depth.** n_loops=8 is worse than n_loops=4
   (loss 4.26 vs 4.19), suggesting intermediate loop states aren't well-ordered
   for early truncation.

This directly confirms the upstream finding from kyegomez/OpenMythos#28: ACT
binds the model to its training depth. The model learns to produce valid output
at exactly n_loops=16 but cannot leverage additional compute. This validates
the concern in pzerfos/OpenMythos#5.

---

## Architecture Comparison: OpenMythos vs Qwen3.6-35B-A3B

Wrote a detailed comparison document (`docs/arch-analysis/openmythos-vs-qwen3.6.md`)
covering engineering trade-offs between the two architectures. Qwen3.6 config
sourced from HuggingFace `config.json` and cross-referenced with the Qwen3.5-Omni
technical report (arXiv:2604.15804). Key architectural gaps identified:

1. **Gated DeltaNet hybrid attention** — Qwen3.6 uses 75% linear attention
   (O(1) state per layer, no KV cache) + 25% full GQA. Constant-memory
   inference for most layers vs OpenMythos's MLA which still grows with seq_len.

2. **Fine-grained MoE** — Qwen3.6: 256 experts x dim 512, top-8.
   OpenMythos: 64 experts x dim 2048, top-4. Smaller, more numerous experts
   give better load balancing.

3. **Partial rotary + mRoPE** — Qwen3.6 applies RoPE to only 25% of dims
   with YaRN scaling to 1M context. OpenMythos applies full RoPE, no
   extension logic.

4. **Multi-token prediction** — Qwen3.6 trains with MTP for denser signal.
   OpenMythos does not.

Notable: Qwen3.6-35B-A3B and Qwen3.5-35B-A3B share an identical architecture
(confirmed via HF discussion). The 3.5 -> 3.6 bump is training/data only.

---

## Files Created/Modified

| File | Action |
|------|--------|
| `evaluations/eval_checkpoint.py` | Created — checkpoint eval with generation + depth sweep |
| `deploy/bluevela/bsub_eval.sh` | Created — LSF job submission for eval |
| `docs/arch-analysis/openmythos-vs-qwen3.6.md` | Created — architecture comparison |
| `docs/logbook/2026-04-24-eval-and-analysis.md` | Created — this logbook |

## Commits

```
e70cb9c docs: architecture comparison of OpenMythos 1B vs Qwen3.6-35B-A3B
4a97c63 feat(eval): add checkpoint evaluation script with generation + depth sweep
d1c51b2 fix(eval): use temperature=0.01 instead of 0.0 for greedy decode
```

---

## Next Steps

### Immediate

1. **Decide on ACT** (pzerfos/OpenMythos#5) — The depth sweep confirms ACT
   depth-binding. Three options remain:
   - **A.** Keep ACT, accept fixed-depth operation (current state)
   - **B.** Disable ACT, train with stochastic depth sampling — enables
     depth extrapolation per upstream findings
   - **C.** Replace ACT weighted sum with soft attention over loop outputs
     (untested)

2. **Scale to 10B tokens on 16 GPUs** — The 1B PoC validates the full
   pipeline. Submit a longer run on the normal queue (`-q normal -G grp_granite_`)
   with 16 GPUs for ~2.5 days to 10B tokens.

3. **Add lm-eval-harness integration** — Standard benchmarks (HellaSwag, ARC,
   MMLU) would allow comparison with published results for similarly-sized
   models.

### Deferred

4. **Fix 14 pre-existing test failures** — RoPE dimension mismatch after
   upstream flash-attn merge (13 tests) + LTI spectral radius boundary (1 test)

5. **router_bias load balancing** (pzerfos/OpenMythos#3) — Most impactful
   deferred code review item. Expert utilization imbalance grows with longer
   training.

6. **FSDP1 -> FSDP2 migration** (pzerfos/OpenMythos#6) — Lower memory,
   torch.compile support. Worth doing before the 10B run.

7. **Study Gated DeltaNet** — Qwen3.6's hybrid attention is a generation ahead
   for long-context efficiency. Worth investigating for future OpenMythos
   iterations.
