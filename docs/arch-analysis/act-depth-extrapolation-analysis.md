# ACT vs Depth Extrapolation: Architectural Analysis

**Date:** 2026-04-23
**Context:** Upstream empirical work (kyegomez/OpenMythos#28, 13 ablation runs)
combined with our FSDP deadlock investigation (issue #4) and premature halting
concern (issue #5).

---

## The Two Promises

OpenMythos advertises two properties of its recurrent-depth architecture:

1. **Adaptive Computation Time (ACT):** per-token variable compute — easy tokens
   halt early, hard tokens use more loop iterations. Saves inference cost.

2. **Depth Extrapolation:** train at `n_loops=N`, infer at `n_loops=N+k` for
   harder problems. More compute at inference unlocks new capability without
   retraining. Cited from Saunshi et al. (2025) and the Parcae architecture.

These are fundamentally different value propositions:

| | ACT (adaptive compute) | Depth extrapolation |
|---|---|---|
| **Goal** | Save cost on easy inputs | Unlock capability on hard inputs |
| **Direction** | Fewer loops than training | More loops than training |
| **Mechanism** | Learned halting policy | Emergent generalization |
| **Constraint** | Stays within trained depth budget | Goes beyond trained depth |

## The Empirical Conflict

Upstream work (13 runs, 117M params, 491M tokens each on H100) systematically
ablated every architectural component to identify what binds the model to its
training loop count. Summary:

| Ablation | Removes depth-binding? |
|----------|----------------------|
| Disable `loop_index_embedding` + per-loop LoRA | No |
| Freeze LTI injection (`log_A`, `log_dt`, `B`) | No |
| Freeze MoE router weights | No |
| Break recurrence (`h = trans_out` instead of LTI) | No |
| **Disable ACT halting** | **Yes** |

**ACT is the only mechanism that creates the depth-binding.** With ACT enabled:
- Fixed-loop training produces a sharp V-shaped PPL curve centered on the
  training depth (PPL doubles if you go 4 loops beyond training depth)
- Random-loop training produces a flat plateau (more loops buy nothing)

With ACT disabled + random-loop training:
- First and only monotonically decreasing PPL curve (131 → 60 from 1 to 12 loops)
- The qualitative shape the depth-extrapolation literature promises

## Why ACT Prevents Depth Extrapolation

The ACT weighted sum is:

```
h_out = w_1·h_1 + w_2·h_2 + ... + w_T·h_T
```

where weights `w_t` come from the learned halting probabilities and sum to 1.0.

The model learns a **weighting policy** during training that is specific to the
training depth distribution. This policy encodes "at iteration t, this position
should have contributed X% of its final value." At inference depths outside the
training distribution:

- **Fewer loops than training:** the policy hasn't assigned enough weight yet
  when the loop ends prematurely. The remainder trick patches this numerically
  but the hidden states haven't been refined enough. Result: degraded output.

- **More loops than training:** the policy has already assigned all weight by
  the trained depth. Extra iterations produce hidden states that receive zero
  weight in the sum. The model literally cannot use the additional compute
  because ACT has already "closed the books."

Without ACT (returning `h_T` from the final iteration directly), the model is
forced to produce a usable hidden state at every depth. This implicit constraint
makes the function extrapolate — more iterations refine `h` further, and the
model works at any depth.

## The Design Space

### Option A: ACT for inference savings (current approach)

Keep ACT. Train at a generous depth (e.g., `n_loops=16`). Accept that inference
is bounded by the training depth. ACT saves cost on easy tokens by halting early
within that budget.

**When to use:** Inference cost is the priority. The model serves a fixed class
of problems and doesn't need to scale beyond its training compute budget.

**Requirements:**
- ACT initialization must not cause premature halting (issue #5). The model
  needs to learn to use the full depth when inputs demand it, not halt at
  iteration 3/16 by default.
- Possible mitigations: negative halt bias initialization, threshold annealing
  during warmup, minimum iterations floor.

**Limitations:**
- No depth extrapolation. `n_loops` at inference is capped at the trained value.
- Training cost is proportional to the maximum depth you want available.

### Option B: Depth extrapolation (disable ACT)

Disable ACT. Return `h_T` directly from the final loop iteration. Train with
stochastic depth sampling (random `n_loops` per step).

**When to use:** The model needs to handle problems of variable difficulty at
inference, and you want to trade more inference compute for better answers on
hard inputs.

**Requirements:**
- Stochastic depth sampling during training (uniform or curriculum over a range)
- No ACT halting — the model must produce valid output at any depth

**Limitations:**
- No per-token adaptive compute. Every token in a batch uses the same number
  of iterations (set at inference time).
- Batch-level depth selection is possible (easy batches get fewer loops) but
  per-token granularity is lost.

### Option C: Soft attention over loop outputs (untested)

Replace the ACT weighted sum with a learned soft attention over loop outputs,
trained jointly with stochastic depth sampling. Instead of a halting probability
that "closes the books," use an attention mechanism that can redistribute weight
across all available iterations at any depth.

**Hypothesis:** This could give both adaptive compute (attention weights shift
per token) and depth extrapolation (the attention mechanism generalizes to
unseen depths because it operates over the available loop outputs rather than
encoding a fixed-depth weighting policy).

**Status:** Untested. The upstream 13-run matrix did not evaluate this.
Would require architectural changes to `RecurrentBlock` to replace the ACT
accumulation with a post-hoc attention over `[h_1, h_2, ..., h_T]`.

## Recommendation for OpenMythos

**Near-term (current PoC):** Option A. The ongoing 1B-token run (job 34019)
validates the pipeline with ACT enabled. Fix the initialization (issue #5) so
the model learns to use all 16 iterations when needed.

**Next experiment:** Run a second 1B-token training with ACT disabled +
stochastic depth (Option B) on the same data and compare:
- Loss at the trained depth range
- Depth extrapolation: evaluate at `n_loops` outside the training range
- Inference cost: fixed cost per token vs ACT savings

**Future investigation:** Option C (soft attention over loop outputs) as a
potential way to get both properties. This is the most architecturally
interesting direction but needs design work before implementation.

## References

- Graves, A. (2016). "Adaptive Computation Time for Recurrent Neural Networks."
  arXiv:1603.08983 — original ACT paper.
- Saunshi et al. (2025). Referenced by OpenMythos README for depth-extrapolation
  property of looped transformers.
- kyegomez/OpenMythos#28 — 13-run empirical ablation study by @tonyzdev.
  Conclusively identifies ACT as the depth-binding mechanism. Reproduction code
  and full eval logs available.
- pzerfos/OpenMythos#4 — ACT early exit FSDP deadlock (fixed).
- pzerfos/OpenMythos#5 — ACT premature halting during warmup (open).
