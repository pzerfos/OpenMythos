# ACT Early Exit Causes FSDP Deadlock

**Date:** 2026-04-23
**Goal:** Investigate why training consistently hangs after ~33 steps on 4-GPU FSDP runs.

---

## Symptom

Every 4-GPU training run on BlueVela hangs at the same point (~step 33) and
dies 10 minutes later from an NCCL watchdog timeout. This was initially
attributed to preemption on the `preemptable` queue, but the pattern is too
consistent — same step count, same 600s timeout, same crash signature across
multiple runs on different nodes and code versions.

Timeline from job 33050 (post-MoE-optimization, node p2-r17-n1):

```
18:46:13 — step 33 completes normally (loss 11.80, gnorm 5.94)
           ↓ training hangs — no more steps logged
18:56:15 — NCCL watchdog fires after 600s timeout
18:57:23 — torchrun sends SIGTERM, all ranks SIGABRT
```

Previous run (pre-optimization, job 31808, node p5-r24-n2) shows the identical
pattern: training hangs after step 32, NCCL timeout 600s later.

## Diagnosis

The NCCL error dump reveals a **rank desync** — the ranks are stuck at different
collective sequence numbers:

| Rank | SeqNum stuck at | Last enqueued | Last completed |
|------|----------------|---------------|----------------|
| 0    | 4777           | 4777          | 4776           |
| 2    | 4777           | 4777          | 4776           |
| 3    | 4777           | 4777          | 4776           |
| **1**| **4782**       | **4785**      | **4781**       |

Ranks 0, 2, 3 are stuck waiting at collective 4777 (`_ALLGATHER_BASE`,
NumelIn=227,787,968 — an FSDP parameter unshard). Rank 1 advanced **5 more
collectives** past the others before getting stuck waiting for them in return.

This is the classic signature of a collective ordering mismatch: rank 1 executed
a different number of FSDP all-gathers than the other ranks, so they can never
synchronize again.

## Root Cause: ACT early exit in RecurrentBlock

The bug is at `open_mythos/main.py:915-916`:

```python
# Inside RecurrentBlock.forward(), in the recurrent loop:
if halted.all() and kv_cache is None:
    break
```

`halted.all()` evaluates on the **local batch** of each rank independently.
Each rank processes different data, so ACT halting probabilities differ per rank.
When rank 1's batch has all positions halt at loop iteration `t` but ranks 0, 2, 3
still have active positions, rank 1 breaks out of the loop while the others
continue to iteration `t+1`.

Each additional loop iteration calls `self.block(combined, freqs_cis, ...)`,
where `self.block` is an FSDP-wrapped `TransformerBlock`. That triggers
`_ALLGATHER_BASE` to unshard the block's parameters. When rank 1 has already
exited the loop, it never issues these collectives — the remaining ranks wait
forever, and the NCCL watchdog kills everything after 600s.

### Why it triggers at ~step 33

At initialization, the ACT `halt` linear layer has random N(0, 0.02) weights.
`sigmoid(0) = 0.5`, so halting probability starts around 0.5 per iteration. With
`act_threshold=0.99`, cumulative probability crosses 0.99 after ~3-4 iterations,
causing `halted.all()` to be True on most batches. After ~30 steps of training
the halting weights begin to diverge from their initial values, creating slight
differences in halting patterns across ranks. As soon as one rank's batch halts
one iteration earlier than another's — deadlock.

### Why it wasn't caught earlier

- **Single-GPU training:** No collectives, `break` is safe.
- **Tests:** All run on CPU, single-process. The 48 ACT tests verify weight
  invariants and halting correctness but can't detect a multi-rank collective
  ordering bug.
- **The bug is data-dependent:** It only fires when one rank's batch halts before
  another's. Early in training (steps 1-30) the halting probabilities are nearly
  uniform across all data, so all ranks exit at the same iteration. It takes a
  few dozen steps for the learned halting weights to create enough per-batch
  variance to desynchronize the ranks.

## Relationship to ACT's Purpose

Adaptive Computation Time (Graves, 2016) is designed to let the model allocate
variable compute per token — easy tokens halt early, hard tokens keep looping.
The `halted.all()` early exit is an optimization: if every position in the batch
has converged, skip the remaining iterations to save compute.

This optimization is fundamentally valid for the model's semantics — halted
positions get zero weight on subsequent iterations regardless of whether the loop
continues. The problem is purely a **systems-level interaction with FSDP**: the
early exit causes different ranks to execute different numbers of FSDP-sharded
forward passes, which is an illegal state in collective communication.

Key observations for any fix:

1. **ACT halting itself is not the problem.** The per-position weighting
   (halted positions contribute zero) works correctly regardless of whether
   the loop runs to completion. The weighted sum of hidden states produces the
   same result whether we skip iterations with zero weight or execute them.

2. **The early exit is an optimization, not a correctness requirement.** Removing
   it changes performance (unnecessary iterations run) but not model output.

3. **The constraint is that all ranks must execute the same number of
   FSDP-sharded forward passes.** Any fix must ensure this while preserving as
   much of the ACT compute-saving benefit as possible.

## Possible Fix Approaches

All four approaches produce **identical model outputs**. The per-position
weighting already handles halted positions correctly — after halting,
`still_running` is False, so `weight * still_running.float()` is zero and
`h_out` doesn't change for those positions. The `break` is purely a compute
optimization, not a correctness mechanism. The choice is about compute
efficiency and implementation complexity.

### Compute savings at stake

For the 1B config (16 loop iterations, `act_threshold=0.99`), sigmoid
initialization near 0.5 means cumulative halting probability crosses 0.99
after ~3-4 iterations. The early exit saves **~75% of recurrent block compute**
— 12 out of 16 iterations of the FSDP-sharded TransformerBlock + MoE + LoRA.

### A. Disable early exit under FSDP (simplest)

```python
if halted.all() and kv_cache is None and not is_fsdp:
    break
```

Preserves single-GPU early exit. Under FSDP, all ranks always run all `n_loops`
iterations. Correct but gives up the compute savings of ACT under distributed
training.

- **Cost:** Loses ~75% savings — every rank runs all 16 iterations even when all
  positions halted at iteration 3. Expect ~3-4x slower steps in the recurrent
  block (from ~2.5s toward ~8-10s, since prelude/coda/embedding are outside the
  loop). Still faster than the pre-MoE-optimization 16.7s.
- **Benefit:** Zero complexity, zero risk of new bugs.

### B. All-reduce the halting flag (preserve early exit)

```python
if kv_cache is None:
    all_halted = halted.all()
    if ddp:
        all_halted_tensor = torch.tensor([all_halted], device=h.device)
        dist.all_reduce(all_halted_tensor, op=dist.ReduceOp.MIN)
        all_halted = all_halted_tensor.item() > 0
    if all_halted:
        break
```

All ranks agree on whether to exit. Early exit only fires when *every position
across every rank* has halted. Adds one all-reduce per loop iteration.

- **Cost:** One scalar `all_reduce(MIN)` per iteration: ~10-50us on intra-node
  NCCL. Over 16 iterations at 2.5s/step, that's ~0.01-0.03% overhead. Negligible.
  Introduces `torch.distributed` dependency into model code — `RecurrentBlock`
  currently has no distributed awareness. Need to thread a `ddp` flag or detect
  from context.
- **Benefit:** Preserves early exit. With 4 ranks seeing correlated data
  distributions, they tend to halt within 1-2 iterations of each other. If ranks
  halt at iterations 3, 3, 4, 5 → exits at 5 (saves ~69% vs all 16). The
  compute between the earliest and latest halt is wasted on already-halted ranks,
  but this is a small fraction of total loop compute.
- **Worst case:** One rank never halts → degrades to A for that step (rare).

### C. Mask computation instead of breaking

Don't break out of the loop. Instead, skip the expensive computation for halted
positions by masking them before the transformer block.

- **Cost:** Very complex to implement correctly. Attention couples all positions
  in the sequence — to truly save FLOPs you'd need to compact non-halted
  positions into a smaller tensor, run the block on just those, then scatter
  back. This creates variable tensor shapes across ranks (different numbers of
  halted positions), which is exactly the kind of shape mismatch that breaks
  FSDP collectives. Alternatively, zero out halted positions but still run
  full-size tensors — but `zeros @ W` costs the same as `x @ W` on a GPU.
  No actual FLOPs saved.
- **Benefit:** Conceptually closest to ACT intent, but in practice delivers
  marginal-to-no savings for high complexity. Not worth it.

### D. No-op forward for halted ranks

When a rank's batch is fully halted, still call `self.block(...)` but with
zero-weight input, ensuring the FSDP collectives fire but the GPU does the
minimal work.

- **Cost:** The GPU still does the full work — `zeros @ W` is the same cost as
  `x @ W`. No FLOPs saved unless explicit short-circuit logic is added inside
  `TransformerBlock`, which gets messy and fragile.
- **Benefit:** None over A, with more complexity. Strictly worse than B.

### Recommendation

**B is the sweet spot.** All-reduce cost is negligible (~0.01% overhead), compute
savings are real (~69-75%), and C/D don't deliver on their promise. The
architectural boundary concern (distributed awareness in model code) is
manageable — pass a boolean flag, don't import `dist` unless needed.

---

## ACT Initialization: Premature Halting During Warmup

Separate from the FSDP deadlock, there is an architectural concern about ACT
behavior during early training that warrants future investigation.

With the current initialization (halt layer weights from N(0, 0.02), no bias
override), `sigmoid(~0)` ≈ 0.5 per iteration. Cumulative halting probability
crosses the 0.99 threshold after just 3-4 iterations out of 16. This means the
model is effectively using only ~20-25% of its recurrent depth budget from the
very first step.

The recurrent depth is the core architectural advantage of the Recurrent-Depth
Transformer — deeper iterations enable more complex reasoning chains. If the
model halts at iteration 3/16 during warmup, it never learns to use iterations
4-16, because those iterations receive zero gradient (halted positions have zero
weight). The model may be stuck in a local minimum of shallow computation.

Possible mitigations:
- **Initialize halt bias to a negative value** (e.g., -2.0 → sigmoid ≈ 0.12),
  so early training uses most iterations and the model learns *when* to halt
  rather than halting by default.
- **Anneal the ACT threshold** during warmup — start with threshold=1.0
  (never halt) and decay toward the target threshold over the first N steps.
- **Minimum iterations floor** — always run at least `min_loops` iterations
  regardless of halting, with ACT only controlling iterations beyond that.

This is tracked as a separate issue (pzerfos/OpenMythos#5) since it affects
model quality independent of the FSDP fix.

## Status

- Bug confirmed via NCCL collective sequence analysis
- Root cause identified: ACT `halted.all()` early exit + FSDP rank desync
- Reproduced on two independent runs (jobs 31808 and 33050, different nodes)
- Fix approach: B (all-reduce halt flag) — recommended
- Separate concern: premature ACT halting during warmup (issue #5)
- **Fix applied:** approach B — all-reduce halt flag with `dist.all_reduce(MIN)`
  in `RecurrentBlock.forward()` (commit pending)
- 10 new tests in `tests/test_act_fsdp_fix.py` — all pass
- 252 total tests pass (14 pre-existing failures in test_main.py unchanged)
