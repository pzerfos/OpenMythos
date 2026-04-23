# Flash Attention Integration & Throughput Investigation

**Date:** 2026-04-23
**Goal:** Integrate upstream Flash Attention 2 support, install on BlueVela, and investigate throughput bottlenecks.

---

## Upstream Merge

Merged 3 upstream commits (`7d78ebe`, `963e112`, `227dbb1`):
- **Flash Attention 2** in GQAttention — conditional import, native GQA handling, fallback to manual attention
- **Causal mask dtype fix** — mask now created in activation dtype (resolves code review item #9)
- Tests moved to `tests/`, new benchmarks and examples added
- Conflict resolution: preserved our FSDP dtype casts in the fallback attention path

## flash-attn Installation

Built `flash-attn 2.8.3` from source on BlueVela login node:
- `CUDA_HOME=/opt/share/cuda-13.0`
- Installed into conda env `openmythos`
- Import verified: `from flash_attn import flash_attn_func` works

## Training with Flash Attention (Job 29260)

4 GPUs, FSDP, micro_batch=1, grad_accum=4, flash-attn enabled.

```
step  4/305175 | loss 12.6406 | gnorm 4.08 | lr 4.50e-07
step 10/305175 | loss 12.6250 | gnorm 4.33 | lr 1.35e-06
step 15/305175 | loss 12.5312 | gnorm 4.42 | lr 2.10e-06
step 18/305175 | loss 12.4688 | gnorm 4.49 | lr 2.55e-06
```

Loss dropping (12.64 → 12.47), training stable.

### Key Finding: Flash Attention Does Not Improve Step Time

| Config | Step time |
|--------|-----------|
| Manual attention (job 26518) | ~16s/step |
| Flash attention (job 29260) | ~16s/step |

**Flash-attn is not the bottleneck.** At seq_len=2048 with micro_batch=1, attention is cheap. The dominant cost is the **MoE dispatch loop** — 256 Python iterations per forward pass (topk=4 x n_experts=64), executed 16 times per step (once per recurrent loop). That's 4,096 Python-level expert dispatches per forward, plus backward.

### Throughput Analysis

- ~16s/step, 32,768 tokens/step → ~2,048 tokens/s
- 10B tokens target → 305,175 steps → **~56 days**
- MoE dispatch loop is the critical path for speedup

---

## Findings & Observations

*(will be updated as the day progresses)*

---

## Next Steps

1. **Optimize MoE dispatch** — Replace nested Python loop with grouped/batched dispatch. This is the #1 throughput lever.
2. **Monitor job 29260** — Let it run to accumulate more convergence data
3. **Code review fixes** — ACT remainder, MoE epsilon, router_bias (see `docs/code-review-2026-04-23.md`)
