# router_bias Load Balancing: Aux-Loss-Free MoE

**Date:** 2026-04-29
**Goal:** Wire up the existing `router_bias` buffer in `MoEFFN` so it actually drives load balancing across MoE experts during training (DeepSeek-V3's Algorithm 1). Close pzerfos/OpenMythos#3.

---

## The Problem

`MoEFFN` registers a `router_bias` buffer of shape `(n_experts,)` (`open_mythos/main.py:491`). The top-k expert selection already uses `logits + router_bias`, while the gating weights use unbiased `softmax(logits)` — this is DeepSeek-V3's "aux-loss-free" design: the bias steers *which* experts fire without distorting gradients.

The bias starts at zero and **is never updated during training**. So the machinery is inert, and routing drifts freely.

### Why this matters

With `n_experts=64` and `top_k=4`, ideal utilization is `6.25%` of tokens per expert. Left unmanaged, the router exhibits a positive-feedback loop:

1. Some experts see slightly more traffic early in training (random init).
2. Those experts get more gradient and improve faster.
3. Improved experts are picked more often.
4. Cold experts receive little gradient, stay close to random, get picked even less.
5. In the limit, the model collapses to using ~`top_k` experts and wastes the remaining parameter capacity.

This is the canonical "expert collapse" problem that every MoE paper describes. The standard fix in the DeepSeek-MoE line is the aux-loss-free bias scheme (DeepSeek-V3 §4.3, 2024), which OpenMythos's code is *structured* to implement but never actually runs.

### Impact on job 67208 (in flight)

Job 67208 has trained for ~122,500 steps with bias stuck at zero. We don't know how skewed expert utilization currently is because we've never measured it. The fix in this branch is off by default (`router_bias_update_rate=0.0`) — landing on main is safe. A separate decision (with diagnostic) determines whether to enable it mid-run.

## The Algorithm (DeepSeek-V3 Alg. 1)

After each optimizer step, for each MoE layer:

```
c[i] ← count of (token, slot) pairs that picked expert i since last update
total ← sum(c)                    # = N_tokens_this_step * top_k
target ← total / n_experts        # uniform allocation
for each expert i:
    if c[i] > target:   router_bias[i] -= u       # overused
    if c[i] < target:   router_bias[i] += u       # underused
c ← 0
```

`u` is small (paper uses `1e-3`). Because the bias never enters the softmax that produces gating weights, the forward-pass gradient is unaffected — only *selection* is nudged.

## Design

### 1. Count accumulation

Add a non-persistent `expert_counts` buffer of shape `(n_experts,)`, `int64`, to `MoEFFN`. In `forward`, after computing `topk_idx`, do:

```python
with torch.no_grad():
    self.expert_counts += torch.bincount(
        topk_idx.reshape(-1), minlength=self.n_experts
    )
```

No gradient flows through counts (they're integer buffers). `persistent=False` keeps them out of the checkpoint — they reset every optimizer step anyway.

### 2. Bias update

Add a method on `MoEFFN`:

```python
def update_router_bias(self, rate: float, ddp: bool) -> dict:
    """Apply DeepSeek-V3 Algorithm 1. Returns diagnostic stats."""
    if ddp:
        dist.all_reduce(self.expert_counts, op=dist.ReduceOp.SUM)
    total = self.expert_counts.sum()
    target = total.float() / self.n_experts
    diff = self.expert_counts.float() - target
    self.router_bias -= rate * torch.sign(diff)
    stats = {
        "max_over_mean": self.expert_counts.max() / target,
        "min_over_mean": self.expert_counts.min() / target,
        "stddev_over_mean": self.expert_counts.float().std() / target,
    }
    self.expert_counts.zero_()
    return stats
```

### 3. Top-level convenience

Add `OpenMythos.update_router_biases(rate, ddp) -> dict` that walks all `MoEFFN` modules and aggregates diagnostics (max over all layers).

### 4. Training-script integration

Follow the `recurrent_mode` pattern — local variables, not config fields:

```python
router_bias_update_rate = 0.0    # 0.0 = disabled
```

After `optimizer.step()`:

```python
if router_bias_update_rate > 0.0:
    stats = model.update_router_biases(router_bias_update_rate, ddp=ddp)
    if master and step % log_every == 0:
        log_clearml("router_imbalance_max_over_mean", stats["max_over_mean"], step)
        log_clearml("router_imbalance_stddev_over_mean", stats["stddev_over_mean"], step)
```

### 5. FSDP/multi-rank safety

- `router_bias` is a buffer, not a param, and `persistent=True` (it's part of `state_dict` for resumption). FSDP doesn't shard buffers by default.
- `expert_counts` is `persistent=False` — no checkpoint entry, no FSDP sharding concerns.
- Counts are all-reduced once per optimizer step. Small cost: `n_experts × int64 × n_moe_layers` per step — negligible vs. the activation all-gathers FSDP is already doing.
- `torch.sign(diff)` is rank-deterministic given the same summed counts, so `router_bias` stays in sync across ranks without explicit broadcast.

### 6. What we're explicitly NOT doing

- **No auxiliary load-balancing loss.** DeepSeek-V3 replaced aux loss with this bias scheme precisely because the aux loss distorts gradients.
- **No per-micro-step updates.** Update is once per optimizer step; counts accumulate across grad_accum micro-steps.
- **No change to `router_bias` init or the forward-path math.** The machinery was already present; we're just making it live.

## Implementation Plan

1. Add `expert_counts` buffer + bincount in `MoEFFN.forward`.
2. Add `MoEFFN.update_router_bias(rate, ddp)`.
3. Add `OpenMythos.update_router_biases(rate, ddp)`.
4. Wire into `training/1b_poc_fineweb.py` after `optimizer.step()` with `router_bias_update_rate=0.0` default.
5. Tests:
   - Counts accumulate correctly over multiple forward passes.
   - After updates, hot experts' bias decreases, cold experts' bias increases.
   - Router *output* unchanged by bias (gating weights remain `softmax(logits)`, not `softmax(logits+bias)`).
   - `rate=0.0` is a no-op.
   - Diagnostic stats are finite and reasonable.

## Outcome

Implemented on branch `feat/router-bias-load-balancing`. All 332 tests pass (7 new).

### Changes

- `MoEFFN`:
  - Added non-persistent `expert_counts` buffer (int64, shape `(n_experts,)`).
  - `forward()` accumulates `bincount(topk_idx.reshape(-1), minlength=n_experts)` under `torch.no_grad()`.
  - New `update_router_bias(rate, ddp=False)` — all-reduces counts across ranks (when `ddp=True`), shifts `router_bias` by `-rate * sign(counts - target)`, resets counts, returns `{max,min,stddev}_over_mean` stats.
- `OpenMythos.update_router_biases(rate, ddp=False)` — walks all MoE modules, returns worst-case imbalance stats across layers. Short-circuits when `rate == 0.0`.
- `training/1b_poc_fineweb.py`:
  - New local `router_bias_update_rate = 0.0` (disabled by default).
  - Call after `optimizer.step()`; unwraps `model.module` under DDP.
  - Emits two new ClearML scalars when enabled: `router_imbalance_max_over_mean` and `router_imbalance_stddev_over_mean`.

### Tests added (7)

- `test_expert_counts_accumulate` — counts add across forwards; total = `B*T*topk` per call.
- `test_update_router_bias_shifts_toward_balance` — with forced imbalance, hot experts' bias goes negative, cold experts' bias goes positive.
- `test_update_router_bias_resets_counts` — counts zeroed after update.
- `test_update_router_bias_rate_zero_is_noop` — `rate=0.0` leaves `router_bias` unchanged.
- `test_router_bias_does_not_affect_gating_weights` — output unchanged by bias increments that don't alter `topk_idx`.
- `test_update_router_biases_walks_all_moe_layers` — top-level wrapper touches every MoE layer.
- `test_update_router_biases_rate_zero` — top-level early-exit on `rate=0.0`.

### Current run (job 67208) — not enabled yet

Default on main remains `router_bias_update_rate = 0.0`. Enabling it mid-run is a separate decision; we'd need either (a) a clean checkpoint pause/resume or (b) accept that the bias will start aggressively correcting 122k+ steps of drift. Recommendation: leave the current run as-is, enable at the *next* scale-up so we start from a balanced state. A diagnostic-only mode — log the stats every step but keep `rate=0` — could quantify the current drift without changing training dynamics; adding that is a small follow-up if we want it.

### Follow-ups (not in scope for this branch)

- Diagnostic-only mode so we can measure imbalance on the current run without updating the bias.
- Sensitivity sweep on `rate` (1e-4, 1e-3, 1e-2) at small scale to pick a safe value for the next run.
