# OpenMythos Code Review — 2026-04-23

Thorough review of the full model implementation (`open_mythos/main.py`), training scripts, and supporting code. Performed during the 1B PoC training bring-up on BlueVela.

## Context

While debugging FSDP mixed precision dtype mismatches during 4-GPU training, we commissioned a full code review covering: FSDP correctness, numerical stability, recurrent block logic, KV cache, MoE routing, and general code quality. Several dtype issues had already been fixed prior to this review (RMSNorm, softmax, nn.Linear entry point casts). The review confirmed those fixes are correct and identified additional issues.

## FSDP Dtype Fixes Already Applied

These were fixed during the training bring-up and confirmed correct by the review:

1. **RMSNorm dtype preservation** (`main.py:108-110`): Computes in float32 for stability, casts output back to input dtype. Without this, downstream `nn.Linear` layers receive float32 input against bfloat16 weights.

2. **Softmax dtype cast** (`main.py:245, 389`): `F.softmax` upcasts to float32. Added `.to(v.dtype)` before `torch.matmul(attn, v)` in both `GQAttention` and `MLAttention`.

3. **nn.Linear entry point casts** (`main.py`): Added `x = x.to(weight.dtype)` at the entry of `GQAttention` (line 221), `MLAttention` (line 344), `Expert` (line 429), `MoEFFN` (line 482), `LoRAAdapter` (line 595), and `ACTHalting` (line 758).

---

## Important Issues To Fix

### 1. ACT halting: no remainder for positions that exhaust all loop iterations

**File:** `open_mythos/main.py`, `RecurrentBlock.forward`, after the loop (~line 868)
**Severity:** Important — affects training quality for hard tokens

When a position does not halt within `n_loops` iterations (cumulative halting probability never reaches `act_threshold`), the accumulated weights for that position sum to less than 1.0. The output `h_out` for that position is under-weighted compared to positions that halted normally (whose weights sum to exactly 1.0 via the remainder trick).

**Current behavior:** The remainder trick (`weight = 1 - cumulative_p`) only activates when `cumulative_p + p >= threshold`. If the loop ends first, no remainder is applied.

**Fix:** After the loop exits, assign a final remainder weight for positions that never halted:

```python
# After the for loop ends:
not_halted = ~halted
if not_halted.any():
    final_remainder = (1.0 - cumulative_p).clamp(min=0) * not_halted.float()
    h_out = h_out + final_remainder.unsqueeze(-1) * h
```

### 2. MoE score renormalization: division by zero risk

**File:** `open_mythos/main.py:493`
**Severity:** Important — can produce NaN during early training with bfloat16

```python
topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)  # renorm
```

If all top-k softmax scores underflow to zero in bfloat16 (possible with random weights early in training), this divides by zero.

**Fix:**
```python
topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True).clamp(min=1e-9)
```

### 3. `router_bias` is never updated — load balancing is inactive

**File:** `open_mythos/main.py:461` (definition), `training/1b_poc_fineweb.py` and `training/3b_fine_web_edu.py` (usage)
**Severity:** Important — affects expert utilization over long training runs

The `router_bias` buffer is initialized to zeros and the docstring says it should be "adjusted externally during training," but neither training script updates it. The bias term on line 491 (`logits + self.router_bias`) has no effect, meaning the aux-loss-free load balancing described in the MoEFFN docstring is not operational.

**Options:**
- Implement periodic bias updates in the training loop (track per-expert token counts, increment bias for underutilized experts) — this is the DeepSeek-V3 approach
- Or document that this is intentionally disabled for the PoC

### 4. MoE dispatch loop is O(topk x n_experts) — throughput bottleneck

**File:** `open_mythos/main.py:496-504`
**Severity:** Important for throughput, not a correctness bug

```python
out = torch.zeros_like(flat)
for i in range(self.topk):           # 4 iterations
    expert_ids = topk_idx[:, i]
    token_scores = topk_scores[:, i].unsqueeze(-1)
    for eid in range(self.n_experts):  # 64 iterations
        mask = expert_ids == eid
        if not mask.any():
            continue
        out[mask] += token_scores[mask] * self.routed_experts[eid](flat[mask])
```

256 Python-level iterations per forward pass (4 x 64). Each iteration does boolean masking, indexing, expert forward, and scatter-add.

**Fix:** Replace with a grouped dispatch pattern — gather all tokens per expert into a single batch, run the expert once, scatter results back. This is standard in production MoE implementations (e.g., Megablocks, Tutel).

### 5. `__init__.py` exports nonexistent symbols

**File:** `open_mythos/__init__.py:52`
**Severity:** Important — broken public API

`__all__` includes `load_tokenizer` and `get_vocab_size` which don't exist anywhere in the codebase. `from open_mythos import load_tokenizer` raises `ImportError`.

**Fix:** Remove them from `__all__`, or implement them.

### 6. `fused=True` on AdamW crashes on CPU

**File:** `training/1b_poc_fineweb.py:497`, `training/3b_fine_web_edu.py:438`
**Severity:** Important — crashes CPU-only runs

```python
optimizer = torch.optim.AdamW(
    model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95), fused=True
)
```

`fused=True` requires CUDA tensors. On CPU, this raises an error.

**Fix:**
```python
use_fused = "cuda" in device
optimizer = torch.optim.AdamW(
    model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95), fused=use_fused
)
```

### 7. LoRAAdapter.self.B dtype depends on FSDP wrap policy

**File:** `open_mythos/main.py:576`
**Severity:** Important — fragile under config changes

`self.B = nn.Parameter(torch.randn(rank, dim) * 0.02)` is a raw `nn.Parameter`, not an `nn.Linear` weight. It gets cast to bfloat16 only because `LoRAAdapter` is inside `RecurrentBlock` which is in the FSDP wrap policy. If someone removes `RecurrentBlock` from the policy, `self.B` stays float32 and `down @ self.B` produces a dtype mismatch.

**Fix:** Add defensive cast: `return down @ self.B.to(down.dtype)`

---

## Minor Issues

### 8. `loop_index_embedding` computes trig in bfloat16

**File:** `open_mythos/main.py:539-543`

When `h.dtype` is bfloat16, `theta ** (arange / loop_dim)` and `sin()`/`cos()` are computed in bfloat16 (~3 decimal digits precision). This loses precision in positional encodings.

**Fix:** Compute in float32, cast result back to `h.dtype`.

### 9. Causal mask is always float32

**File:** `open_mythos/main.py:958-959`

`torch.full(..., float("-inf"))` creates a float32 mask. Adding it to bfloat16 attention scores promotes the entire attention computation to float32, increasing memory usage.

**Fix:** Create mask with model dtype, or cast at the call site.

### 10. RoPE lazy extension not implemented

CLAUDE.md says "Frequencies are lazily extended when sequence length exceeds the precomputed range," but the code precomputes for `cfg.max_seq_len` and has no extension logic. Sequences longer than `max_seq_len` will silently produce wrong results.

### 11. `amp_ctx` variable flow is confusing

**File:** `training/1b_poc_fineweb.py:473-481`, same in `3b_fine_web_edu.py`

`amp_ctx` is only assigned in the `else` branch but used unconditionally on line 481. The `# type: ignore` comment masks a real readability issue.

---

## Confirmed Correct

- FSDP dtype casts at all nn.Linear entry points
- RMSNorm dtype preservation
- Softmax dtype cast in both attention classes
- KV cache accumulation and start_pos handling
- ACT weight accumulation for positions that DO halt (remainder trick works correctly)
- RoPE frequencies during cached decode (same position for all loop iterations — correct)
- `.detach()` on cached KV (correct — cache is only used during inference under no_grad)
- `e = x` alias in OpenMythos.forward (safe — h is rebound, not mutated in-place)
- Per-loop-iteration KV cache keys in RecurrentBlock (correct design — each depth has its own attention space)
