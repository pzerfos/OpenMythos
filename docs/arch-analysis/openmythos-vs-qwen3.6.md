# Architecture Comparison: OpenMythos 1B vs Qwen3.6-35B-A3B

**Date:** 2026-04-24
**Purpose:** Compare engineering trade-offs between the OpenMythos Recurrent-Depth Transformer and Qwen3.6's hybrid DeltaNet MoE architecture.

**Sources:**
- OpenMythos: `open_mythos/main.py`, `open_mythos/variants.py` (mythos_1b config), `training/1b_poc_fineweb.py` (runtime overrides)
- Qwen3.6: `config.json` from [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), Qwen3.5-Omni Technical Report ([arXiv:2604.15804](https://arxiv.org/abs/2604.15804))

**Note:** Qwen3.6-35B-A3B has an identical architecture to Qwen3.5-35B-A3B (confirmed via [HF discussion](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/discussions/12)). The 3.5 -> 3.6 bump is a training/data improvement, not an architecture change.

---

## 1. Side-by-Side Specs

All OpenMythos values verified from source. The "as trained" column reflects runtime overrides in `training/1b_poc_fineweb.py` (vocab_size from tokenizer, max_seq_len=2048).

| | **OpenMythos 1B** (as trained) | **Qwen3.6-35B-A3B** |
|---|---|---|
| **Total params** | ~1B | 35B |
| **Active params/token** | ~352M (top-4 of 64 routed + 2 shared experts) | ~3B (top-8 of 256 routed + 1 shared expert) |
| **Hidden dim** | 2048 | 2048 |
| **Layers** | 2 prelude + 1 recurrent (x16 loops) + 2 coda | 40 (30 DeltaNet + 10 full attention) |
| **Effective depth** | ~20 passes (16 loops + 4 dense) | 40 |
| **Attention type** | MLA (DeepSeek-V2 style) | Hybrid: 75% Gated DeltaNet + 25% GQA |
| **Attention heads** | 16 Q heads; MLA uses compressed latents | 16 Q / 2 KV (full attn); 16 QK / 32 V (DeltaNet) |
| **MLA latent dims** | kv_lora_rank=256, q_lora_rank=512, qk_rope=32, qk_nope=64, v=64 | N/A (not MLA) |
| **Head dim** | 128 (dim/n_heads for GQA); 32+64=96 (MLA QK) | 256 (full attn); 128 (DeltaNet) |
| **FFN (recurrent block)** | MoE: 64 experts + 2 shared, top-4, expert_dim=2048, SwiGLU | MoE: 256 experts + 1 shared, top-8, expert_dim=512, SiLU |
| **FFN (prelude/coda)** | Dense SwiGLU, inner_dim=2048*4/3=2730 | All layers use MoE (no dense FFN layers) |
| **Activation** | SwiGLU: `silu(gate(x)) * up(x)` | SiLU |
| **Norm** | RMSNorm (eps=1e-6) | RMSNorm (eps=1e-6) |
| **Vocab** | ~200K (gpt-oss-20b tokenizer, overrides config default of 32K) | 248K (byte-level BPE, padded to 248,320) |
| **Training seq_len** | 2048 (overrides config default of 4096) | N/A |
| **Max seq_len (config)** | 4096 | 262,144 (YaRN to 1M) |
| **RoPE theta** | 500K | 10M |
| **RoPE type** | Standard complex phasor, full rotary | mRoPE (interleaved, sections=[11,11,10], partial_rotary=0.25) |
| **Depth adaptation** | LoRA rank=8 per loop + sinusoidal loop embedding (dim//8 channels) | N/A (unique weights per layer) |
| **State injection** | LTI: `h = A*h + B*e + transformer_out`, spectral radius < 1 by ZOH construction | N/A |
| **Adaptive compute** | ACT halting (threshold=0.99) | None (fixed cost per token) |
| **Embedding tie** | Yes (LM head = embed weights) | No |
| **Multimodal** | No | Yes (vision encoder: 27-layer ViT, 1152 hidden, patch=16) |
| **Multi-token prediction** | No | Yes (1 MTP layer) |

## 2. Engineering Trade-Offs

### 2.1 Depth: Recurrence (Weight-Shared) vs Stacking (Unique Weights)

This is the fundamental architectural difference.

OpenMythos has 1 transformer block looped 16 times -- those 16 iterations share the same ~70M-param block, differentiated only by rank-8 LoRA adapters and sinusoidal loop embeddings. Qwen3.6 has 40 unique layers, each with independent parameters.

**OpenMythos advantages:**
- Massively parameter-efficient: 16x depth from 1 block's parameters
- Theoretical test-time compute scaling (more loops = more thinking)
- LTI injection guarantees stable recurrence regardless of depth

**OpenMythos disadvantages:**
- Weight sharing limits per-layer representational capacity; LoRA rank 8 adds minimal specialization
- ACT depth-binding problem (pzerfos/OpenMythos#5, confirmed by upstream ablations in [kyegomez/OpenMythos#28](https://github.com/kyegomez/OpenMythos/issues/28)): depth extrapolation doesn't work in practice with ACT enabled
- FSDP creates collective-ordering traps when ACT causes different ranks to exit at different loop iterations (the deadlock bug fixed in commit `6c5659c`)
- Pipeline parallelism is awkward: the recurrent block is a single stage that can't split across pipeline stages

**Qwen3.6 advantages:**
- Full independent capacity per layer; no weight-sharing bottleneck
- No data-dependent control flow: straightforward FSDP and pipeline parallelism
- No collective ordering issues

**Qwen3.6 disadvantages:**
- ~35x more total parameters for ~2x the effective depth
- Fixed compute per token: no adaptive computation, every token pays the same cost regardless of difficulty

### 2.2 Attention: MLA vs Gated DeltaNet Hybrid

**OpenMythos MLA** (DeepSeek-V2 style):
- Compresses KV into low-rank latents (kv_lora_rank=256 for 1B config)
- Cache grows with sequence length but is ~8-10x smaller than standard KV cache
- Reconstructs full K/V from compressed latents on demand
- All layers use the same attention mechanism

**Qwen3.6 hybrid** (layout: `3x DeltaNet + 1x full attention`, repeated 10x):
- 30 Gated DeltaNet layers: linear attention with O(1) recurrent state per layer (no KV cache)
- 10 full GQA layers: standard attention with KV cache (16 Q heads / 2 KV heads, head_dim=256)
- Only 25% of layers need a KV cache at all
- DeltaNet uses `linear_conv_kernel_dim=4` for local context

**Trade-off:** Qwen3.6 is a generation ahead for long-context inference memory. DeltaNet layers have constant memory regardless of sequence length. MLA still has a cache that grows linearly with sequence length, just compressed. At 262K context, Qwen3.6's hybrid approach is dramatically more memory-efficient. However, MLA preserves full quadratic attention quality at every layer, while DeltaNet makes a quality trade-off for efficiency.

### 2.3 MoE: Fewer Large Experts vs Many Small Experts

**OpenMythos:** 64 experts x expert_dim=2048, top-4 routing, 2 shared experts. Each expert is a full SwiGLU FFN (gate + up + down projections, 3 x 2048 x 2048 = ~25M params/expert). Fewer, larger experts = more capacity per expert but coarser routing granularity.

**Qwen3.6:** 256 experts x expert_dim=512, top-8 routing, 1 shared expert. Much smaller experts (~1.6M params/expert), more of them, more activated per token. This is the DeepSeek-V3 "fine-grained MoE" philosophy taken further.

**Trade-off:** Qwen3.6's fine-grained design is better for expert utilization and load balancing. OpenMythos's large-expert design is simpler but more prone to expert collapse, especially since our `router_bias` load balancing is not operational (pzerfos/OpenMythos#3).

### 2.4 LTI State Injection (Unique to OpenMythos)

The recurrence `h = A*h + B*e + transformer_out` with guaranteed spectral radius < 1 (via ZOH discretization of learned `log_A` and `log_dt`) is unique to OpenMythos. It prevents hidden state drift/explosion across loop iterations and keeps the original Prelude encoding `e` alive at every step. Qwen3.6 has no equivalent mechanism since its layers don't share weights and don't loop.

### 2.5 Context Length: 2K (Training) / 4K (Config) vs 262K

Partly a maturity gap (OpenMythos could extend RoPE theta and train longer), partly architectural:
- Qwen3.6's mRoPE with YaRN scaling and partial rotary (only 25% of dims get RoPE) is designed for extreme context lengths
- OpenMythos applies full RoPE to all dims and doesn't have extension logic implemented (pzerfos/OpenMythos#3)
- Qwen3.6's DeltaNet layers make long context computationally feasible; OpenMythos's full-attention MLA at every loop iteration would be expensive at 262K tokens

### 2.6 Vocabulary: ~200K vs 248K

Comparable. Qwen3.6's 248K vocab uses byte-level BPE with expanded multilingual coverage (10-60% encoding efficiency improvement over Qwen3's 150K vocab per the Qwen3.5-Omni report). OpenMythos uses the gpt-oss-20b tokenizer (~200K tokens). Both have large embedding tables.

### 2.7 Practical Deployment

| | **OpenMythos** | **Qwen3.6** |
|---|---|---|
| FSDP complexity | High: recurrent loop + ACT creates collective traps | Standard: no data-dependent control flow |
| Pipeline parallelism | Awkward: recurrent block is a single stage | Natural: 40 layers split cleanly |
| KV cache (inference) | Compressed (MLA) but grows with seq_len | Near-constant for 75% of layers (DeltaNet) |
| Adaptive compute | ACT (works for fixed depth; broken for depth extrapolation) | None: fixed cost/token |
| Test-time scaling | Theoretical: increase `n_loops` | Not possible without retraining |
| Multimodal | Text only | Text + vision (image/video) |
| Serving frameworks | Custom only (no standard support) | vLLM, SGLang, TensorRT-LLM, etc. |
| torch.compile | Limited (FSDP1, pzerfos/OpenMythos#6) | Supported |

## 3. Summary

Qwen3.6 is a production-oriented design that trades parameter efficiency for engineering simplicity and proven scalability. OpenMythos explores a more exotic point in the design space -- recurrent depth + ACT + MoE -- that could be more compute-efficient *if* the ACT depth-binding problem is solved, but currently has significant systems-level friction (FSDP deadlocks, no depth extrapolation, harder to parallelize).

The most notable architectural gaps to study for future OpenMythos iterations:
1. **Gated DeltaNet hybrid attention** -- constant-memory inference for most layers is a major advantage for long-context serving
2. **Fine-grained MoE** (256 small experts, top-8) -- better load balancing than 64 large experts, top-4
3. **Partial rotary + mRoPE** -- enables much longer context with less positional encoding overhead
4. **Multi-token prediction** -- training signal density improvement
