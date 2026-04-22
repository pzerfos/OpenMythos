# Looped Transformer Comparison: LoopFormer vs OpenMythos

## 1. High-Level Architecture

| Aspect | LoopFormer | OpenMythos (RDT) |
|--------|-----------|------------------|
| **Paper** | "Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation" (ICLR 2026) | Reconstruction of Recurrent-Depth Transformer (RDT) |
| **Pipeline** | Single shared block, looped T times | **Three-stage**: Prelude (run once) -> Recurrent Block (looped T times) -> Coda (run once) |
| **Loop target** | All transformer blocks are shared & looped | Only the middle recurrent block loops; Prelude/Coda are standard non-looped blocks |
| **FFN type** | Dense SwiGLU (2.5x expansion) | **MoE** in recurrent block (64-512 routed experts); dense SwiGLU in Prelude/Coda |
| **Attention** | Standard MHA with Flash Attention | Swappable: **GQA** or **MLA** (DeepSeek-V2 style) |
| **Scale** | ~100M params (NanoGPT-class research) | 1B to **1T** param variants defined |

## 2. Core Looping Mechanism

### LoopFormer: Continuous-Time Shortcut Modulation

- **3 blocks** looped **8 times** = 24 effective layers
- Timestep conditioning via **adaLN** (adaptive LayerNorm): each iteration receives a continuous time signal `t` and time-delta `dt`
- Uses two `TimestepEmbedder` modules (sinusoidal, diffusion-model style): one for accumulated time, one for step size
- adaLN produces 4 modulation vectors per block: `gate_msa`, `gate_mlp`, `scale_msa`, `scale_mlp`
- **Zero-initialized** modulation ensures identity behavior at the start
- Trajectories are **continuous** and **variable-length**: the model is trained on both uniform `[1/8]*8` and randomly sampled short trajectories

### OpenMythos: Recurrent Depth with LTI Stability

- **1 block** looped **16 times** (default), with Prelude/Coda stages flanking
- Loop-index embedding via **sinusoidal positional encoding** (theta=10000) injected into hidden states
- **Input re-injection**: the frozen Prelude output `e` is re-injected at every iteration to prevent drift
- **LTI-stable recurrence**: `h_{t+1} = A*h_t + B*e + TransformerBlock(h_t, e)` where A has spectral radius < 1 by construction (ZOH discretization)
- **Depth-wise LoRA**: per-loop scale vectors modulate block output differently at each depth

## 3. Halting & Convergence

| Mechanism | LoopFormer | OpenMythos |
|-----------|-----------|------------|
| **Adaptive halting** | **None** - fixed trajectory always executed | **ACT** (Adaptive Computation Time) with per-position early exit |
| **Halting threshold** | N/A | Cumulative probability >= 0.99 |
| **Remainder trick** | N/A | Yes - ensures probability mass sums to exactly 1.0 |
| **Stability guarantee** | None (relies on training) | **Mathematical**: LTI spectral radius < 1 via ZOH discretization |
| **Depth extrapolation** | Not explicitly supported | **Built-in**: LoRA scales clamp to last known value for unseen depths |

This is a **fundamental design divergence**: LoopFormer trusts the model to learn stable iteration through training, while OpenMythos enforces stability architecturally.

## 4. Training Methodology

| Parameter | LoopFormer | OpenMythos |
|-----------|-----------|------------|
| **Data** | Pile (dedup), OpenWebText, FineWeb-Edu (binary memmap) | FineWeb-Edu (HuggingFace streaming) |
| **Sequence length** | 1024 | 2048 |
| **Batch tokens/step** | ~4M (12 * 1024 * 40 * 8 GPUs) | Variable (micro_batch=4, grad_accum=256/world_size) |
| **Optimizer** | AdamW (0.9, 0.95) | AdamW (0.9, 0.95), fused |
| **LR** | 6e-4 peak | 3e-4 peak |
| **LR schedule** | Linear warmup (2000) + cosine decay | Linear warmup (2000) + cosine decay |
| **Weight decay** | 0.2 | 0.1 |
| **Grad clip** | 1.0 | 1.0 |
| **Precision** | AMP (bf16/fp16) | bf16 on H100/A100, fp16+GradScaler on older |
| **Distribution** | DDP | **FSDP** (Fully Sharded) |
| **Total training** | 50K steps | 30B tokens |

### LoopFormer's Multi-Objective Loss (Unique)

LoopFormer uses a **three-component loss** that is central to its design:

1. **Primary NTP loss** on uniform trajectory `[1/8]*8`
2. **Secondary NTP loss** (0.1x weight) on a **random short trajectory** (1-7 variable-size steps that sum to 1.0)
3. **Consistency loss** (0.1x weight): MSE between hidden states from uniform vs. short trajectory

This trains the model to produce consistent representations regardless of iteration count - the "elastic depth" property.

### OpenMythos' Training

Standard cross-entropy NTP loss only. The ACT mechanism adds a small ponder cost but no explicit multi-trajectory training.

## 5. Architectural Components

| Component | LoopFormer | OpenMythos |
|-----------|-----------|------------|
| **Normalization** | RMSNorm (`elementwise_affine=False` for adaLN) | RMSNorm (learned per-channel weight) |
| **Positional encoding** | **Absolute** learned embeddings (1024 positions) | **RoPE** (theta=500K-2M depending on scale) |
| **Activation** | GELU | SiLU/SwiGLU |
| **Weight tying** | Yes (embedding = LM head) | Yes (embedding = LM head) |
| **MoE** | No | Yes - fine-grained MoE with aux-loss-free load balancing |
| **KV cache** | Not implemented (training-focused) | Full dict-based KV cache for inference |
| **Dropout** | 0.0 (configurable) | 0.0 for pretraining (0.1 for fine-tuning) |

## 6. Loop Conditioning Comparison

This is the most important architectural contrast:

| Aspect | LoopFormer | OpenMythos |
|--------|-----------|------------|
| **How the loop knows "where" it is** | Continuous time embeddings (t + dt) fed through adaLN modulation | Sinusoidal loop-index embedding + depth-wise LoRA scales |
| **Modulation style** | **Multiplicative** gate + scale on RMSNorm outputs (4 vectors per block) | **Additive** loop embedding + **multiplicative** LoRA scaling |
| **Parameterization** | Single MLP per block -> 4 modulation vectors | Separate LoRA scale tensor per loop iteration |
| **Variable depth** | Native - continuous time can represent any trajectory | LoRA clamps to last scale for extrapolation |
| **Drift prevention** | Consistency loss during training | LTI recurrence + input re-injection (architectural) |

## 7. Design Philosophy

### LoopFormer

- **"Soft" approach**: learns stability through multi-objective training
- Inspired by **diffusion model** conditioning (adaLN from DiT)
- Elastic depth is a training-time property enforced by trajectory sampling
- Simpler architecture, research-scale (~100M params)
- NanoGPT fork - optimized for experimentation

### OpenMythos

- **"Hard" approach**: enforces stability mathematically via LTI guarantees
- Inspired by **control theory** (linear time-invariant systems, ZOH discretization)
- ACT provides runtime adaptivity - different tokens get different compute
- Production-oriented architecture with MoE, MLA, FSDP
- Scales from 1B to 1T with pre-defined variant configurations

## 8. Key Innovations Unique to Each

### LoopFormer Only

- **Shortcut modulation** via adaLN (borrowed from diffusion models)
- **Multi-trajectory training** with consistency loss
- **Continuous time representation** allowing arbitrary step sizes

### OpenMythos Only

- **Three-stage pipeline** (Prelude/Recurrent/Coda) separating encoding, reasoning, and decoding
- **LTI-stable recurrence** with mathematical convergence guarantee
- **Adaptive Computation Time** with per-position early exit
- **MoE in recurrent block only** (dense elsewhere)
- **MLA attention** option for extreme KV cache compression
- **Depth extrapolation** via LoRA clamping
- **Input re-injection** from frozen encoder output

## 9. Strengths & Tradeoffs

| | LoopFormer | OpenMythos |
|---|-----------|------------|
| **Strength** | Elegant simplicity; elastic depth; easy to experiment with | Mathematical stability; runtime adaptivity; production-ready scaling |
| **Tradeoff** | No convergence guarantee; fixed compute per token | More complex architecture; more hyperparameters to tune |
| **Best for** | Research on looped transformer properties | Building production looped-transformer systems at scale |
| **Risk** | Hidden state could diverge with untested trajectories | ACT overhead; MoE routing complexity |

## 10. Summary

Both implement weight-shared looped transformers but from very different angles. LoopFormer takes a **learned, soft** approach to loop conditioning (diffusion-style adaLN + consistency training), while OpenMythos takes a **principled, hard** approach (control-theory stability + ACT halting). LoopFormer is simpler and more research-friendly; OpenMythos is more architecturally sophisticated and production-oriented.

## 11. Quick Reference

**Shared foundation** -- Both loop a shared transformer block multiple times to create effective depth from fewer parameters.

**Key differences at a glance:**

| Axis | LoopFormer | OpenMythos |
|------|-----------|------------|
| Pipeline | Single shared block, looped uniformly | Three-stage: Prelude / Recurrent Block / Coda |
| Loop conditioning | Continuous-time adaLN (diffusion-style) with 4 modulation vectors per block | Sinusoidal loop-index embedding + depth-wise LoRA |
| Stability | None (learned via consistency loss) | Mathematical: LTI diagonal injection, spectral radius < 1 by construction |
| Adaptive halting | None (fixed trajectory) | ACT with per-position early exit and remainder trick |
| FFN | Dense SwiGLU | MoE in recurrent block (64-512 experts); dense in Prelude/Coda |
| Attention | Standard MHA + Flash Attention | MLA (default, 10-20x cache compression) or GQA |
| Training loss | Three-component: primary NTP + short-trajectory NTP + consistency MSE | Standard cross-entropy |
| Scale | ~100M (NanoGPT-class) | 1B-1T (7 variants) |
| Depth extrapolation | Not explicitly supported | Built-in via LoRA clamping |
| Philosophy | Learned, soft (diffusion-model inspired) | Principled, hard (control-theory inspired) |

**Bottom line**: LoopFormer is a lightweight research prototype that learns stable looping through multi-objective training (elastic depth via adaLN + consistency loss). OpenMythos is a production architecture that enforces stability mathematically (LTI guarantees) and adds runtime adaptivity (ACT), efficiency (MoE/MLA), and scaling (up to 1T parameters).
