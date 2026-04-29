# Looped Transformer Comparison: Parcae vs OpenMythos

## 1. High-Level Architecture

| Aspect | Parcae | OpenMythos (RDT) |
|--------|--------|------------------|
| **Paper** | "Parcae: Scaling Laws For Stable Looped Language Models" (Prairie et al., ICLR 2026; arXiv 2604.12946) | Reconstruction of Recurrent-Depth Transformer (RDT) |
| **Pipeline** | **Three-stage**: Prelude (2-8 layers) -> Core Block (2-8 shared layers, looped) -> Coda (2-8 layers) | **Three-stage**: Prelude (2-4 layers) -> Recurrent Block (1 block, looped) -> Coda (2-4 layers) |
| **Loop target** | Multiple shared core layers (2-8) looped as a group | Single transformer block looped T times |
| **FFN type** | Dense: BaseMLP (ReLU^2) or GatedMLP | **MoE** in recurrent block (64-512 routed experts); dense SwiGLU in Prelude/Coda |
| **Attention** | MHA with **GQA** support + optional **value embedding gates** | Swappable: **MLA** (default, DeepSeek-V2 style) or **GQA** |
| **Scale** | 140M to **1.3B** (4 variants, trained) | 1B to **1T** (7 variants defined; **only the 1B variant has been trained to date**) |
| **State space** | Separate `recurrent_embedding_dimension` (can differ from model dim) + explicit C output projection | Recurrent state shares model dim; no separate output projection |

Both projects share the Prelude/Core/Coda three-stage design and LTI-based stability guarantees, making them architecturally closer to each other than either is to LoopFormer (which loops all blocks uniformly with adaLN conditioning).

## 2. Core Looping Mechanism

### Parcae: Multi-Layer Core Block with Stochastic Depth

- **2-8 shared layers** looped **8 times** (default `mean_recurrence`) = up to 64 effective layers
- The core block operates in a **separate recurrent embedding dimension** that may differ from `n_embd`; an adapter projects input into this space and a learned C matrix projects back out
- Three iteration modes: `per-batch`, `per-sequence` (default), `per-token`
- **Forward-only + backprop split**: first `n` iterations run under `torch.no_grad()`, last `k` iterations accumulate gradients (default `mean_backprop_depth=4`)
- **DiagonalInjection adapter**: re-injects input embeddings into recurrent state via LTI dynamics at every iteration, _before_ the core transformer layers process the state
- Stochastic depth: iteration count sampled from Poisson or curriculum schedule each training step
- **Curriculum learning**: iteration count ramps up during training (linear or sqrt schedule over a configurable number of steps)

### OpenMythos: Single-Block Recurrence with ACT

- **1 block** looped **16 times** (default), with Prelude/Coda flanking
- Loop-index embedding via **sinusoidal positional encoding** (theta=10000) injected into first `dim//8` channels of hidden state
- **Input re-injection**: frozen Prelude output `e` added to hidden state at every iteration via both direct addition and LTI B term
- **LTI-stable recurrence**: `h_{t+1} = A*h_t + B*e + TransformerBlock(h_t, e)`
- **Depth-wise LoRA**: per-loop scale vectors modulate block output at each depth (shared down/up projections + per-loop scale embeddings)

### Key Structural Difference

Parcae loops **multiple layers** (2-8) as the core block, meaning each "iteration" is a pass through several transformer layers. OpenMythos loops a **single block**, so each iteration is one attention + FFN pass. Parcae compensates with fewer mean iterations (8 vs 16) but more layers per iteration.

## 3. Halting & Convergence

| Mechanism | Parcae | OpenMythos |
|-----------|--------|------------|
| **Adaptive halting** | **None** at inference; stochastic depth sampling at training | **ACT** with per-position early exit |
| **Halting threshold** | N/A | Cumulative probability >= 0.99 |
| **Remainder trick** | N/A | Yes - ensures probability mass sums to exactly 1.0 |
| **Stability guarantee** | **Mathematical**: DiagonalInjection with `exp(-dt*A)` decay, spectral radius <= 1 | **Mathematical**: LTI spectral radius < 1 via ZOH discretization |
| **Depth extrapolation** | Paper demonstrates test-time depth extrapolation beyond training depth | **Built-in**: LoRA scales clamp to last known value for unseen depths |
| **Stochastic depth** | Yes - 6+ Poisson/curriculum sampling schemes | **Yes** (default on main since PR #7): uniform `n_loops ~ U[1, 32]` per optimizer step, broadcast from rank 0 for FSDP collective-ordering safety |
| **Iteration modes** | per-batch, per-sequence, per-token | Per-batch (all sequences share `n_loops` per step); ACT retained as opt-in alternative via `recurrent_mode="act"` |
| **Truncated BPTT** | Yes - only last `k` iterations get gradients (saves memory) | No - full gradient through all iterations |

The divergence here has narrowed since PR #7. Both projects now use stochastic depth during training — Parcae with rich Poisson/curriculum sampling plus truncated BPTT, OpenMythos with uniform per-step sampling and full BPTT. At inference, Parcae runs fixed iterations. OpenMythos runs either fixed iterations (`bypass_act=True`, default for stochastic-depth-trained checkpoints) or adapts per position via ACT (when the checkpoint was trained under `recurrent_mode="act"`). Checkpoints are cross-mode compatible.

## 4. Stability Mechanism Comparison

Both projects enforce stability mathematically through diagonal LTI systems -- this is a core shared foundation. The Parcae paper provides the theoretical framework that both implementations build on.

### Parcae: DiagonalInjection (`parcae_lm/modules/injection.py`)

```
x_{t+1} = exp(-dt * A) * x_t + dt * B @ e
```
- `A_log` parameter stores `log(A)`, initialized to **zero** (so `A=1`, decay starts at `exp(-dt)`)
- `dt` via **softplus** of a bias parameter, initialized for a target decay of `sqrt(1/5)`
- `B` matrix initialized as **identity** (or scaled orthogonal)
- Decay factor: `decay = exp(-dt * A)` where all elements are in (0, 1)
- Monitoring: `get_spectral_norm()` returns max decay, `get_contraction_factor()` returns mean decay
- All SSM parameters excluded from weight decay (`_no_weight_decay = True`)

### OpenMythos: LTIInjection (`open_mythos/main.py:656-714`)

```
h_{t+1} = A * h_t + B * e + TransformerBlock(h_t, e)
```
- Continuous A forced negative via `-exp(log_A)`, then discretized via ZOH
- `A_discrete = exp(dt * A_continuous)` with clamping at +/-20 for numerical safety
- `B` initialized to `0.1 * ones(dim)` (scalar per dimension)
- All diagonal elements of `A_discrete` in (0, 1) regardless of gradient magnitude

### Comparison

| Aspect | Parcae | OpenMythos |
|--------|--------|------------|
| **A parameterization** | `A = exp(A_log)`, always positive | `-exp(log_A)`, always negative continuous |
| **Discretization** | `exp(-dt * A)` (direct) | ZOH: `exp(dt * A_cont)` with clamping |
| **B structure** | Matrix `(state_dim, input_dim)`, identity-initialized | Vector `(dim,)`, initialized to 0.1 |
| **dt activation** | Softplus | exp (via log_dt) |
| **Initial behavior** | Near-identity (A_log=0, so A=1) | Near-identity (by construction) |
| **Transformer output** | Applied _after_ injection (state -> adapter -> core layers) | Added to injection output: `A*h + B*e + TransformerBlock(h, e)` |
| **Numerical safety** | Implicit (softplus never negative) | Explicit clamping at +/-20 |

A subtle but important ordering difference: in Parcae, injection happens **before** the core transformer layers (`adapter(x, e)` then `core_block(x)`). In OpenMythos, the transformer output is **added to** the LTI update (`A*h + B*e + ℛ(h, e)`).

## 5. Loop Conditioning Comparison

| Aspect | Parcae | OpenMythos |
|--------|--------|------------|
| **How the loop knows "where" it is** | Step index + total steps passed as tensors (used for KV cache indexing, not for modulation) | Sinusoidal loop-index embedding (theta=10000) in first `dim//8` channels |
| **Explicit loop embedding** | **No** - no per-iteration learned or sinusoidal embedding | **Yes** - sinusoidal + depth-wise LoRA scales |
| **Per-iteration adaptation** | DiagonalInjection applies same transform at every step | LoRA scale vectors differ per loop iteration |
| **LoRA mechanism** | No | Shared down/up projections + per-loop scale embeddings |
| **Input re-injection** | DiagonalInjection: `dt * B @ e` every step | Direct addition `h + e` plus LTI `B * e` term |
| **Drift prevention** | LTI decay + fixed injection | LTI decay + re-injection + LoRA stabilization |
| **Variable depth** | Stochastic sampling varies depth per training step | LoRA clamps to last scale for extrapolation; ACT halts early |

This is a fundamental design choice: Parcae relies on the **dynamical system itself** to differentiate iterations (through the LTI state evolution), while OpenMythos **explicitly tells** the shared block which iteration it's on via sinusoidal embeddings and per-loop LoRA.

## 6. Training Methodology

| Parameter | Parcae | OpenMythos |
|-----------|--------|------------|
| **Data** | Parquet datasets; FineWeb-Edu, Huginn | FineWeb-Edu (HuggingFace streaming) |
| **Sequence length** | 2048 | 2048 |
| **Optimizer** | **MuonAdamW** (Muon momentum + AdamW, per-parameter scaling) | AdamW (fused) |
| **LR** | 8e-3 peak (MuonAdamW regime) | 3e-4 peak |
| **Betas** | (0.8, 0.95) | (0.9, 0.95) |
| **LR schedule** | Warmup + cosine/linear/trapezoid decay | Linear warmup (2000) + cosine decay |
| **Weight decay** | 0.2 (linearly decays to 0 for Muon params) | 0.1 |
| **Grad clip** | 1.0 | 1.0 |
| **Precision** | AMP (bf16/fp16) | bf16 on H100/A100, fp16+GradScaler on older |
| **Distribution** | Lightning Fabric (DDP or FSDP, multiple sharding strategies) | PyTorch FSDP (FULL_SHARD) |
| **Total training** | 12B-100B tokens (scale-dependent) | 30B tokens |
| **Vocab size** | 32,768 | openai/gpt-oss-20b tokenizer |
| **Token packing** | Best-fit packing (100% utilization) | Standard batching |
| **Init strategy** | "scaled-zero" with orthogonal init | Standard init |

### Parcae's Unique Training Features

1. **MuonAdamW optimizer**: Per-parameter and Muon (momentum-based) updates enabling higher learning rates
2. **Stochastic depth with 6+ sampling schemes**: `fixed`, `poisson-truncated-full` (default), `poisson-unbounded`, `poisson-fill`, `poisson-full`, `poisson-bounded`, plus curriculum variants
3. **Curriculum learning**: Forward and/or backward iteration counts ramp up over configurable schedule (linear or sqrt)
4. **Truncated BPTT**: Only last `mean_backprop_depth` iterations accumulate gradients; earlier iterations run with `no_grad()`
5. **Best-fit token packing**: 100% token utilization via packing
6. **Selective torch.compile**: Blocks compiled pre/post DDP
7. **Stateful checkpointing**: Exact batch determinism with step-seeded RNG for sampling
8. **Per-sequence depth sampling**: Each sequence independently samples its iteration count, reducing loss spikes vs per-batch sampling

### OpenMythos' Training

Standard cross-entropy NTP loss. Two switchable recipes controlled by the `recurrent_mode` local in the training script:

- `"stochastic_depth"` (default on main since PR #7): `n_loops` sampled uniformly from `[stochastic_depth_min, stochastic_depth_max] = [1, 32]` once per optimizer step on rank 0, then broadcast to all ranks to prevent FSDP collective-ordering deadlock. ACT is bypassed (`bypass_act=True`).
- `"act"`: original recipe — fixed `n_loops` with per-position ACT halting plus a small ponder cost.

Checkpoints are cross-mode compatible: `ACTHalting` weights are always present in the state_dict and simply receive no gradient under `bypass_act=True`. DeepSeek-V3 aux-loss-free MoE load balancing (`router_bias_update_rate`) is implemented (PR #8/#9) and currently live at `1e-3` on the in-flight 10B-token run.

## 7. Loss Function

| Aspect | Parcae | OpenMythos |
|--------|--------|------------|
| **Loss type** | Cross-entropy with optimizations | Standard cross-entropy |
| **Implementation** | 3 variants: `full-triton` (Triton-optimized), `cce` (cut_cross_entropy), `hhe`, or PyTorch | PyTorch `F.cross_entropy` |
| **Z-regularization** | Optional: `logits.logsumexp(-1).pow(2).mean()` | No |
| **Logit softcap** | Optional: `softcap * tanh(logits / softcap)` | No |
| **Logit scaling** | Configurable `logit_scale` from init (muP-style) | No |
| **Ignore index** | Configurable (default -100) | Standard |

## 8. Architectural Components

| Component | Parcae | OpenMythos |
|-----------|--------|------------|
| **Normalization** | RMSNorm (eps=1e-5) | RMSNorm (eps=1e-6) |
| **Positional encoding** | **RoPE** (theta=50,000) | **RoPE** (theta=500,000-2,000,000) |
| **Activation** | **ReLU^2** (default), GatedMLP available | SiLU/SwiGLU |
| **Weight tying** | Yes (embedding = LM head) | Yes (embedding = LM head) |
| **MoE** | No (dense throughout) | Yes - fine-grained MoE with aux-loss-free load balancing (recurrent block only) |
| **MLA** | No | Yes (default attention type, 10-20x KV cache compression) |
| **KV cache** | ParcaeDynamicCache (step-indexed for recurrent layers) | Dict-based + MLA latent compression |
| **QK normalization** | Yes (RMS) | Yes (within MLA latent norms) |
| **Value embeddings** | Yes - alternating layers get learned VE with sigmoid gates (range 0-2) | No |
| **Bias terms** | No | No |
| **Output projection C** | Yes - learned linear from recurrent state dim to model dim | No (state dim = model dim) |
| **Prelude norm** | Optional (after prelude, before injection) | No |
| **FFN expansion** | 4x n_embd | 4/3x for dense (Prelude/Coda); small expert_dim for MoE |

## 9. Scale & Variant Configurations

### Parcae (4 variants)

| Model | Params | Prelude | Core Layers | Coda | Dim | Heads | KV Heads | FFN Size | Recurrence | Backprop Depth | Context |
|-------|--------|---------|-------------|------|-----|-------|----------|----------|------------|----------------|---------|
| Parcae-140M | 140M | 2 | 2 | 2 | 768 | 6 | 6 | 3072 | 8 | 4 | 2048 |
| Parcae-370M | 370M | 4 | 4 | 4 | 1024 | 8 | 8 | 4096 | 8 | 4 | 2048 |
| Parcae-770M | 770M | 6 | 6 | 6 | 1280 | 10 | 10 | 5120 | 8 | 4 | 2048 |
| Parcae-1.3B | 1.3B | 8 | 8 | 8 | 1536 | 12 | 12 | 6144 | 8 | 4 | 2048 |

All use: `injection_type=diagonal`, `sampling_scheme=poisson-truncated-full`, `recurrent_iteration_method=per-sequence`, `mlp=BaseMLP(ReLU^2)`, `qk_norm=True`, `init_strategy=scaled-zero`, `init_orthogonal=True`, `tie_embeddings=True`

### OpenMythos (7 variants)

| Model | Params | Prelude | Coda | Dim | Heads | Experts | Expert Dim | Loop Iters | Context |
|-------|--------|---------|------|-----|-------|---------|-----------|------------|---------|
| mythos_1b | 1B | 2 | 2 | 2048 | 16 | 64 | 2048 | 16 | 4K |
| mythos_3b | 3B | 2 | 2 | 3072 | 24 | 64 | 4096 | 16 | 4K |
| mythos_10b | 10B | 3 | 3 | 4096 | 32 | 128 | 5632 | 24 | 8K |
| mythos_50b | 50B | 4 | 4 | 6144 | 48 | 256 | 9728 | 32 | 8K |
| mythos_100b | 100B | 4 | 4 | 8192 | 64 | 256 | 13568 | 32 | 1M |
| mythos_500b | 500B | 6 | 6 | 12288 | 96 | 512 | 23040 | 48 | 1M |
| mythos_1t | 1T | 6 | 6 | 16384 | 128 | 512 | 34560 | 64 | 1M |

All use: MLA attention, aux-loss-free MoE load balancing, ACT halting

### Scaling Differences

Parcae scales all three stages **symmetrically** (equal Prelude/Core/Coda layer counts) and keeps recurrence fixed at 8. OpenMythos keeps Prelude/Coda **thin** (2-6 layers) and **increases loop iterations** with scale (16 -> 64), reflecting different scaling philosophies.

## 10. Scaling Laws (Parcae Paper Contribution)

The Parcae paper's primary contribution is a rigorous scaling law analysis for looped transformers:

**Training-time power laws** (fixed parameters, varying data and recurrence):
```
mu_rec_optimal  proportional to  FLOP^0.40
D_optimal       proportional to  FLOP^0.78
```

**Test-time depth extrapolation law**:
```
L(T) = L_inf + Z * exp(-z * T)
```

**Unified law** bridging training and inference:
```
L_unified(T | mu_rec, D) = [Training floor] + Z * exp(-z * T * mu_rec^{-1})
```

Predictions achieve 0.85-1.31% average error on held-out model scales.

**Key result**: 770M Parcae matches or exceeds 1.3B standard Transformer on CORE benchmark, with 23-88% parameter efficiency gains.

OpenMythos does not provide scaling law analysis. Variants are defined up to 1T parameters, but only the 1B variant has been trained to date; scaling behavior beyond that is unverified.

## 11. Design Philosophy

### Parcae

- **Scaling law research platform**: Primary contribution is understanding how looped transformers scale
- **Dynamical systems framing**: Recasts looped models as LTI systems to derive stability conditions theoretically
- **Training-time adaptivity**: Stochastic depth and curriculum learning vary compute during training
- **Inference simplicity**: Fixed iteration count at inference (no ACT overhead)
- **Dense architecture**: No MoE complexity; all parameters always active
- **Novel optimizer**: MuonAdamW for potentially better optimization landscape
- **Multi-layer core**: Each iteration passes through multiple shared layers
- **Separate state dimension**: Core block can operate in a different dimension than the main model

### OpenMythos

- **Research architecture with extreme-scale ambitions**: variants defined from 1B to 1T; only the 1B variant has been trained to date
- **Single-block recurrence**: maximizes weight sharing with one looped block
- **Switchable adaptivity**: ACT per-position early exit OR uniform stochastic-depth sampling during training; runtime-selectable, checkpoints are cross-compatible
- **Efficiency targets**: MoE for sparse parameter scaling, MLA for 10–20× KV cache compression vs standard attention
- **Explicit depth conditioning**: LoRA + sinusoidal embeddings give fine-grained per-iteration control
- **Long context target**: RoPE theta = 500K–2M aims at long-context training at larger scales; not yet verified in practice
- Uses PyTorch FSDP1 today; FSDP2 migration is under evaluation (feasibility benchmark showed a 15% step-time reduction on the stochastic-depth recipe, but no final decision has been made)

## 12. Key Innovations Unique to Each

### Parcae Only

- **Multi-layer looped core block** (2-8 shared layers iterated as a group, not just 1 block)
- **Separate recurrent embedding dimension** with C output projection (state space can differ from model dim)
- **Rich stochastic depth sampling**: 6+ Poisson/curriculum/fixed schemes (vs OpenMythos's simpler uniform `U[1, 32]`)
- **Truncated BPTT** with configurable forward-only / backprop split
- **Per-sequence depth sampling** (each sequence independently samples its iteration count)
- **MuonAdamW optimizer** (momentum-based with per-parameter scaling)
- **Value embedding gates** (alternating layers get learned VE with sigmoid-gated blending)
- **Best-fit token packing** (100% utilization)
- **Triton-optimized cross-entropy** with z-regularization and logit softcapping
- **Scaling law analysis**: Training-time and test-time power laws for looped transformers
- **"Scaled-zero" initialization** with orthogonal init

### OpenMythos Only

- **Adaptive Computation Time** with per-position early exit and remainder trick (opt-in via `recurrent_mode="act"`, preserved alongside the stochastic-depth recipe)
- **MoE in recurrent block only** (dense elsewhere) with **live aux-loss-free load balancing** (DeepSeek-V3 Algorithm 1; PR #8/#9 wired up the update, enabled at `rate=1e-3` on the in-flight 10B run)
- **Multi-Latent Attention (MLA)** for 10-20x KV cache compression
- **Depth-wise LoRA** with per-loop scale embeddings
- **Sinusoidal loop-index embedding** in first `dim/8` channels
- **Depth extrapolation** via LoRA clamping to last known value
- **High RoPE theta** (500K-2M) designed for long-context support; untested at scale
- **Fine-grained MoE** with small expert dimensions for sparse FLOPs
- **Rank-0 broadcast of `n_loops`** for FSDP collective-ordering safety (analogous in purpose to Parcae's `lockstep_n`/`lockstep_k` seeded-RNG mechanism)

## 13. Shared Design Decisions

Both projects converge on several architectural choices, suggesting a common line of research:

| Shared Choice | Implementation |
|---------------|----------------|
| **Three-stage pipeline** | Prelude -> Looped Core -> Coda |
| **LTI stability** | Diagonal `exp(-dt*A)` decay guaranteeing spectral radius < 1 |
| **Input re-injection** | Encoded input injected at every loop iteration |
| **RMSNorm** | Pre-norm architecture throughout |
| **RoPE** | Rotary position embeddings for sequence position |
| **Weight tying** | Embedding = LM head |
| **No bias** | Bias-free linear layers |
| **AdamW family** | Optimizer with beta2=0.95 and gradient clipping at 1.0 |
| **Cosine decay** | Learning rate schedule with warmup |
| **FineWeb-Edu** | Training data source |
| **2048 seq length** | Comparable training sequence length |
| **FSDP-safe depth sampling** | Parcae: seeded `torch.Generator` with `lockstep_n`/`lockstep_k` ensuring ranks agree. OpenMythos: sample on rank 0 and broadcast. Both avoid NCCL collective-ordering deadlock. |
| **Stochastic depth during training** | Parcae: Poisson/curriculum schemes with truncated BPTT. OpenMythos: uniform `U[1, 32]` with full BPTT. Same principle, different sophistication. |

Both reference the Parcae paper's dynamical systems formulation of looped transformer stability.

## 14. Strengths & Tradeoffs

| | Parcae | OpenMythos |
|---|--------|------------|
| **Strength** | Rigorous scaling laws; stochastic depth for training robustness; simpler dense architecture; formal stability theory | Runtime adaptivity via ACT; extreme scaling (1T); MoE/MLA efficiency; long context (1M) |
| **Tradeoff** | No adaptive compute at inference; smaller scale (max 1.3B); shorter context (2048) | More complex architecture; MoE routing overhead; more hyperparameters to tune |
| **Best for** | Understanding how looped transformers scale; dense-model research | Researching scale-up recipes for sparse (MoE), adaptive-depth looped transformers; trained only at 1B to date so extreme-scale behavior is unverified |
| **Risk** | Fixed inference compute may waste FLOPs on easy tokens | ACT overhead; MoE load balancing complexity; MLA implementation complexity |

## 15. Summary

Parcae and OpenMythos share the same foundational architecture -- a three-stage pipeline (Prelude/Core/Coda) with LTI-stable diagonal injection guaranteeing spectral radius < 1. This makes them much closer to each other than either is to LoopFormer (which uses adaLN conditioning from diffusion models and has no stability guarantee).

Where they diverge is along three axes:

1. **Training-time vs runtime adaptivity**: Parcae invests in stochastic depth sampling, curriculum learning, and truncated BPTT for training flexibility while keeping inference simple (fixed iterations). OpenMythos runs fixed iterations during training but adapts at inference via ACT early exit.

2. **Dense vs sparse**: Parcae is fully dense (all parameters active, value embeddings for extra capacity). OpenMythos uses MoE for parameter efficiency and MLA for KV cache compression, enabling much larger scales.

3. **Theory vs production**: Parcae's primary contribution is scaling laws and the dynamical systems stability framework. OpenMythos's contribution is a production architecture that applies these principles at 1B-1T scale with efficiency mechanisms.

Parcae is the theoretical foundation and scaling-law research platform (verified up to 1.3B). OpenMythos is a research architecture that applies those principles to sparse (MoE, MLA) and adaptive-depth (ACT or stochastic) looping, with variants defined up to 1T but trained only at 1B to date.

## 16. Quick Reference

**Shared foundations** -- Both use a three-stage pipeline (Prelude/Core/Coda) with diagonal LTI injection (`exp(-dt*A)` decay) guaranteeing spectral radius < 1. They share RMSNorm, RoPE, weight tying, and cosine-decay AdamW training.

**Key differences at a glance:**

| Axis | Parcae | OpenMythos |
|------|--------|------------|
| Core block | 2-8 shared layers per iteration | 1 block per iteration |
| Halting | Stochastic depth at training, fixed at inference | Switchable: ACT (per-position early exit) OR uniform stochastic depth (default on main) |
| FFN | Dense (ReLU^2) | MoE (recurrent block only) |
| Attention | GQA + value embedding gates | MLA (10-20x cache compression) |
| Loop conditioning | No explicit loop embedding; LTI state evolution differentiates iterations | Sinusoidal loop-index + depth-wise LoRA |
| Scale | 140M-1.3B (4 variants, trained) | 1B-1T (7 variants defined; only 1B trained) |
| Scaling laws | Formal: `mu_rec ~ FLOP^0.40`, `D ~ FLOP^0.78` | Not analyzed |
| Philosophy | Theory + scaling law research | Research arch for sparse + adaptive-depth looping; extreme-scale ambitions, unverified |

**Bottom line**: Parcae provides the theoretical framework and scaling laws for stable looped transformers, verified up to 1.3B. OpenMythos applies those principles in a research architecture with MoE, MLA, and switchable ACT/stochastic-depth looping; variants are defined up to 1T but only the 1B variant has been trained to date.

## 17. Cross-Pollination — Ideation (Not Implemented)

> **Status: exploratory.** Nothing in this section has been implemented, decided, or committed to. Items are ranked by estimated ROI and compatibility with the current OpenMythos stack as a menu of possible future directions, not a roadmap. Each would require its own design, ablation, and sign-off before landing.

### Ideas from Parcae that could be ported to OpenMythos

| # | Idea | Compatibility | Rough effort | Why it might be worth considering |
|---|---|---|---|---|
| 1 | **Truncated BPTT** (first `n−k` iterations under `torch.no_grad()`, last `k` with gradients) | High | ~1 day + tests | Biggest memory lever in the list. Full BPTT's memory cost scales linearly with `n_loops`, limiting how deep we can train. With `k≪n_loops` we could train at deeper targets on the same hardware. Parcae hit a known FSDP issue with `no_grad` + `find_unused_parameters=False` (`parcae_lm/models/parcae/parcae.py:297-299`); the workaround is documented there. MoE `expert_counts` accumulation under `no_grad` is fine (int64 buffer, gradient-free). |
| 2 | **Fused Triton CE + z-regularization + logit softcap** | High (architecture-orthogonal) | ~0.5 day | Training-loop-only change. Parcae's three loss variants (`full-triton`, `cce`, `hhe`) are drop-in; `logits.logsumexp(-1).pow(2).mean()` and `softcap * tanh(logits/softcap)` are additive stability tweaks with no architectural coupling. |
| 3 | **Per-sequence depth sampling** with `torch.where` gating | High | ~1 day | Today `n_loops` is sampled once per optimizer step on rank 0 and applied to the whole batch. Per-sequence sampling (as in `parcae.py:337-356`) reduces the loss-spike variance from shallow-sample steps. FSDP-safe if all ranks iterate to the same max depth and mask per-sample. |
| 4 | **Poisson-truncated-full sampling** (vs current uniform) | High | hours | Drop-in replacement for the `random.randint(min, max)` call on rank 0. Parcae's default sampling scheme skews coverage toward typical depths while guaranteeing occasional deeper/shallower draws — plausibly a better distribution than flat uniform. |
| 5 | **Curriculum depth ramp** (linear or sqrt over N steps) | High | hours | Slots into the same sampler. Useful for scale-up runs: start shallow so the early loss drop is cheap, ramp to target. Parcae offers both shapes out of the box (`parcae.py:428-442`). |
| 6 | **Prelude-norm** (extra RMSNorm between Prelude and recurrent block) | Trivial | minutes | All 4 Parcae variants enable it (`prelude_norm=True`). One `nn.RMSNorm` instance and one line in `forward`. Would need an ablation before committing. |
| 7 | **`_no_weight_decay` on LTI SSM params** (`log_A`, `log_dt`, `B`) | Trivial | minutes | Parcae marks its SSM params this way (`injection.py:31,35,40`). Needs a check of whether OpenMythos's optimizer setup already excludes them; if not, one-line fix in the param-group split. |
| 8 | **Identity-initialized learnable `B` matrix** (vs current scalar `0.1 * ones(dim)`) | Medium | ~0.5 day | More parameters per `LTIInjection`, but Parcae's identity init provides cleaner signal preservation at step 0. Could be ablated at 1B. |
| 9 | **Scaled-zero + orthogonal init** | Medium | ~1 day + ablation | Architecture-agnostic but requires controlled comparison; changing init mid-recipe is risky without evidence. |
| 10 | **MuonAdamW optimizer** | Medium (orthogonal but invasive) | ~2–3 days | Parcae uses LR = 8e-3 with MuonAdamW; OpenMythos uses 3e-4 with fused AdamW. Potential training-efficiency win but demands careful re-tuning, and Muon × MoE is untested. |
| 11 | **Best-fit token packing** | Medium (dataloader change only) | ~1–2 days | Pure throughput win (~100% token utilization vs standard padding). Independent of model architecture. |
| 12 | **Value embeddings on alternating layers** | Low | multi-day | Each VE layer adds `vocab × kv_dim` parameters (~33M per layer at our vocab=32k, kv_dim=1024). Unclear ROI on top of MoE's existing capacity mechanism. Defer. |

### Architecturally incompatible

These Parcae design choices would fundamentally conflict with the current OpenMythos architecture and are **not suitable for porting** without a full redesign:

- **Multi-layer core block** (Parcae's 2–8 shared layers per iteration) — contradicts OpenMythos's single-block-shared-weights philosophy; would dilute parameter efficiency and complicate MoE-per-recurrent-block routing.
- **Separate `recurrent_embedding_dimension` + learned C output projection** — would require replacing `LTIInjection` wholesale and would break state-dict compatibility with every existing checkpoint.
- **Dense-only architecture** — OpenMythos is committed to MoE for sparse scaling; regressing to dense defeats the parameter-efficiency motivation.
- **Injection ordering** (Parcae applies `adapter(x, e)` **before** the core layers; OpenMythos adds the transformer output **into** the LTI update) — flipping the ordering would require re-deriving the stability bound with the loop-index embedding and LoRA delta in the residual path.

### Top-3 candidates for a closer look

If and when we decide to pursue any of these, the three with the best signal-to-noise ratio for OpenMythos specifically are:

1. **Truncated BPTT** — unlocks training at deeper `n_loops` targets under the same memory budget.
2. **Fused CE + z-reg + logit softcap** — measured loss-quality wins in Parcae with zero architectural coupling.
3. **Per-sequence or Poisson-truncated-full depth sampling** — refinements of the stochastic-depth recipe already on main.

All three compose cleanly with MoE, MLA, ACT, and the current LTI injection, and none of them touch checkpoint-persistent state.
