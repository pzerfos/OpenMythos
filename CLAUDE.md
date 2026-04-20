# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenMythos is a theoretical open-source reconstruction of a **Recurrent-Depth Transformer (RDT)** architecture. The model uses a three-stage pipeline: Prelude (standard transformer blocks, run once) -> Recurrent Block (single transformer block looped T times with LTI-stable state injection) -> Coda (standard transformer blocks, run once). Key innovations include Adaptive Computation Time (ACT) halting, depth-wise LoRA, fine-grained Mixture-of-Experts (MoE), and switchable attention (GQA or Multi-Latent Attention).

## Environment Setup

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Poetry inside the venv
pip install poetry

# Install the project and all dev dependencies
poetry install --with dev,lint,test
```

Always activate the virtual environment before working:
```bash
source .venv/bin/activate
```

## Testing

```bash
# Run all tests
pytest test_main.py -v              # Root-level comprehensive tests (678 lines)
pytest tests/ -v -s                 # tests/ directory (tokenizer, RoPE debug)

# Run a single test
pytest test_main.py -v -k "test_name"
```

Tests run on CPU by default (small configs with dim=64, 2 experts, etc.). No GPU required for the test suite.

## Linting & Formatting

```bash
black .
ruff check .
ruff check --fix .
```

Config: line-length 88, target Python 3.10. Settings in `pyproject.toml`.

## Training

```bash
# Single GPU
python training/3b_fine_web_edu.py

# Multi-GPU (FSDP)
torchrun --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") training/3b_fine_web_edu.py
```

Training uses FineWeb-Edu streaming dataset, AdamW optimizer, bfloat16 (H100/A100) or float16+GradScaler (older GPUs), linear warmup (2000 steps) then cosine decay. Training has separate dependencies in `training/requirements.txt`.

## Architecture (main.py)

The entire model lives in `open_mythos/main.py` (~1050 lines). Key class hierarchy:

- **`MythosConfig`** — dataclass holding all hyperparameters (attention type, MoE config, ACT threshold, loop count, etc.)
- **`OpenMythos(nn.Module)`** — full model: embedding -> prelude -> recurrent -> coda -> LM head (weight-tied with embeddings)
  - **`TransformerBlock`** — pre-norm block with swappable attention + FFN
    - **`GQAttention`** — Grouped Query Attention with KV cache
    - **`MLAttention`** — Multi-Latent Attention (DeepSeek-V2 style); caches compressed `c_kv` + `k_rope` (~10-20x smaller cache)
    - **`MoEFFN`** — fine-grained MoE with shared + routed experts, aux-loss-free load balancing
    - **`Expert`** — single SwiGLU FFN (`down(silu(gate(x)) * up(x))`)
  - **`RecurrentBlock`** — the looped core: at each iteration applies transformer block, depth-wise LoRA, LTI injection, and accumulates ACT halting weights
    - **`LTIInjection`** — linear time-invariant state update with guaranteed spectral radius < 1 (ZOH discretization)
    - **`ACTHalting`** — per-position adaptive halting probability; early-converging positions stop looping
    - **`LoRAAdapter`** — per-loop-index low-rank adaptation with depth scaling

## Other Key Files

- **`open_mythos/variants.py`** — pre-configured model scales from 1B to 1T parameters
- **`open_mythos/tokenizer.py`** — wrapper around HuggingFace `AutoTokenizer` (default: `openai/gpt-oss-20b`)
- **`open_mythos/moda.py`** — alternative Mixture-of-Depths Attention architecture with depth KV cache

## Key Design Patterns

- **Attention type is runtime-switchable** via `MythosConfig.attn_type` ("gqa" or "mla"). Both share the same `TransformerBlock` interface; MLA reconstructs full K/V from compressed latents on demand.
- **RoPE frequencies are precomputed** as complex phasors and stored as buffers. GQA and MLA use separate rope buffers (different head dims). Frequencies are lazily extended when sequence length exceeds the precomputed range.
- **The recurrent block injects sinusoidal loop-index embeddings** to differentiate behavior across iterations, plus re-injects the original input `e` at every loop to prevent hidden state drift.
- **KV cache** is accumulated across sequence positions (prefill + decode). During generation, `start_pos` tracks the current decode position.
- **ACT remainder trick**: the final loop iteration gets weight `R = 1 - sum(previous weights)` to ensure per-position weights sum to exactly 1.
