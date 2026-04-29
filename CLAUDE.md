# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## Project Overview

OpenMythos is an open-source reconstruction of a **Recurrent-Depth Transformer (RDT)** architecture. The model has a three-stage pipeline: Prelude (standard transformer blocks, run once) → Recurrent Block (single transformer block looped T times with LTI-stable state injection) → Coda (standard transformer blocks, run once). Key features: switchable recurrent-training recipe (ACT vs stochastic_depth), fine-grained MoE with DeepSeek-V3 aux-loss-free load balancing, depth-wise LoRA, switchable attention (GQA or Multi-Latent Attention).

The canonical state-of-the-world roadmap is maintained in the latest logbook under `docs/logbook/`. Currently: `docs/logbook/2026-04-28-option-b-and-upstream-pr.md` tracks open/closed items and forward plan; per-feature deep-dives live in dedicated date-prefixed files.

## Experimentation Workflow

**Dev happens on the local laptop; training happens on BlueVela.** The loop:

```
local edit → pytest → git commit → git push origin → BlueVela git pull → bsub
```

1. **Write code and tests on the laptop.** Always in the `.venv` (see below). Run `pytest tests/` before pushing — the unit suite is CPU-only and catches most regressions in under 2 min.
2. **Push to `origin` (IBM-internal fork):** `git push origin <branch>`. Do not push to `upstream` (public) without explicit intent.
3. **Pull on BlueVela:** `ssh pzerfos@login4.bluevela.rmf.ibm.com`, then `cd /u/pzerfos/OpenMythos && git pull`. The login node has internet; compute nodes do not.
4. **Submit training / eval:** `bash deploy/bluevela/bsub_*.sh` from the BlueVela repo. Logs land under `/u/pzerfos/data/granite-mythos/output/experiments/errs_and_logs/`, checkpoints under `/proj/checkpoints/pzerfos/openmythos/checkpoints/`.

Reference memory (persisted across Claude sessions): `BlueVela GPU server`, `BlueVela network restrictions`, `BlueVela checkpoint storage`.

## Environments

Two distinct environments. They are **not** interchangeable.

### Local laptop — `.venv` (mandatory)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install poetry
poetry install --with dev,lint,test
```

**Always activate the venv before any local work:**
```bash
source .venv/bin/activate
```

The venv is where `pytest`, `black`, and `ruff` run. Do not install project deps system-wide on the laptop.

### BlueVela — conda env `openmythos` (mandatory)

BlueVela's system Python is 3.9, below our `>=3.10` requirement. We use a dedicated conda env:

```bash
# First-time setup (already done once):
bash deploy/bluevela/setup_env.sh
```

LSF jobs activate it non-interactively — the bsub scripts do:
```bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate openmythos
```

Never run training from the system Python on BlueVela.

## Testing

All tests run on CPU in the local `.venv` with small configs (dim=64, few experts). No GPU required.

```bash
source .venv/bin/activate
pytest tests/ -v              # full suite (334 tests as of 2026-04-29)
pytest tests/test_main.py -v -k "test_name"   # single test
```

Always run the full suite before opening a PR.

## Linting & Formatting

```bash
black .
ruff check .
ruff check --fix .
```

Config: line-length 88, target Python 3.10. Settings in `pyproject.toml`.

## Training

The 1B production training script is `training/1b_poc_fineweb.py`. Legacy `training/3b_fine_web_edu.py` exists for the 3B variant.

### Submission on BlueVela

```bash
cd /u/pzerfos/OpenMythos
bash deploy/bluevela/bsub_1b_10b.sh   # submits to preemptable queue, 8 GPUs single node
bjobs -u pzerfos                      # check status
```

Logs stream to `/u/pzerfos/data/granite-mythos/output/experiments/errs_and_logs/`. Checkpoints auto-resume from the latest `step_*.pt` in the checkpoint dir.

### Key local knobs in `training/1b_poc_fineweb.py`

These are plain locals near the top of `main()`. Edit in place, commit, push, pull on BlueVela. `variant` additionally honors a `--variant` CLI flag for submission-time selection (see below); the others are locals only.

- **`recurrent_mode`** — `"stochastic_depth"` (default) or `"act"`. Stochastic_depth samples `n_loops` uniformly from `[stochastic_depth_min, stochastic_depth_max]` per optimizer step (broadcast from rank 0 to prevent FSDP collective-ordering deadlock). ACT uses the original adaptive halting recipe. Checkpoints are cross-mode compatible.
- **`stochastic_depth_min`, `stochastic_depth_max`** — defaults `1` and `32`.
- **`router_bias_update_rate`** — DeepSeek-V3 aux-loss-free load balancing. Default `1e-3` (live on the in-flight 10B run since step 156,000 on 2026-04-29, via commit `5570b5f`; prior default was `0.0`). Set to `0.0` to disable. Emits `router_imbalance_{max_over_mean,stddev_over_mean,ratio}` to ClearML when > 0. See `docs/logbook/2026-04-29-enable-router-bias-update.md`.
- **`variant`** — NoPE ablation variant. Default `"baseline"` (full RoPE via `mythos_1b()`). Other values: `"scoped"` (NoPE in recurrent block only, via `mythos_1b_scoped_nope()`) and `"partial"` (MLA `qk_rope_head_dim=0`, via `mythos_1b_partial_nope()`). Overridable at submission time via `--variant {baseline,scoped,partial}` CLI flag (argparse `parse_known_args` tolerates torchrun's own args). Non-baseline variants write checkpoints under `nope-ablation/{variant}/` and report to a separate ClearML task `nope-ablation/{variant}`. See `docs/superpowers/specs/2026-04-29-nope-for-recurrent-depth-design.md`.

## Architecture

The entire model lives in `open_mythos/main.py`. Key class hierarchy:

- **`MythosConfig`** — dataclass holding all hyperparameters.
- **`OpenMythos(nn.Module)`** — embedding → prelude → recurrent → coda → LM head (weight-tied with embeddings). `forward(..., bypass_act: bool)` routes between ACT and stochastic_depth modes. `update_router_biases(rate, ddp)` applies DeepSeek-V3 Algorithm 1 across every MoE layer.
  - **`TransformerBlock`** — pre-norm block with swappable attention + FFN.
    - **`GQAttention`** — Grouped Query Attention with KV cache. Uses Flash Attention 2 when installed, falls back to manual attention otherwise.
    - **`MLAttention`** — Multi-Latent Attention (DeepSeek-V2 style); caches compressed `c_kv` + `k_rope` (~10-20× smaller cache).
    - **`MoEFFN`** — fine-grained MoE with shared + routed experts. Aux-loss-free load balancing via `router_bias`; `expert_counts` accumulate in `forward`, `update_router_bias(rate, ddp)` all-reduces and shifts bias by `-rate * sign(counts - target)`. Grouped dispatch (sort + batch per expert) replaces the naive nested loop (~6.7× speedup).
    - **`Expert`** — single SwiGLU FFN (`down(silu(gate(x)) * up(x))`).
  - **`RecurrentBlock`** — the looped core. At each iteration applies the transformer block, depth-wise LoRA, LTI injection, and (under ACT) accumulates halting weights. Under `bypass_act=True`, runs all `n_loops` iterations and returns the final hidden state.
    - **`LTIInjection`** — linear time-invariant state update with guaranteed spectral radius < 1 (ZOH discretization; clamp `log_dt+log_A` to `[-13, 20]` keeps `A < 1.0` strictly in fp32).
    - **`ACTHalting`** — per-position adaptive halting probability.
    - **`LoRAAdapter`** — per-loop-index low-rank adaptation.

## Other Key Files

- **`open_mythos/variants.py`** — pre-configured model scales from 1B to 1T parameters.
- **`open_mythos/tokenizer.py`** — wrapper around HuggingFace `AutoTokenizer` (default: `openai/gpt-oss-20b`).
- **`open_mythos/moda.py`** — alternative Mixture-of-Depths Attention architecture.
- **`evaluations/eval_checkpoint.py`** — standalone checkpoint eval (generation + depth sweep); submit via `deploy/bluevela/bsub_eval.sh`.
- **`deploy/bluevela/`** — BlueVela LSF deployment scripts (setup, training, eval submission).
- **`docs/logbook/`** — dated logbook entries; canonical source of truth for project state, decisions, and next steps. Read the latest entry first when picking up work.

## Key Design Patterns

- **Attention type is runtime-switchable** via `MythosConfig.attn_type` (`"gqa"` or `"mla"`). Both share the same `TransformerBlock` interface; MLA reconstructs full K/V from compressed latents on demand.
- **RoPE frequencies are precomputed** as complex phasors and stored as buffers. GQA and MLA use separate rope buffers (different head dims). `OpenMythos.forward` slices `freqs_cis[start_pos:start_pos + T]`; unit tests that bypass the top-level forward must slice explicitly.
- **The recurrent block injects sinusoidal loop-index embeddings** to differentiate behavior across iterations, plus re-injects the original input `e` at every loop to prevent hidden state drift.
- **KV cache** is accumulated across sequence positions (prefill + decode). During generation, `start_pos` tracks the current decode position.
- **FSDP collective-ordering invariant:** any call that contains an `all_reduce` / `all_gather` must execute on every rank in the same order. Rank-local early-exits (e.g. ACT's original `halted.all()` break, or skipping a MoE layer whose local counts are zero) cause NCCL-watchdog deadlocks. See `docs/logbook/2026-04-23-act-fsdp-deadlock.md` for the canonical case; `MoEFFN.update_router_bias` and `RecurrentBlock.forward` carry pinned `INVARIANT:` comments.
- **FSDP MixedPrecision and buffers:** `training/1b_poc_fineweb.py` uses `MixedPrecision(param_dtype=bf16, reduce_dtype=bf16)` and deliberately omits `buffer_dtype`. Casting buffers to bf16 would corrupt int64 `expert_counts` (saturates above 256) and fp32 `router_bias` (update of `1e-3` falls below bf16 resolution).
- **Checkpoints are cross-mode compatible** between `recurrent_mode="act"` and `"stochastic_depth"` — `ACTHalting` weights are present in both cases and simply receive no gradient under `bypass_act=True`.
