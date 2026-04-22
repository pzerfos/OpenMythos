# OpenMythos 1B Proof-of-Concept Training Design

## Goal

Validate that the OpenMythos 1B recurrent-depth transformer architecture trains correctly and learns to generate coherent text. Train on ~10B tokens of FineWeb-Edu with ClearML experiment tracking, running on 2 GPUs on the BlueVela cluster via IBM LSF. Claude Code orchestrates the full workflow from the user's laptop via SSH.

## Success Criteria

1. Training loss steadily decreases over the run (visible in ClearML dashboard)
2. Model generates recognizable English text from fixed prompts after training

## Approach

Fork the existing `training/3b_fine_web_edu.py` into a 1B-targeted script. Add ClearML integration. Create a `deploy/` directory for compute-backend-specific launch configs (BlueVela now, Granite.build later).

---

## Model Configuration

Use the 1B variant from `open_mythos/variants.py`:

| Parameter | Value |
|-----------|-------|
| dim | 2048 |
| n_heads | 16 |
| n_kv_heads | 4 |
| attn_type | mla |
| max_seq_len | 4096 |
| max_loop_iters | 16 |
| n_experts | 64 (2 shared) |
| expert_dim | 2048 |
| n_experts_per_tok | 4 |
| lora_rank | 8 |
| rope_theta | 500,000 |
| vocab_size | 32,000 |
| prelude_layers | 2 |
| coda_layers | 2 |

## Training Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Target tokens | 10B | ~12-16 hrs on 2x A100/H100 |
| Sequence length | 2048 | Shorter than max (4096), fine for PoC |
| Micro-batch size | 8 | 1B fits more per batch than 3B |
| Max LR | 3e-4 | Standard for this scale |
| Min LR | 3e-5 | Cosine decay floor |
| Warmup steps | 2000 | Linear warmup |
| LR schedule | Linear warmup + cosine decay | Inherited from existing script |
| Optimizer | Fused AdamW | betas=(0.9, 0.95), weight_decay=0.1 |
| Grad clip | 1.0 | Max norm |
| Precision | bfloat16 | Native on H100/A100 |
| Checkpoint every | 1000 steps | Keeps last 3 |
| Log every | 10 steps | |
| Dataset | FineWeb-Edu sample-10BT | Streaming, no full materialization |
| Distributed strategy | FSDP (2 GPUs) | Full shard, wrap TransformerBlock + RecurrentBlock |

Grad accumulation computed automatically: `grad_accum = max(1, 256 // (world_size * micro_batch))`.

## ClearML Integration

### Authentication

ClearML SDK reads credentials from environment variables — no interactive `clearml-init` required:

- `CLEARML_API_HOST` — ClearML server URL
- `CLEARML_API_ACCESS_KEY` — API access key
- `CLEARML_API_SECRET_KEY` — API secret key

### Task Initialization

On rank 0 only:

```python
from clearml import Task
task = Task.init(
    project_name=os.environ.get("CLEARML_PROJECT", "granite-mythos"),
    task_name=os.environ.get("EXPERIMENT_NAME", "1b-poc-fineweb-10B"),
)
task.connect(vars(cfg))  # log all MythosConfig fields as hyperparameters
```

### Metrics (reported every 10 steps, rank 0 only)

| Metric | Series | Purpose |
|--------|--------|---------|
| `train/loss` | Training loss | Core convergence signal |
| `train/grad_norm` | Gradient L2 norm | Stability monitoring — spikes indicate problems |
| `train/lr` | Learning rate | Verify warmup + cosine schedule |
| `train/throughput_mtok_s` | M tokens/sec | GPU utilization |
| `train/tokens_seen` | Cumulative tokens | Progress toward 10B target |

### Artifacts

- Hyperparameters: auto-logged via `task.connect()`
- Final checkpoint path: registered as ClearML artifact
- Generation samples: logged as text at end of training

## Post-Training Generation Test

After training completes:

1. Switch to eval mode (`model.eval()`, `torch.no_grad()`)
2. Run greedy decode on fixed prompts:
   - `"The purpose of education is"`
   - `"In the beginning, there was"`
   - `"The most important scientific discovery"`
3. Generate up to 128 tokens per prompt using KV-cache-based autoregressive decode
4. Print to stdout and log to ClearML as text artifacts
5. No quality threshold — visual sanity check only (recognizable English = pass, gibberish = investigate)

## Environment Variables

All secrets and runtime configuration are read from environment variables. No hardcoded credentials.

### Required

| Variable | Purpose |
|----------|---------|
| `CLEARML_API_HOST` | ClearML server URL |
| `CLEARML_API_ACCESS_KEY` | ClearML API access key |
| `CLEARML_API_SECRET_KEY` | ClearML API secret key |
| `HF_TOKEN` | HuggingFace token for FineWeb-Edu dataset access |

### Optional (with defaults)

| Variable | Purpose | Default |
|----------|---------|---------|
| `CLEARML_PROJECT` | ClearML project name | `granite-mythos` |
| `EXPERIMENT_NAME` | ClearML task name | `1b-poc-fineweb-10B` |
| `OUTPUT_DIR` | Checkpoint and log directory | `/u/pzerfos/data/granite-mythos/output/experiments` |
| `NUM_GPUS` | GPUs to request in bsub | `2` |
| `TARGET_TOKENS` | Token budget in billions | `10` |

### Prerequisites

- HuggingFace: accept FineWeb-Edu license at https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu, generate token at https://huggingface.co/settings/tokens
- ClearML: obtain API credentials from your ClearML server admin or https://app.clear.ml/settings

## Directory Structure

```
training/
├── 3b_fine_web_edu.py          # Existing 3B script (unchanged)
├── 1b_poc_fineweb.py           # New — 1B PoC training script
└── requirements.txt            # Modified — add clearml>=1.16.0

deploy/
├── bluevela/
│   ├── bsub_1b_poc.sh          # LSF job submission script
│   └── setup_env.sh            # One-time conda env + dependency setup
└── granite-build/
    └── (future — Granite.build job config)
```

### File Details

**`training/1b_poc_fineweb.py`** — Forked from `3b_fine_web_edu.py`. Changes:
- Model config: 1B variant instead of 3B
- Token budget: reads `TARGET_TOKENS` env var (default 10B)
- Micro-batch: 8 (up from 4)
- ClearML: Task.init, metric reporting, hyperparameter logging, artifact registration
- Generation test: runs fixed prompts after training, logs to ClearML
- Output dir: reads `OUTPUT_DIR` env var

**`training/requirements.txt`** — Add `clearml>=1.16.0`.

**`deploy/bluevela/bsub_1b_poc.sh`** — LSF job script adapted from existing `bsubcmd.sh`:
- Queue: `preemptable`, group: `grp_preemptable`
- GPUs: reads `NUM_GPUS` env var (default 2), exclusive process mode
- 1 node
- Activates conda env before launching
- Uses `blaunch` with `torchrun --nproc_per_node=$NUM_GPUS` for multi-GPU FSDP
- Logs to `$OUTPUT_DIR/errs_and_logs/`
- Passes through all environment variables to the job

**`deploy/bluevela/setup_env.sh`** — One-time setup script:
- Validates required env vars are set (errors with clear message if missing)
- Clones or pulls the repo from `ssh://git@github.ibm.com/pzerfos/OpenMythos.git`
- Creates conda env `openmythos` with Python 3.10
- Installs Poetry, project deps, and training requirements
- Verifies ClearML connectivity
- Verifies HuggingFace token works

**`deploy/granite-build/`** — Empty placeholder for future Granite.build integration.

## Remote Workflow (Claude Code-Driven)

Claude Code orchestrates the full workflow from the user's laptop via SSH to `pzerfos@login4.bluevela.rmf.ibm.com`.

### Phase 1 — Setup (one-time)

1. User exports required env vars locally (or in `~/.bashrc` on BlueVela)
2. Claude Code SSHs in, runs `deploy/bluevela/setup_env.sh`
3. Validates environment is ready

### Phase 2 — Submit

4. Claude Code SSHs in, runs `bash deploy/bluevela/bsub_1b_poc.sh`
5. Returns the LSF job ID

### Phase 3 — Monitor

6. Claude Code checks job status via `ssh ... "bjobs"`
7. Claude Code tails log files via `ssh ... "tail -50 $OUTPUT_DIR/errs_and_logs/..."`
8. ClearML dashboard provides independent live monitoring of loss curves and metrics

### SSH Command Pattern

All remote commands follow:
```bash
ssh pzerfos@login4.bluevela.rmf.ibm.com "<command>"
```
No interactive sessions needed. SSH key auth assumed.

## What Stays Unchanged

- `open_mythos/main.py` — model architecture
- `open_mythos/variants.py` — variant configs
- `open_mythos/tokenizer.py` — tokenizer wrapper
- `training/3b_fine_web_edu.py` — existing 3B training script
- All tests
