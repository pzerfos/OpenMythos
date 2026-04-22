# 1B PoC Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a 1B PoC training script with ClearML integration and BlueVela LSF deployment scripts, forked from the existing 3B training script.

**Architecture:** Fork `training/3b_fine_web_edu.py` → `training/1b_poc_fineweb.py` with 1B config, env-var-driven hyperparameters, ClearML tracking, and post-training generation. Add `deploy/bluevela/` with LSF job script and environment setup script.

**Tech Stack:** PyTorch, FSDP, ClearML, IBM LSF, venv, FineWeb-Edu (HuggingFace streaming)

**Spec:** `docs/superpowers/specs/2026-04-22-1b-poc-training-design.md`

---

### Task 1: Add clearml to training requirements

**Files:**
- Modify: `training/requirements.txt`

- [ ] **Step 1: Add clearml dependency**

In `training/requirements.txt`, add `clearml>=1.16.0` after the existing entries:

```
torch>=2.11.0
datasets>=3.6.0
loguru>=0.7.3
open-mythos
clearml>=1.16.0
```

- [ ] **Step 2: Verify the file**

Run: `cat training/requirements.txt`
Expected: 5 lines, `clearml>=1.16.0` is the last entry.

- [ ] **Step 3: Commit**

```bash
git add training/requirements.txt
git commit -m "feat(training): add clearml to training requirements"
```

---

### Task 2: Create the 1B PoC training script

**Files:**
- Create: `training/1b_poc_fineweb.py`
- Reference (read-only): `training/3b_fine_web_edu.py`, `open_mythos/variants.py`

This is the main deliverable. Fork from `3b_fine_web_edu.py` with these changes:
- Import `mythos_1b` instead of `mythos_3b`
- Read `TARGET_TOKENS`, `OUTPUT_DIR`, `EXPERIMENT_NAME`, `CLEARML_PROJECT` from env vars
- Micro-batch 8 instead of 4
- Add ClearML Task.init (rank 0 only), metric reporting, hyperparameter logging
- Add post-training generation test

- [ ] **Step 1: Create the training script**

Create `training/1b_poc_fineweb.py` with this content:

```python
#!/usr/bin/env python3
"""
OpenMythos 1B PoC pretraining on FineWeb-Edu with FSDP + AdamW + ClearML.

Single GPU:
    python training/1b_poc_fineweb.py

Multi-GPU:
    torchrun --nproc_per_node=N training/1b_poc_fineweb.py

Environment variables (required):
    CLEARML_API_HOST       -- ClearML server URL
    CLEARML_API_ACCESS_KEY -- ClearML API access key
    CLEARML_API_SECRET_KEY -- ClearML API secret key
    HF_TOKEN               -- HuggingFace token for FineWeb-Edu

Environment variables (optional):
    CLEARML_PROJECT  -- ClearML project name (default: granite-mythos)
    EXPERIMENT_NAME  -- ClearML task name (default: 1b-poc-fineweb-10B)
    OUTPUT_DIR       -- checkpoint dir (default: /u/pzerfos/data/granite-mythos/output/experiments)
    TARGET_TOKENS    -- token budget in billions (default: 10)
    NUM_GPUS         -- number of GPUs (informational, actual count from torchrun)
"""

import os
import math
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from loguru import logger
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from contextlib import nullcontext

from datasets import load_dataset

from open_mythos import OpenMythos
from open_mythos.main import TransformerBlock, RecurrentBlock
from open_mythos.variants import mythos_1b
from open_mythos.tokenizer import MythosTokenizer


# ---------------------------------------------------------------------------
# ClearML (lazy — only initialized on rank 0)
# ---------------------------------------------------------------------------

_clearml_task = None
_clearml_logger = None


def init_clearml(cfg, training_hparams: dict):
    """Initialize ClearML tracking on rank 0. No-op if ClearML env vars are missing."""
    global _clearml_task, _clearml_logger
    try:
        from clearml import Task

        project = os.environ.get("CLEARML_PROJECT", "granite-mythos")
        task_name = os.environ.get("EXPERIMENT_NAME", "1b-poc-fineweb-10B")

        _clearml_task = Task.init(project_name=project, task_name=task_name)
        _clearml_task.connect(vars(cfg), name="model_config")
        _clearml_task.connect(training_hparams, name="training_hparams")
        _clearml_logger = _clearml_task.get_logger()
        logger.info(f"ClearML initialized: project={project}, task={task_name}")
    except Exception as e:
        logger.warning(f"ClearML init failed (training continues without tracking): {e}")


def log_clearml(series: str, value: float, step: int):
    """Report a scalar to ClearML if available."""
    if _clearml_logger is not None:
        _clearml_logger.report_scalar("train", series, iteration=step, value=value)


def log_clearml_text(title: str, text: str):
    """Log text to ClearML if available."""
    if _clearml_task is not None:
        _clearml_task.get_logger().report_text(f"## {title}\n\n{text}")


def register_clearml_artifact(name: str, path: str):
    """Register a file artifact in ClearML if available."""
    if _clearml_task is not None:
        _clearml_task.upload_artifact(name, artifact_object=path)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class FineWebEduDataset(IterableDataset):
    """
    Streaming FineWeb-Edu loader yielding fixed-length (input, target) pairs.

    Documents are concatenated into a rolling buffer and sliced into
    fixed-length chunks. Sharding is two-dimensional: world_size ranks x
    num_workers DataLoader workers per rank.
    """

    def __init__(self, encoding, seq_len: int, subset: str, rank: int, world_size: int):
        self.encoding = encoding
        self.seq_len = seq_len
        self.subset = subset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        worker = get_worker_info()
        num_workers = worker.num_workers if worker else 1
        worker_id = worker.id if worker else 0

        total_shards = self.world_size * num_workers
        shard_index = self.rank * num_workers + worker_id

        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=self.subset,
            split="train",
            streaming=True,
        ).shard(num_shards=total_shards, index=shard_index)

        buf = []
        for sample in ds:
            buf.extend(self.encoding.encode(sample["text"]))
            while len(buf) >= self.seq_len + 1:
                chunk = buf[: self.seq_len + 1]
                buf = buf[self.seq_len + 1 :]
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:], dtype=torch.long),
                )


# ---------------------------------------------------------------------------
# LR schedule: linear warmup -> cosine decay
# ---------------------------------------------------------------------------


def get_lr(step: int, warmup: int, total: int, max_lr: float, min_lr: float) -> float:
    if step < warmup:
        return max_lr * step / warmup
    if step >= total:
        return min_lr
    decay = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * decay))


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def _list_ckpts(ckpt_dir: str) -> list[str]:
    if not os.path.isdir(ckpt_dir):
        return []
    return sorted(
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.startswith("step_") and f.endswith(".pt")
    )


def save_checkpoint(
    model, optimizer, step: int, cfg, vocab_size: int,
    ckpt_dir: str, ddp: bool, master: bool, keep_last: int = 3,
) -> None:
    if ddp:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            model_state = model.state_dict()
            optim_state = FSDP.optim_state_dict(model, optimizer)
    else:
        model_state = model.state_dict()
        optim_state = optimizer.state_dict()

    if not master:
        return

    os.makedirs(ckpt_dir, exist_ok=True)
    final_path = os.path.join(ckpt_dir, f"step_{step:07d}.pt")
    tmp_path = final_path + ".tmp"
    torch.save(
        {
            "step": step,
            "model": model_state,
            "optimizer": optim_state,
            "cfg": cfg,
            "vocab_size": vocab_size,
        },
        tmp_path,
    )
    os.replace(tmp_path, final_path)

    for old in _list_ckpts(ckpt_dir)[:-keep_last]:
        try:
            os.remove(old)
        except OSError as exc:
            logger.warning(f"Failed to prune old checkpoint {old}: {exc}")

    logger.success(f"Checkpoint saved -> {final_path}")
    register_clearml_artifact(f"checkpoint_step_{step}", final_path)


def load_checkpoint(model, optimizer, path: str, ddp: bool) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if ddp:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        ):
            model.load_state_dict(ckpt["model"])
            optim_state = FSDP.optim_state_dict_to_load(
                model=model, optim=optimizer, optim_state_dict=ckpt["optimizer"],
            )
            optimizer.load_state_dict(optim_state)
    else:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

    return int(ckpt["step"])


# ---------------------------------------------------------------------------
# Post-training generation test
# ---------------------------------------------------------------------------


GENERATION_PROMPTS = [
    "The purpose of education is",
    "In the beginning, there was",
    "The most important scientific discovery",
]


def run_generation_test(model, encoding, device: str, ddp: bool):
    """Run greedy generation on fixed prompts and log results."""
    logger.info("Running post-training generation test...")

    # Unwrap FSDP for generation (generate uses KV cache which needs the raw model)
    raw_model = model.module if ddp else model
    raw_model.eval()

    results = []
    for prompt_text in GENERATION_PROMPTS:
        tokens = encoding.encode(prompt_text)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

        with torch.no_grad():
            output_ids = raw_model.generate(
                input_ids,
                max_new_tokens=128,
                temperature=0.8,
                top_k=40,
            )

        generated = encoding.decode(output_ids[0].tolist())
        result = f"**Prompt:** {prompt_text}\n**Generated:** {generated}\n"
        results.append(result)
        logger.info(f"\n{result}")

    all_results = "\n---\n".join(results)
    log_clearml_text("Generation Samples", all_results)
    raw_model.train()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # ------------------------------------------------------------------
    # Distributed init
    # ------------------------------------------------------------------
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        rank = local_rank = 0
        world_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"

    master = rank == 0

    if master:
        logger.info(
            f"GPUs: {torch.cuda.device_count()}  |  World size: {world_size}  |  Device: {device}"
        )

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    encoding = MythosTokenizer()
    vocab_size = encoding.vocab_size

    if master:
        logger.info(f"Tokenizer: gpt-oss-20b  |  Vocab size: {vocab_size:,}")

    # ------------------------------------------------------------------
    # Hyperparameters (env-var configurable with defaults)
    # ------------------------------------------------------------------
    seq_len = 2048
    micro_batch = 8
    target_tokens_b = int(os.environ.get("TARGET_TOKENS", "10"))
    target_tokens = target_tokens_b * 1_000_000_000
    grad_accum = max(1, 256 // (world_size * micro_batch))
    global_batch_tok = world_size * micro_batch * grad_accum * seq_len
    total_steps = target_tokens // global_batch_tok
    warmup_steps = 2000
    lr = 3e-4
    min_lr = 3e-5
    wd = 0.1
    log_every = 10
    ckpt_every = 1000
    output_dir = os.environ.get(
        "OUTPUT_DIR", "/u/pzerfos/data/granite-mythos/output/experiments"
    )
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    dataset_subset = "sample-10BT"

    training_hparams = {
        "seq_len": seq_len,
        "micro_batch": micro_batch,
        "target_tokens": target_tokens,
        "grad_accum": grad_accum,
        "global_batch_tok": global_batch_tok,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "lr": lr,
        "min_lr": min_lr,
        "weight_decay": wd,
        "log_every": log_every,
        "ckpt_every": ckpt_every,
        "output_dir": output_dir,
        "dataset_subset": dataset_subset,
        "world_size": world_size,
    }

    if master:
        logger.info(
            f"seq_len={seq_len} | micro_batch={micro_batch} | grad_accum={grad_accum} | "
            f"global_batch_tokens={global_batch_tok:,} | total_steps={total_steps:,} | "
            f"target_tokens={target_tokens_b}B"
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    cfg = mythos_1b()
    cfg.vocab_size = vocab_size
    cfg.max_seq_len = seq_len

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16

    model = OpenMythos(cfg)

    if ddp:
        mp_policy = MixedPrecision(
            param_dtype=amp_dtype, reduce_dtype=amp_dtype, buffer_dtype=amp_dtype,
        )
        wrap_policy = ModuleWrapPolicy({TransformerBlock, RecurrentBlock})
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp_policy,
            auto_wrap_policy=wrap_policy,
            device_id=local_rank,
        )
    else:
        model = model.to(device)
        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
            if "cuda" in device
            else nullcontext()
        )

    amp_ctx = nullcontext() if ddp else amp_ctx  # type: ignore[possibly-undefined]

    if master:
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Parameters: {n_params:,}  |  AMP dtype: {amp_dtype}")

    # ------------------------------------------------------------------
    # ClearML init (after model is built so we can log config)
    # ------------------------------------------------------------------
    if master:
        init_clearml(cfg, training_hparams)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95), fused=True
    )

    # ------------------------------------------------------------------
    # Resume from latest checkpoint (if any)
    # ------------------------------------------------------------------
    start_step = 0
    existing_ckpts = _list_ckpts(ckpt_dir)
    if existing_ckpts:
        latest = existing_ckpts[-1]
        if master:
            logger.info(f"Resuming from checkpoint: {latest}")
        start_step = load_checkpoint(model, optimizer, latest, ddp)
        if master:
            logger.success(f"Resumed at step {start_step}")

    # ------------------------------------------------------------------
    # Dataset + DataLoader
    # ------------------------------------------------------------------
    dataset = FineWebEduDataset(encoding, seq_len, dataset_subset, rank, world_size)
    loader = DataLoader(dataset, batch_size=micro_batch, num_workers=4, pin_memory=True)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    if master:
        os.makedirs(ckpt_dir, exist_ok=True)

    model.train()
    data_iter = iter(loader)
    t0 = time.perf_counter()
    step = start_step

    while step < total_steps:
        cur_lr = get_lr(step, warmup_steps, total_steps, lr, min_lr)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)

            x = x.to(device if not ddp else f"cuda:{local_rank}", non_blocking=True)
            y = y.to(device if not ddp else f"cuda:{local_rank}", non_blocking=True)

            sync = (
                nullcontext()
                if (not ddp or micro_step == grad_accum - 1)
                else model.no_sync()
            )
            with sync, amp_ctx:
                logits = model(x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, vocab_size), y.view(-1)
                )
                loss = loss / grad_accum

            loss.backward()
            loss_accum += loss.item()

        if ddp:
            grad_norm = model.clip_grad_norm_(1.0)
        else:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        step += 1

        if master and step % log_every == 0:
            dt = time.perf_counter() - t0
            tok_per_sec = global_batch_tok * log_every / dt
            tokens_seen = step * global_batch_tok

            logger.info(
                f"step {step:6d}/{total_steps} | loss {loss_accum:.4f} "
                f"| gnorm {float(grad_norm):.2f} | lr {cur_lr:.2e} "
                f"| {tok_per_sec / 1e6:.2f}M tok/s "
                f"| {tokens_seen / 1e9:.1f}B tokens seen"
            )

            log_clearml("loss", loss_accum, step)
            log_clearml("grad_norm", float(grad_norm), step)
            log_clearml("lr", cur_lr, step)
            log_clearml("throughput_mtok_s", tok_per_sec / 1e6, step)
            log_clearml("tokens_seen_B", tokens_seen / 1e9, step)

            t0 = time.perf_counter()

        if step % ckpt_every == 0:
            save_checkpoint(
                model, optimizer, step, cfg, vocab_size, ckpt_dir, ddp, master
            )

    # Final checkpoint
    if step > start_step and step % ckpt_every != 0:
        save_checkpoint(model, optimizer, step, cfg, vocab_size, ckpt_dir, ddp, master)

    # ------------------------------------------------------------------
    # Post-training generation test (rank 0 only)
    # ------------------------------------------------------------------
    if master:
        run_generation_test(model, encoding, device if not ddp else f"cuda:{local_rank}", ddp)

    if ddp:
        dist.barrier()
        dist.destroy_process_group()

    if master:
        logger.success("Training complete.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script parses without syntax errors**

Run: `python -c "import ast; ast.parse(open('training/1b_poc_fineweb.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Verify imports resolve (in the project venv)**

Run: `source .venv/bin/activate && python -c "from training import __path__" 2>&1 || python -c "import open_mythos; from open_mythos.variants import mythos_1b; print('imports OK')"`
Expected: `imports OK`

- [ ] **Step 4: Commit**

```bash
git add training/1b_poc_fineweb.py
git commit -m "feat(training): add 1B PoC training script with ClearML integration"
```

---

### Task 3: Create BlueVela environment setup script

**Files:**
- Create: `deploy/bluevela/setup_env.sh`

This script is run once on BlueVela to set up the venv and validate all dependencies.

- [ ] **Step 1: Create the setup script**

Create `deploy/bluevela/setup_env.sh`:

```bash
#!/usr/bin/env bash
# One-time environment setup for OpenMythos training on BlueVela.
#
# Usage:
#   ssh pzerfos@login4.bluevela.rmf.ibm.com
#   cd /path/to/OpenMythos
#   bash deploy/bluevela/setup_env.sh
#
# Required environment variables (set in ~/.bashrc or export before running):
#   CLEARML_API_HOST       -- ClearML server URL
#   CLEARML_API_ACCESS_KEY -- ClearML API access key
#   CLEARML_API_SECRET_KEY -- ClearML API secret key
#   HF_TOKEN               -- HuggingFace token for FineWeb-Edu access

set -euo pipefail

# ---------------------------------------------------------------------------
# Validate required environment variables
# ---------------------------------------------------------------------------
REQUIRED_VARS=(CLEARML_API_HOST CLEARML_API_ACCESS_KEY CLEARML_API_SECRET_KEY HF_TOKEN)
MISSING=()

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var:-}" ]; then
        MISSING+=("$var")
    fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "ERROR: Missing required environment variables:"
    for var in "${MISSING[@]}"; do
        echo "  - $var"
    done
    echo ""
    echo "Set them in ~/.bashrc or export before running this script."
    echo "See docs/superpowers/specs/2026-04-22-1b-poc-training-design.md for details."
    exit 1
fi

echo "All required environment variables are set."

# ---------------------------------------------------------------------------
# Create venv and install dependencies
# ---------------------------------------------------------------------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

echo "Setting up venv in $REPO_DIR/.venv ..."

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Created new venv."
else
    echo "Existing venv found, reusing."
fi

source .venv/bin/activate

pip install --upgrade pip
pip install poetry
poetry install
pip install -r training/requirements.txt

echo "Dependencies installed."

# ---------------------------------------------------------------------------
# Verify ClearML connectivity
# ---------------------------------------------------------------------------
echo "Verifying ClearML connectivity..."
python3 -c "
from clearml import Task
print('ClearML SDK loaded successfully.')
print(f'  Host: {Task._get_default_session().config.get(\"api.host\", \"(from env)\")}')
print('ClearML OK.')
"

# ---------------------------------------------------------------------------
# Verify HuggingFace token
# ---------------------------------------------------------------------------
echo "Verifying HuggingFace token..."
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
user = api.whoami()
print(f'  Logged in as: {user[\"name\"]}')
print('HuggingFace OK.')
"

# ---------------------------------------------------------------------------
# Verify model imports
# ---------------------------------------------------------------------------
echo "Verifying OpenMythos imports..."
python3 -c "
from open_mythos import OpenMythos
from open_mythos.variants import mythos_1b
from open_mythos.tokenizer import MythosTokenizer
cfg = mythos_1b()
print(f'  1B config: dim={cfg.dim}, experts={cfg.n_experts}, loops={cfg.max_loop_iters}')
print('OpenMythos OK.')
"

echo ""
echo "========================================="
echo "  Setup complete! Ready to submit jobs."
echo "  Run: bash deploy/bluevela/bsub_1b_poc.sh"
echo "========================================="
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x deploy/bluevela/setup_env.sh`

- [ ] **Step 3: Verify it parses**

Run: `bash -n deploy/bluevela/setup_env.sh && echo "syntax OK"`
Expected: `syntax OK`

- [ ] **Step 4: Commit**

```bash
git add deploy/bluevela/setup_env.sh
git commit -m "feat(deploy): add BlueVela environment setup script"
```

---

### Task 4: Create BlueVela LSF job submission script

**Files:**
- Create: `deploy/bluevela/bsub_1b_poc.sh`
- Reference (read-only): `../bsubcmd.sh` (existing LSF script pattern)

- [ ] **Step 1: Create the bsub script**

Create `deploy/bluevela/bsub_1b_poc.sh`:

```bash
#!/usr/bin/env bash
# Submit OpenMythos 1B PoC training job to BlueVela LSF.
#
# Usage:
#   bash deploy/bluevela/bsub_1b_poc.sh
#
# Required environment variables:
#   CLEARML_API_HOST, CLEARML_API_ACCESS_KEY, CLEARML_API_SECRET_KEY, HF_TOKEN
#
# Optional environment variables:
#   CLEARML_PROJECT  -- ClearML project (default: granite-mythos)
#   EXPERIMENT_NAME  -- ClearML task name (default: 1b-poc-fineweb-10B)
#   OUTPUT_DIR       -- output directory (default: /u/pzerfos/data/granite-mythos/output/experiments)
#   NUM_GPUS         -- GPUs to request (default: 2)
#   TARGET_TOKENS    -- token budget in billions (default: 10)

set -euo pipefail

# ---------------------------------------------------------------------------
# Validate required environment variables
# ---------------------------------------------------------------------------
REQUIRED_VARS=(CLEARML_API_HOST CLEARML_API_ACCESS_KEY CLEARML_API_SECRET_KEY HF_TOKEN)
MISSING=()

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var:-}" ]; then
        MISSING+=("$var")
    fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "ERROR: Missing required environment variables:"
    for var in "${MISSING[@]}"; do
        echo "  - $var"
    done
    exit 1
fi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-/u/pzerfos/data/granite-mythos/output/experiments}"
NUM_GPUS="${NUM_GPUS:-2}"
TARGET_TOKENS="${TARGET_TOKENS:-10}"
CLEARML_PROJECT="${CLEARML_PROJECT:-granite-mythos}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-1b-poc-fineweb-10B}"

umask 0002

DATE=$(date "+%Y-%m-%d-%H-%M")
LOG_DIR="${OUTPUT_DIR}/errs_and_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${DATE}.log"
ERR_FILE="${LOG_DIR}/${DATE}.err"

QUEUE=preemptable
NUM_NODES=1
BSUB_GROUP=grp_preemptable
JOB_NAME="pz-mythos-1b-poc"

echo "========================================="
echo "  OpenMythos 1B PoC Training"
echo "========================================="
echo "  GPUs:          $NUM_GPUS"
echo "  Target tokens: ${TARGET_TOKENS}B"
echo "  Output dir:    $OUTPUT_DIR"
echo "  ClearML:       $CLEARML_PROJECT / $EXPERIMENT_NAME"
echo "  Log:           $LOG_FILE"
echo "  Err:           $ERR_FILE"
echo "========================================="

# ---------------------------------------------------------------------------
# Submit LSF job
# ---------------------------------------------------------------------------
bsub \
    -J "${JOB_NAME}" \
    -q "${QUEUE}" \
    -o "${LOG_FILE}" \
    -e "${ERR_FILE}" \
    -n "${NUM_NODES}" \
    -gpu "num=${NUM_GPUS}/task:mode=exclusive_process" \
    -G "${BSUB_GROUP}" \
    blaunch \
    PYTHONUNBUFFERED=1 \
    CLEARML_API_HOST="${CLEARML_API_HOST}" \
    CLEARML_API_ACCESS_KEY="${CLEARML_API_ACCESS_KEY}" \
    CLEARML_API_SECRET_KEY="${CLEARML_API_SECRET_KEY}" \
    HF_TOKEN="${HF_TOKEN}" \
    CLEARML_PROJECT="${CLEARML_PROJECT}" \
    EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    TARGET_TOKENS="${TARGET_TOKENS}" \
    bash -c "
        cd ${REPO_DIR} && \
        source .venv/bin/activate && \
        torchrun --nproc_per_node=${NUM_GPUS} training/1b_poc_fineweb.py
    " 2>&1 | tee "${OUTPUT_DIR}/${DATE}_submit.log"
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x deploy/bluevela/bsub_1b_poc.sh`

- [ ] **Step 3: Verify it parses**

Run: `bash -n deploy/bluevela/bsub_1b_poc.sh && echo "syntax OK"`
Expected: `syntax OK`

- [ ] **Step 4: Commit**

```bash
git add deploy/bluevela/bsub_1b_poc.sh
git commit -m "feat(deploy): add BlueVela LSF job submission script for 1B PoC"
```

---

### Task 5: Create granite-build placeholder directory

**Files:**
- Create: `deploy/granite-build/.gitkeep`

- [ ] **Step 1: Create the placeholder**

```bash
mkdir -p deploy/granite-build
touch deploy/granite-build/.gitkeep
```

- [ ] **Step 2: Commit**

```bash
git add deploy/granite-build/.gitkeep
git commit -m "chore(deploy): add granite-build placeholder directory"
```

---

### Task 6: Integration smoke test

Run a quick local validation that the training script initializes correctly without GPUs or ClearML credentials.

**Files:** None (read-only testing)

- [ ] **Step 1: Verify script loads and parses config correctly**

Run:
```bash
source .venv/bin/activate
python -c "
import sys
sys.path.insert(0, '.')
from open_mythos.variants import mythos_1b
from open_mythos.tokenizer import MythosTokenizer

cfg = mythos_1b()
enc = MythosTokenizer()
cfg.vocab_size = enc.vocab_size
cfg.max_seq_len = 2048

print(f'Config: dim={cfg.dim}, heads={cfg.n_heads}, experts={cfg.n_experts}')
print(f'Vocab: {cfg.vocab_size}')
print(f'Seq len: {cfg.max_seq_len}')
print('Smoke test passed.')
"
```
Expected: Config printed, `Smoke test passed.`

- [ ] **Step 2: Verify ClearML import works (without credentials)**

Run:
```bash
source .venv/bin/activate
pip install clearml>=1.16.0 -q
python -c "from clearml import Task; print('ClearML import OK')"
```
Expected: `ClearML import OK`

- [ ] **Step 3: Verify all bash scripts parse**

Run:
```bash
bash -n deploy/bluevela/setup_env.sh && bash -n deploy/bluevela/bsub_1b_poc.sh && echo "All scripts OK"
```
Expected: `All scripts OK`

- [ ] **Step 4: Run existing tests to verify no regressions**

Run: `pytest test_main.py -v -x --timeout=120`
Expected: All tests pass.

---

### Task 7: Final commit and push

- [ ] **Step 1: Check status**

Run: `git status`
Expected: Clean working tree (all changes committed in previous tasks).

- [ ] **Step 2: Push to origin**

Run: `git push origin main`
Expected: Push succeeds to `ssh://git@github.ibm.com/pzerfos/OpenMythos.git`. Do NOT push to upstream.
