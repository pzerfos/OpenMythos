#!/usr/bin/env python3
"""
Evaluate an OpenMythos checkpoint: qualitative generation + depth extrapolation sweep.

Single GPU (no FSDP needed):
    python evaluations/eval_checkpoint.py \
        --checkpoint /proj/checkpoints/pzerfos/openmythos/checkpoints/step_0031000.pt \
        --dataset-path /proj/datasets/pzerfos/fineweb-edu-100B/sample/100BT \
        --device cuda

Depth sweep only (skip generation):
    python evaluations/eval_checkpoint.py --checkpoint ... --skip-generation

Generation only (skip depth sweep):
    python evaluations/eval_checkpoint.py --checkpoint ... --skip-depth-sweep
"""

import argparse
import glob as _glob
import math
import os
import time

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from open_mythos import OpenMythos
from open_mythos.main import MythosConfig
from open_mythos.tokenizer import MythosTokenizer


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_model_from_checkpoint(
    path: str, device: str
) -> tuple[OpenMythos, MythosConfig, int, int]:
    """Load a trained OpenMythos model from a checkpoint file.

    Returns (model, cfg, step, vocab_size).
    """
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    cfg = ckpt["cfg"]
    vocab_size = ckpt["vocab_size"]
    step = ckpt["step"]

    model = OpenMythos(cfg)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Step: {step:,}")
    print(f"  Vocab: {vocab_size:,}")
    print(f"  Params: {n_params:,}")
    print(f"  Config: dim={cfg.dim}, n_heads={cfg.n_heads}, n_experts={cfg.n_experts}, "
          f"max_loop_iters={cfg.max_loop_iters}, attn_type={cfg.attn_type}")
    print(f"  Device: {device}")

    return model, cfg, step, vocab_size


# ---------------------------------------------------------------------------
# Qualitative generation
# ---------------------------------------------------------------------------

GENERATION_PROMPTS = [
    # Factual / knowledge
    "The theory of general relativity describes",
    "Photosynthesis is the process by which",
    "The French Revolution began in",
    # Reasoning / explanation
    "The reason why the sky appears blue is",
    "To solve a quadratic equation, you can",
    # Creative / open-ended
    "Once upon a time, in a kingdom far away,",
    "The most surprising thing about the ocean is",
    # Technical / educational (FineWeb-Edu domain)
    "Machine learning algorithms can be broadly classified into",
    "The water cycle consists of several stages:",
    # Short-form completion
    "1 + 1 =",
]


def run_generation(
    model: OpenMythos,
    tokenizer: MythosTokenizer,
    device: str,
    max_new_tokens: int = 256,
    n_loops: int = 16,
) -> None:
    """Generate text from prompts and print results."""
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"GENERATION SAMPLES (n_loops={n_loops}, max_new_tokens={max_new_tokens})")
    print(bar)

    for i, prompt in enumerate(GENERATION_PROMPTS, 1):
        input_ids = torch.tensor(
            [tokenizer.encode(prompt)], dtype=torch.long, device=device
        )

        # Sampling generation
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                n_loops=n_loops,
                temperature=0.8,
                top_k=50,
            )
        generated = tokenizer.decode(output_ids[0].tolist())

        print(f"\n--- Sample {i}/{len(GENERATION_PROMPTS)} ---")
        print(f"Prompt:    {prompt}")
        print(f"Generated: {generated}")

    # One greedy sample for comparison
    print(f"\n--- Greedy decode (temperature=0) ---")
    prompt = GENERATION_PROMPTS[0]
    input_ids = torch.tensor(
        [tokenizer.encode(prompt)], dtype=torch.long, device=device
    )
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            n_loops=n_loops,
            temperature=0.0,
            top_k=50,
        )
    print(f"Prompt:    {prompt}")
    print(f"Generated: {tokenizer.decode(output_ids[0].tolist())}")


# ---------------------------------------------------------------------------
# Held-out eval dataset from local parquet
# ---------------------------------------------------------------------------


def build_eval_dataset(
    dataset_path: str,
    tokenizer: MythosTokenizer,
    seq_len: int,
    max_tokens: int,
    n_eval_files: int = 2,
) -> TensorDataset:
    """Build a held-out eval dataset from the last N parquet files.

    Uses the last files alphabetically so they don't overlap with training
    data (training iterates files in order via round-robin).
    """
    files = sorted(_glob.glob(os.path.join(dataset_path, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {dataset_path}")

    eval_files = files[-n_eval_files:]
    print(f"Eval files ({n_eval_files} of {len(files)}): {[os.path.basename(f) for f in eval_files]}")

    all_ids: list[int] = []
    for fp in eval_files:
        table = pq.read_table(fp, columns=["text"])
        for text in table.column("text").to_pylist():
            if not text:
                continue
            all_ids.extend(tokenizer.encode(text))
            if len(all_ids) >= max_tokens:
                break
        if len(all_ids) >= max_tokens:
            break

    all_ids = all_ids[:max_tokens]
    print(f"Eval tokens: {len(all_ids):,}")

    data = torch.tensor(all_ids, dtype=torch.long)
    n_pairs = (len(data) - 1) // seq_len
    data = data[: n_pairs * seq_len + 1]
    x = data[:-1].view(n_pairs, seq_len)
    y = data[1:].view(n_pairs, seq_len)
    return TensorDataset(x, y)


# ---------------------------------------------------------------------------
# Depth extrapolation sweep
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_loss(
    model: OpenMythos,
    loader: DataLoader,
    vocab_size: int,
    device: str,
    n_loops: int | None = None,
    max_batches: int | None = None,
) -> float:
    """Mean cross-entropy loss over the eval dataset at a given n_loops."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x, n_loops=n_loops)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size), y.view(-1), reduction="sum"
        )
        total_loss += loss.item()
        total_tokens += y.numel()
    return total_loss / max(1, total_tokens)


def run_depth_sweep(
    model: OpenMythos,
    cfg: MythosConfig,
    loader: DataLoader,
    vocab_size: int,
    device: str,
    sweep_values: list[int],
    max_batches: int | None = None,
) -> None:
    """Evaluate at multiple n_loops values and print a results table."""
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"DEPTH EXTRAPOLATION SWEEP (trained at n_loops={cfg.max_loop_iters})")
    print(bar)

    results: list[tuple[int, float, float]] = []
    trained_loss = None

    for nl in sweep_values:
        t0 = time.perf_counter()
        loss = evaluate_loss(model, loader, vocab_size, device, n_loops=nl, max_batches=max_batches)
        dt = time.perf_counter() - t0
        ppl = math.exp(min(loss, 20))  # cap to avoid overflow
        results.append((nl, loss, ppl))
        if nl == cfg.max_loop_iters:
            trained_loss = loss
        print(f"  n_loops={nl:>3}  |  loss={loss:.4f}  |  PPL={ppl:>10.2f}  |  {dt:.1f}s")

    # Print summary with deltas
    print(f"\n{'n_loops':>8} | {'loss':>8} | {'PPL':>10} | {'vs trained':>10}")
    print("-" * 45)
    for nl, loss, ppl in results:
        delta = f"{loss - trained_loss:+.4f}" if trained_loss is not None else "N/A"
        marker = " <-- trained" if nl == cfg.max_loop_iters else ""
        print(f"{nl:>8} | {loss:>8.4f} | {ppl:>10.2f} | {delta:>10}{marker}")

    # Diagnosis
    losses = [l for _, l, _ in results]
    min_idx = losses.index(min(losses))
    best_nl = results[min_idx][0]
    print(f"\nBest n_loops: {best_nl} (loss={results[min_idx][1]:.4f})")
    if best_nl > cfg.max_loop_iters:
        print("  --> Model benefits from MORE depth than trained — depth extrapolation works!")
    elif best_nl == cfg.max_loop_iters:
        print("  --> Optimal at training depth — no extrapolation benefit.")
    else:
        print("  --> Optimal BELOW training depth — model may be over-iterating.")

    # Check for U-shape (expected with ACT)
    if min_idx > 0 and min_idx < len(losses) - 1:
        if losses[0] > losses[min_idx] < losses[-1]:
            print("  --> U-shaped curve detected: loss increases at both low and high depth.")
            print("     This is consistent with ACT depth-binding (see issue #5).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an OpenMythos checkpoint with generation + depth sweep."
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to checkpoint .pt file",
    )
    parser.add_argument(
        "--dataset-path",
        default="/proj/datasets/pzerfos/fineweb-edu-100B/sample/100BT",
        help="Path to directory of parquet files for held-out eval",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--eval-tokens", type=int, default=500_000)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument(
        "--depth-sweep", default="1,2,4,8,12,16,24,32",
        help="Comma-separated n_loops values for depth extrapolation sweep",
    )
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-depth-sweep", action="store_true")
    args = parser.parse_args()

    # Load model
    model, cfg, step, vocab_size = load_model_from_checkpoint(args.checkpoint, args.device)
    tokenizer = MythosTokenizer()
    seq_len = cfg.max_seq_len

    # Generation
    if not args.skip_generation:
        run_generation(model, tokenizer, args.device, args.max_new_tokens, cfg.max_loop_iters)

    # Depth sweep
    if not args.skip_depth_sweep:
        eval_ds = build_eval_dataset(
            args.dataset_path, tokenizer, seq_len, args.eval_tokens,
        )
        eval_loader = DataLoader(
            eval_ds, batch_size=args.eval_batch_size, shuffle=False, drop_last=False,
        )
        sweep_values = sorted(int(s) for s in args.depth_sweep.split(",") if s.strip())
        run_depth_sweep(
            model, cfg, eval_loader, vocab_size, args.device, sweep_values, args.max_eval_batches,
        )


if __name__ == "__main__":
    main()
