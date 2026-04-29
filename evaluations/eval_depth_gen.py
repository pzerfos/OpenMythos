"""Depth generalization sweep for the NoPE ablation.

Loads a trained checkpoint and evaluates perplexity at fixed seq_len=2048 with
n_loops swept over {16, 32, 48, 64, 96}. Training max is 32; {48, 64, 96} are
pure depth extrapolation. The publication-worthy axis of the study.

See docs/superpowers/specs/2026-04-29-nope-for-recurrent-depth-design.md §6.3.
"""

import argparse
import math

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from loguru import logger

from open_mythos.main import OpenMythos
from open_mythos.tokenizer import MythosTokenizer
from open_mythos.variants import (
    mythos_1b,
    mythos_1b_partial_nope,
    mythos_1b_scoped_nope,
)

VARIANT_TO_CFG = {
    "baseline": mythos_1b,
    "scoped": mythos_1b_scoped_nope,
    "partial": mythos_1b_partial_nope,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--variant", required=True, choices=["baseline", "scoped", "partial"]
    )
    p.add_argument("--n-sequences", type=int, default=100)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--n-loops-sweep", type=int, nargs="+", default=[16, 32, 48, 64, 96])
    p.add_argument(
        "--held-out-shard",
        default="/proj/datasets/pzerfos/fineweb-edu-100B/sample/100BT/000_00099.parquet",
    )
    return p.parse_args()


@torch.no_grad()
def nll_per_token(model, input_ids: torch.Tensor, n_loops: int) -> float:
    logits = model(input_ids[:, :-1], n_loops=n_loops, bypass_act=True)
    targets = input_ids[:, 1:]
    nll = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="mean",
    )
    return nll.item()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = VARIANT_TO_CFG[args.variant]()
    cfg.max_seq_len = args.seq_len

    logger.info(f"loading checkpoint {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    tokenizer = MythosTokenizer()
    cfg.vocab_size = tokenizer.vocab_size

    model = OpenMythos(cfg)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    # Load held-out sequences
    table = pq.read_table(args.held_out_shard, columns=["text"])
    texts = table.column("text").to_pylist()[: args.n_sequences]

    # Pre-tokenize once
    token_batches = []
    for text in texts:
        tokens = tokenizer.encode(text)[: args.seq_len]
        if len(tokens) < 64:
            continue
        token_batches.append(
            torch.tensor(tokens, dtype=torch.int64, device=device).unsqueeze(0)
        )
    logger.info(f"evaluating on {len(token_batches)} held-out sequences")

    for n_loops in args.n_loops_sweep:
        nlls = [nll_per_token(model, batch, n_loops) for batch in token_batches]
        mean_nll = sum(nlls) / len(nlls)
        ppl = math.exp(mean_nll)
        logger.info(
            f"RESULT variant={args.variant} n_loops={n_loops} "
            f"n_sequences={len(nlls)} mean_nll={mean_nll:.4f} ppl={ppl:.2f}"
        )


if __name__ == "__main__":
    main()
