"""Length generalization sweep for the NoPE ablation.

Loads a trained checkpoint (baseline / scoped / partial) and evaluates
perplexity on a held-out FineWeb-Edu shard at seq_len in {2048, 4096, 8192, 16384}.
For lengths > the training max_seq_len, regenerates freqs_cis on-the-fly.

See docs/superpowers/specs/2026-04-29-nope-for-recurrent-depth-design.md §6.2.
"""

import argparse
import math

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from loguru import logger

from open_mythos.main import OpenMythos, precompute_rope_freqs
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
    p.add_argument("--checkpoint", required=True, help="path to step_NNNNNNN.pt")
    p.add_argument(
        "--variant",
        required=True,
        choices=["baseline", "scoped", "partial"],
        help="must match the variant the checkpoint was trained with",
    )
    p.add_argument("--n-sequences", type=int, default=100)
    p.add_argument(
        "--seq-lengths", type=int, nargs="+", default=[2048, 4096, 8192, 16384]
    )
    p.add_argument(
        "--held-out-shard",
        default="/proj/datasets/pzerfos/fineweb-edu-100B/sample/100BT/000_00099.parquet",
        help="parquet shard not seen during training",
    )
    p.add_argument("--n-loops", type=int, default=16)
    return p.parse_args()


def _regenerate_freqs(model: OpenMythos, max_len: int, device: torch.device) -> None:
    """Regenerate the precomputed RoPE frequencies for a target sequence length."""
    cfg = model.cfg
    head_dim = cfg.dim // cfg.n_heads
    model.freqs_cis = precompute_rope_freqs(head_dim, max_len, cfg.rope_theta).to(device)
    model.freqs_cis_mla = precompute_rope_freqs(
        cfg.qk_rope_head_dim, max_len, cfg.rope_theta
    ).to(device)


@torch.no_grad()
def evaluate_at_length(model, input_ids: torch.Tensor, n_loops: int) -> float:
    """Returns NLL per predicted token across the batch."""
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
    # Enlarge max_seq_len for the longest eval length
    cfg.max_seq_len = max(args.seq_lengths)

    logger.info(f"loading checkpoint {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    tokenizer = MythosTokenizer()
    cfg.vocab_size = tokenizer.vocab_size

    model = OpenMythos(cfg)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    _regenerate_freqs(model, cfg.max_seq_len, device)

    # Load held-out sequences
    logger.info(f"loading held-out shard: {args.held_out_shard}")
    table = pq.read_table(args.held_out_shard, columns=["text"])
    texts = table.column("text").to_pylist()[: args.n_sequences]

    for seq_len in args.seq_lengths:
        nlls = []
        for text in texts:
            tokens = tokenizer.encode(text)[:seq_len]
            if len(tokens) < 64:
                continue
            input_ids = torch.tensor(tokens, dtype=torch.int64, device=device).unsqueeze(0)
            nll = evaluate_at_length(model, input_ids, args.n_loops)
            nlls.append(nll)
        mean_nll = sum(nlls) / len(nlls)
        ppl = math.exp(mean_nll)
        logger.info(
            f"RESULT variant={args.variant} seq_len={seq_len} "
            f"n_sequences={len(nlls)} mean_nll={mean_nll:.4f} ppl={ppl:.2f}"
        )


if __name__ == "__main__":
    main()
