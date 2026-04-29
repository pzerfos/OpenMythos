"""Tests for the NoPE ablation: pe_mode plumbing, variant configs, cross-mode
checkpoint compatibility, and KV-cache correctness.

See docs/superpowers/specs/2026-04-29-nope-for-recurrent-depth-design.md.
"""

import torch

from open_mythos.main import MythosConfig


def test_mythos_config_pe_mode_defaults_to_rope():
    cfg = MythosConfig()
    assert cfg.pe_mode_prelude == "rope"
    assert cfg.pe_mode_coda == "rope"
    assert cfg.pe_mode_recurrent == "rope"


def test_mythos_config_pe_mode_is_settable():
    cfg = MythosConfig(pe_mode_recurrent="nope")
    assert cfg.pe_mode_recurrent == "nope"
    assert cfg.pe_mode_prelude == "rope"
    assert cfg.pe_mode_coda == "rope"


from open_mythos.main import GQAttention, MLAttention, precompute_rope_freqs


def _gqa_test_cfg() -> MythosConfig:
    return MythosConfig(
        vocab_size=256,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        attn_type="gqa",
        dropout=0.0,
    )


def test_gqa_nope_matches_rope_at_position_0():
    """At position 0, RoPE rotates by angle 0 (identity). NoPE and RoPE outputs
    must be identical for a single-token input at position 0."""
    torch.manual_seed(0)
    cfg = _gqa_test_cfg()
    attn = GQAttention(cfg)
    attn.eval()
    head_dim = cfg.dim // cfg.n_heads
    freqs_cis_full = precompute_rope_freqs(head_dim, cfg.max_seq_len, 500000.0)
    freqs_cis = freqs_cis_full[:1]  # position 0 only
    x = torch.randn(1, 1, cfg.dim)
    with torch.no_grad():
        out_rope = attn(x, freqs_cis, pe_mode="rope")
        out_nope = attn(x, freqs_cis, pe_mode="nope")
    assert torch.allclose(out_rope, out_nope, atol=1e-5)


def test_gqa_nope_differs_from_rope_at_later_positions():
    """For sequences of length > 1, NoPE and RoPE outputs must differ at
    positions > 0 (otherwise pe_mode wasn't actually plumbed through)."""
    torch.manual_seed(0)
    cfg = _gqa_test_cfg()
    attn = GQAttention(cfg)
    attn.eval()
    head_dim = cfg.dim // cfg.n_heads
    freqs_cis_full = precompute_rope_freqs(head_dim, cfg.max_seq_len, 500000.0)
    T = 6
    freqs_cis = freqs_cis_full[:T]
    x = torch.randn(1, T, cfg.dim)
    with torch.no_grad():
        out_rope = attn(x, freqs_cis, pe_mode="rope")
        out_nope = attn(x, freqs_cis, pe_mode="nope")
    # Positions 1+ must differ because RoPE rotates keys at those positions.
    assert not torch.allclose(out_rope[:, 1:], out_nope[:, 1:], atol=1e-3)


def _mla_test_cfg() -> MythosConfig:
    return MythosConfig(
        vocab_size=256,
        dim=128,
        n_heads=4,
        n_kv_heads=4,
        max_seq_len=32,
        attn_type="mla",
        kv_lora_rank=32,
        q_lora_rank=64,
        qk_rope_head_dim=8,
        qk_nope_head_dim=16,
        v_head_dim=16,
        dropout=0.0,
    )


def test_mla_nope_matches_rope_at_position_0():
    torch.manual_seed(0)
    cfg = _mla_test_cfg()
    attn = MLAttention(cfg)
    attn.eval()
    freqs_cis_full = precompute_rope_freqs(
        cfg.qk_rope_head_dim, cfg.max_seq_len, 500000.0
    )
    freqs_cis = freqs_cis_full[:1]
    x = torch.randn(1, 1, cfg.dim)
    with torch.no_grad():
        out_rope = attn(x, freqs_cis, pe_mode="rope")
        out_nope = attn(x, freqs_cis, pe_mode="nope")
    assert torch.allclose(out_rope, out_nope, atol=1e-5)


def test_mla_nope_differs_from_rope_at_later_positions():
    torch.manual_seed(0)
    cfg = _mla_test_cfg()
    attn = MLAttention(cfg)
    attn.eval()
    freqs_cis_full = precompute_rope_freqs(
        cfg.qk_rope_head_dim, cfg.max_seq_len, 500000.0
    )
    T = 6
    freqs_cis = freqs_cis_full[:T]
    x = torch.randn(1, T, cfg.dim)
    with torch.no_grad():
        out_rope = attn(x, freqs_cis, pe_mode="rope")
        out_nope = attn(x, freqs_cis, pe_mode="nope")
    # Positions 1+ must differ because RoPE rotates keys at those positions.
    assert not torch.allclose(out_rope[:, 1:], out_nope[:, 1:], atol=1e-3)


def test_mla_partial_nope_zero_rope_dim():
    """Partial NoPE variant: qk_rope_head_dim=0 puts all budget on qk_nope_head_dim.
    Forward must run without shape errors and produce finite output."""
    cfg = _mla_test_cfg()
    cfg.qk_nope_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
    cfg.qk_rope_head_dim = 0
    attn = MLAttention(cfg)
    attn.eval()
    # freqs_cis has rope_dim//2 = 0 — need a zero-shape tensor
    freqs_cis = torch.zeros(6, 0, dtype=torch.complex64)
    x = torch.randn(1, 6, cfg.dim)
    with torch.no_grad():
        out = attn(x, freqs_cis, pe_mode="rope")  # rope_dim=0 makes rope a no-op
    assert torch.isfinite(out).all()
    assert out.shape == (1, 6, cfg.dim)
