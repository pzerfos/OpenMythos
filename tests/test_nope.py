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


from open_mythos.main import GQAttention, MLAttention, TransformerBlock, precompute_rope_freqs


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


def test_transformer_block_plumbs_pe_mode():
    """TransformerBlock.forward should forward pe_mode to its inner attention."""
    torch.manual_seed(0)
    cfg = _mla_test_cfg()
    block = TransformerBlock(cfg, use_moe=False)
    block.eval()
    freqs_cis = precompute_rope_freqs(
        cfg.qk_rope_head_dim, cfg.max_seq_len, 500000.0
    )[:6]
    x = torch.randn(1, 6, cfg.dim)
    with torch.no_grad():
        out_rope = block(x, freqs_cis, pe_mode="rope")
        out_nope = block(x, freqs_cis, pe_mode="nope")
    assert out_rope.shape == (1, 6, cfg.dim)
    assert not torch.allclose(out_rope[:, 1:], out_nope[:, 1:], atol=1e-3)


from open_mythos.main import OpenMythos, RecurrentBlock


def test_recurrent_block_pe_mode_stored_and_used():
    """RecurrentBlock should read cfg.pe_mode_recurrent at init and use it
    when invoking its inner TransformerBlock on each loop iteration."""
    cfg = _mla_test_cfg()
    cfg.pe_mode_recurrent = "nope"
    cfg.max_loop_iters = 2
    cfg.lora_rank = 4
    cfg.n_experts = 2
    cfg.n_shared_experts = 1
    cfg.n_experts_per_tok = 1
    cfg.expert_dim = 32

    torch.manual_seed(0)
    rec = RecurrentBlock(cfg)
    rec.eval()
    freqs_cis = precompute_rope_freqs(
        cfg.qk_rope_head_dim, cfg.max_seq_len, 500000.0
    )[:6]
    h = torch.randn(1, 6, cfg.dim)
    e = torch.randn(1, 6, cfg.dim)
    with torch.no_grad():
        out_nope = rec(h, e, freqs_cis, n_loops=2, bypass_act=True)

    cfg.pe_mode_recurrent = "rope"
    torch.manual_seed(0)
    rec2 = RecurrentBlock(cfg)
    rec2.eval()
    rec2.load_state_dict(rec.state_dict())
    with torch.no_grad():
        out_rope = rec2(h, e, freqs_cis, n_loops=2, bypass_act=True)

    # Same weights, same input, different pe_mode → different output at later positions
    assert not torch.allclose(out_rope[:, 1:], out_nope[:, 1:], atol=1e-3)


def _tiny_mythos_cfg() -> MythosConfig:
    cfg = _mla_test_cfg()
    cfg.max_loop_iters = 2
    cfg.prelude_layers = 1
    cfg.coda_layers = 1
    cfg.lora_rank = 4
    cfg.n_experts = 2
    cfg.n_shared_experts = 1
    cfg.n_experts_per_tok = 1
    cfg.expert_dim = 32
    return cfg


def test_openmythos_routes_pe_mode_prelude():
    """Setting pe_mode_prelude='nope' must change model output — demonstrates
    the prelude loop in OpenMythos.forward propagates pe_mode_prelude to the
    TransformerBlock call."""
    cfg_all_rope = _tiny_mythos_cfg()
    cfg_nope_prelude = _tiny_mythos_cfg()
    cfg_nope_prelude.pe_mode_prelude = "nope"

    torch.manual_seed(0)
    m1 = OpenMythos(cfg_all_rope)
    m1.eval()

    torch.manual_seed(1)
    m2 = OpenMythos(cfg_nope_prelude)
    m2.eval()
    m2.load_state_dict(m1.state_dict())

    input_ids = torch.randint(0, cfg_all_rope.vocab_size, (1, 6))
    with torch.no_grad():
        out_all_rope = m1(input_ids, n_loops=2, bypass_act=True)
        out_nope_prelude = m2(input_ids, n_loops=2, bypass_act=True)

    # Same weights, same input — different prelude pe_mode must change logits
    assert not torch.allclose(out_all_rope, out_nope_prelude, atol=1e-3)


def test_openmythos_routes_pe_mode_coda():
    """Analogous test for pe_mode_coda."""
    cfg_all_rope = _tiny_mythos_cfg()
    cfg_nope_coda = _tiny_mythos_cfg()
    cfg_nope_coda.pe_mode_coda = "nope"

    torch.manual_seed(0)
    m1 = OpenMythos(cfg_all_rope)
    m1.eval()

    torch.manual_seed(1)
    m2 = OpenMythos(cfg_nope_coda)
    m2.eval()
    m2.load_state_dict(m1.state_dict())

    input_ids = torch.randint(0, cfg_all_rope.vocab_size, (1, 6))
    with torch.no_grad():
        out_all_rope = m1(input_ids, n_loops=2, bypass_act=True)
        out_nope_coda = m2(input_ids, n_loops=2, bypass_act=True)

    # atol=1e-6: the coda NoPE effect is small (1 layer, then norm+head compresses
    # it to ~6e-6 max diff), but must be non-zero — use tighter tolerance than prelude.
    assert not torch.allclose(out_all_rope, out_nope_coda, atol=1e-6)


def test_state_dict_round_trip_across_pe_modes():
    """The pe_mode flags are not parameters/buffers, so state_dict keys and
    shapes must be identical across all three variants. Loading a baseline
    checkpoint into a scoped-NoPE model (and vice versa) must succeed with
    strict=True."""
    cfg_baseline = _tiny_mythos_cfg()
    cfg_scoped = _tiny_mythos_cfg()
    cfg_scoped.pe_mode_recurrent = "nope"

    torch.manual_seed(0)
    m_baseline = OpenMythos(cfg_baseline)

    torch.manual_seed(1)  # different init
    m_scoped = OpenMythos(cfg_scoped)

    # Load baseline state_dict into the scoped-NoPE model with strict=True
    m_scoped.load_state_dict(m_baseline.state_dict(), strict=True)

    # Keys match exactly
    assert set(m_baseline.state_dict().keys()) == set(m_scoped.state_dict().keys())

    # Shapes match exactly
    for k, v in m_baseline.state_dict().items():
        assert v.shape == m_scoped.state_dict()[k].shape, k
