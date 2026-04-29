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
