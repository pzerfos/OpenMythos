# NoPE Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the 3-variant NoPE ablation (baseline RoPE / Scoped NoPE / Partial NoPE via MLA) described in `docs/superpowers/specs/2026-04-29-nope-for-recurrent-depth-design.md`. Produce runnable training + eval artifacts on a dedicated branch so the three 1B-scale training jobs can be launched on or after 2026-05-01 when BlueVela load permits.

**Architecture:** Add a per-site positional-encoding mode (`pe_mode: Literal["rope","nope"]`) to `MythosConfig` and plumb it through `MLAttention` / `GQAttention` / `TransformerBlock` / `RecurrentBlock` / `OpenMythos.__init__`. Partial NoPE is config-only (`qk_rope_head_dim=0`). Submission-side: a single `NOPE_VARIANT` env knob in `training/1b_poc_fineweb.py` picks the config builder, ClearML task name, and checkpoint dir; a bsub wrapper submits all three variants as independent 4-GPU jobs.

**Tech Stack:** Python 3.10, PyTorch 2.11, FSDP1 (existing), FineWeb-Edu dataset, ClearML, BlueVela LSF, `pytest` for unit tests.

---

## File Structure

**Files to modify:**
- `open_mythos/main.py` — `MythosConfig` fields; `pe_mode` plumbing through `GQAttention`, `MLAttention`, `TransformerBlock`, `RecurrentBlock`, `OpenMythos`.
- `open_mythos/variants.py` — add `mythos_1b_scoped_nope()`, `mythos_1b_partial_nope()`.
- `training/1b_poc_fineweb.py` — read `NOPE_VARIANT` env; variant-aware ClearML task name + checkpoint subdir.
- `docs/logbook/2026-04-28-option-b-and-upstream-pr.md` — add NoPE ablation as an open item; soften FSDP2 item language (evaluation completed, decision pending; not "queued"/"decided").

**Files to create:**
- `tests/test_nope.py` — all NoPE unit tests.
- `deploy/bluevela/bsub_nope_ablation.sh` — submits the 3 variants to LSF.
- `deploy/bluevela/run_nope_ablation.sh` — inner runner invoked by bsub (same pattern as `run_fsdp_bench.sh` from the FSDP2 feasibility work — avoids LSF single-quote wrapping bugs).
- `evaluations/eval_length_gen.py` — post-training length generalization sweep.
- `evaluations/eval_depth_gen.py` — post-training depth generalization sweep.
- `docs/logbook/2026-04-29-nope-ablation-queued.md` — companion logbook entry marking the ablation as queued for 2026-05-01.

**Branch:** `feat/nope-ablation` (off `main` at current HEAD `f377271` after spec commit).

---

## Task 1: Create the feature branch

**Files:**
- No file changes. Branch creation only.

- [ ] **Step 1: Verify clean working tree on main**

Run: `git status`
Expected: `nothing to commit, working tree clean` on branch `main`.

- [ ] **Step 2: Create and switch to feature branch**

```bash
git checkout -b feat/nope-ablation
```

- [ ] **Step 3: Confirm branch**

Run: `git log --oneline -1`
Expected: latest commit is `f377271 docs(spec): NoPE for recurrent-depth design spec`.

---

## Task 2: Add `pe_mode_*` fields to `MythosConfig`

**Files:**
- Modify: `open_mythos/main.py` (the `MythosConfig` dataclass around line 17–82)
- Test: `tests/test_nope.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `tests/test_nope.py` with this content:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_nope.py -v`
Expected: FAIL with `AttributeError: 'MythosConfig' object has no attribute 'pe_mode_prelude'`.

- [ ] **Step 3: Add the three fields to `MythosConfig`**

In `open_mythos/main.py`, inside the `MythosConfig` dataclass (below `dropout: float = 0.0`, before the closing of the dataclass), append:

```python
    # Positional encoding mode per site. "rope" uses apply_rope as before;
    # "nope" skips the rotation (and relies on the causal mask for implicit
    # position — see Kazemnejad 2023, arxiv 2305.19466). These knobs allow
    # the Scoped NoPE variant (nope in recurrent block only) without
    # duplicating attention classes.
    pe_mode_prelude: str = "rope"
    pe_mode_coda: str = "rope"
    pe_mode_recurrent: str = "rope"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_nope.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add open_mythos/main.py tests/test_nope.py
git commit -m "feat(model): add pe_mode_{prelude,coda,recurrent} to MythosConfig"
```

---

## Task 3: Plumb `pe_mode` through `GQAttention.forward`

**Files:**
- Modify: `open_mythos/main.py` (`GQAttention.forward`, around lines 213–280)
- Test: `tests/test_nope.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_nope.py`:

```python
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
    # Position 0 still identical; later positions must differ
    assert torch.allclose(out_rope[:, 0], out_nope[:, 0], atol=1e-5)
    assert not torch.allclose(out_rope[:, 1:], out_nope[:, 1:], atol=1e-3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_nope.py -v -k "gqa"`
Expected: FAIL with `TypeError: forward() got an unexpected keyword argument 'pe_mode'`.

- [ ] **Step 3: Modify `GQAttention.forward` to accept and use `pe_mode`**

In `open_mythos/main.py`, change the signature of `GQAttention.forward` (currently at line 213) to add a `pe_mode: str = "rope"` kwarg:

```python
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
        pe_mode: str = "rope",
    ) -> torch.Tensor:
```

Replace the two `apply_rope` calls (currently lines 238–239):

```python
        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)
```

with:

```python
        if pe_mode == "rope":
            q = apply_rope(q, freqs_cis)
            k = apply_rope(k, freqs_cis)
        elif pe_mode != "nope":
            raise ValueError(f"pe_mode must be 'rope' or 'nope', got {pe_mode!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_nope.py -v -k "gqa"`
Expected: both gqa tests PASS.

- [ ] **Step 5: Commit**

```bash
git add open_mythos/main.py tests/test_nope.py
git commit -m "feat(model): plumb pe_mode through GQAttention.forward"
```

---

## Task 4: Plumb `pe_mode` through `MLAttention.forward`

**Files:**
- Modify: `open_mythos/main.py` (`MLAttention.forward`, around lines 354–423)
- Test: `tests/test_nope.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_nope.py`:

```python
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
    assert torch.allclose(out_rope[:, 0], out_nope[:, 0], atol=1e-5)
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_nope.py -v -k "mla"`
Expected: FAIL with `TypeError: forward() got an unexpected keyword argument 'pe_mode'`.

- [ ] **Step 3: Modify `MLAttention.forward`**

Change the signature (currently at line 354) to add `pe_mode: str = "rope"`:

```python
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
        pe_mode: str = "rope",
    ) -> torch.Tensor:
```

Replace the two `apply_rope` calls (currently lines 380 and 394):

```python
        q_rope = apply_rope(q_rope, freqs_cis)
```

becomes:

```python
        if pe_mode == "rope":
            q_rope = apply_rope(q_rope, freqs_cis)
        elif pe_mode != "nope":
            raise ValueError(f"pe_mode must be 'rope' or 'nope', got {pe_mode!r}")
```

And:

```python
        k_rope = apply_rope(k_rope, freqs_cis)  # (B, T, H, rope_dim) ← cached
```

becomes:

```python
        if pe_mode == "rope":
            k_rope = apply_rope(k_rope, freqs_cis)  # (B, T, H, rope_dim) ← cached
        # Under NoPE, k_rope is already the unrotated expanded tensor; concat as-is.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_nope.py -v -k "mla"`
Expected: all mla tests PASS.

- [ ] **Step 5: Commit**

```bash
git add open_mythos/main.py tests/test_nope.py
git commit -m "feat(model): plumb pe_mode through MLAttention.forward"
```

---

## Task 5: Plumb `pe_mode` through `TransformerBlock.forward`

**Files:**
- Modify: `open_mythos/main.py` (`TransformerBlock.forward`, around lines 752–775)
- Test: `tests/test_nope.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_nope.py`:

```python
from open_mythos.main import TransformerBlock


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_nope.py -v -k "transformer_block"`
Expected: FAIL with `TypeError: forward() got an unexpected keyword argument 'pe_mode'`.

- [ ] **Step 3: Modify `TransformerBlock.forward`**

Change the signature (currently at line 752) to add `pe_mode: str = "rope"`:

```python
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
        pe_mode: str = "rope",
    ) -> torch.Tensor:
```

Replace the `self.attn(...)` call (currently line 772):

```python
            self.attn(self.attn_norm(x), freqs_cis, mask, kv_cache, cache_key)
```

with:

```python
            self.attn(
                self.attn_norm(x), freqs_cis, mask, kv_cache, cache_key, pe_mode
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_nope.py -v -k "transformer_block"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add open_mythos/main.py tests/test_nope.py
git commit -m "feat(model): plumb pe_mode through TransformerBlock.forward"
```

---

## Task 6: Plumb `pe_mode_recurrent` through `RecurrentBlock`

**Files:**
- Modify: `open_mythos/main.py` (`RecurrentBlock.__init__` around line 909; `RecurrentBlock.forward` around line 925; the `self.block(...)` call on line 967)
- Test: `tests/test_nope.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_nope.py`:

```python
from open_mythos.main import RecurrentBlock


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_nope.py -v -k "recurrent_block"`
Expected: FAIL — `RecurrentBlock` does not yet read `cfg.pe_mode_recurrent`.

- [ ] **Step 3: Modify `RecurrentBlock.__init__` to store the mode**

In `open_mythos/main.py`, in `RecurrentBlock.__init__` (currently at line 909), after `self.cfg = cfg` (line 915), add:

```python
        self.pe_mode = cfg.pe_mode_recurrent
```

- [ ] **Step 4: Modify `RecurrentBlock.forward` to pass the mode to the block**

In `open_mythos/main.py`, in `RecurrentBlock.forward` (currently at line 925), replace the `self.block(...)` call (currently line 967):

```python
            trans_out = self.block(combined, freqs_cis, mask, kv_cache, cache_key)
```

with:

```python
            trans_out = self.block(
                combined, freqs_cis, mask, kv_cache, cache_key, self.pe_mode
            )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_nope.py -v -k "recurrent_block"`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add open_mythos/main.py tests/test_nope.py
git commit -m "feat(model): plumb pe_mode_recurrent through RecurrentBlock"
```

---

## Task 7: Wire `pe_mode_{prelude,coda}` through `OpenMythos.forward`

**Files:**
- Modify: `open_mythos/main.py` (`OpenMythos.forward` around line 1127; specifically the two loops over `self.prelude` and `self.coda` around lines 1164 and 1170)
- Test: `tests/test_nope.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_nope.py`:

```python
from open_mythos.main import OpenMythos


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

    assert not torch.allclose(out_all_rope, out_nope_coda, atol=1e-3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_nope.py -v -k "routes_pe_mode"`
Expected: FAIL — the prelude and coda loops in `OpenMythos.forward` currently don't pass `pe_mode`, so `TransformerBlock.forward` defaults to `"rope"` regardless of `cfg.pe_mode_prelude` / `cfg.pe_mode_coda`.

- [ ] **Step 3: Update the prelude and coda loops to pass their pe_mode**

In `open_mythos/main.py`, `OpenMythos.forward` (around line 1164):

```python
        for i, layer in enumerate(self.prelude):
            x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"prelude_{i}")
```

becomes:

```python
        for i, layer in enumerate(self.prelude):
            x = layer(
                x, freqs_cis, mask, kv_cache,
                cache_key=f"prelude_{i}",
                pe_mode=self.cfg.pe_mode_prelude,
            )
```

Similarly for the coda loop (around line 1170):

```python
        for i, layer in enumerate(self.coda):
            x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"coda_{i}")
```

becomes:

```python
        for i, layer in enumerate(self.coda):
            x = layer(
                x, freqs_cis, mask, kv_cache,
                cache_key=f"coda_{i}",
                pe_mode=self.cfg.pe_mode_coda,
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_nope.py -v -k "routes_pe_mode"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add open_mythos/main.py tests/test_nope.py
git commit -m "feat(model): route pe_mode_{prelude,coda} through OpenMythos.forward"
```

---

## Task 8: Cross-mode state_dict round-trip test

**Files:**
- Test: `tests/test_nope.py`

- [ ] **Step 1: Write the test**

Append to `tests/test_nope.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_nope.py -v -k "state_dict_round_trip"`
Expected: PASS — no implementation changes needed, only the test.

- [ ] **Step 3: Commit**

```bash
git add tests/test_nope.py
git commit -m "test(nope): state_dict round-trip across pe_mode variants"
```

---

## Task 9: KV-cache correctness test

**Files:**
- Test: `tests/test_nope.py`

- [ ] **Step 1: Write the test**

Append to `tests/test_nope.py`:

```python
def test_kv_cache_prefill_decode_equals_one_shot_under_nope():
    """Under pe_mode='nope' in the recurrent block, incremental prefill+decode
    must produce the same logits as one-shot prefill over the full sequence.
    Failure would indicate the cache mixes rotated and unrotated keys."""
    cfg = _tiny_mythos_cfg()
    cfg.pe_mode_recurrent = "nope"

    torch.manual_seed(0)
    model = OpenMythos(cfg)
    model.eval()

    input_ids = torch.randint(0, cfg.vocab_size, (1, 6))

    # One-shot: prefill the whole sequence
    cache_oneshot: dict = {}
    with torch.no_grad():
        out_oneshot = model(
            input_ids, n_loops=2, kv_cache=cache_oneshot, bypass_act=True
        )

    # Incremental: prefill first 4 tokens, then decode tokens 5 and 6 one by one
    cache_incr: dict = {}
    with torch.no_grad():
        _ = model(
            input_ids[:, :4], n_loops=2, kv_cache=cache_incr, start_pos=0,
            bypass_act=True,
        )
        # Decode token at position 4
        _ = model(
            input_ids[:, 4:5], n_loops=2, kv_cache=cache_incr, start_pos=4,
            bypass_act=True,
        )
        # Decode token at position 5
        out_incr_last = model(
            input_ids[:, 5:6], n_loops=2, kv_cache=cache_incr, start_pos=5,
            bypass_act=True,
        )

    # Compare the logits of the last decode step to the one-shot logits at that pos
    assert torch.allclose(out_incr_last[:, 0], out_oneshot[:, 5], atol=1e-4)
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_nope.py -v -k "kv_cache_prefill_decode"`
Expected: PASS — correct-by-construction since the MLA path only skips `apply_rope` for Q and K when pe_mode="nope", and the cached K is then already the unrotated value. If it FAILS, investigate whether the recurrent loop's per-iter `cache_key` is being mixed across prefill and decode (it should not be — `f"recurrent_loop_{t}"` is shared but prefill writes while decode reads the same key, which is the intended behavior).

- [ ] **Step 3: Commit**

```bash
git add tests/test_nope.py
git commit -m "test(nope): KV-cache correctness under pe_mode=nope"
```

---

## Task 10: Variant config helpers

**Files:**
- Modify: `open_mythos/variants.py` (append two new functions after `mythos_1b()`, around line 33)
- Test: `tests/test_nope.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_nope.py`:

```python
from open_mythos.variants import (
    mythos_1b,
    mythos_1b_scoped_nope,
    mythos_1b_partial_nope,
)


def test_mythos_1b_scoped_nope_sets_only_recurrent_to_nope():
    cfg = mythos_1b_scoped_nope()
    base = mythos_1b()
    assert cfg.pe_mode_prelude == "rope"
    assert cfg.pe_mode_coda == "rope"
    assert cfg.pe_mode_recurrent == "nope"
    # All other fields match the baseline
    assert cfg.dim == base.dim
    assert cfg.n_experts == base.n_experts
    assert cfg.qk_rope_head_dim == base.qk_rope_head_dim
    assert cfg.qk_nope_head_dim == base.qk_nope_head_dim


def test_mythos_1b_partial_nope_collapses_mla_rope_dim():
    cfg = mythos_1b_partial_nope()
    base = mythos_1b()
    assert cfg.pe_mode_prelude == "rope"
    assert cfg.pe_mode_coda == "rope"
    assert cfg.pe_mode_recurrent == "rope"
    assert cfg.qk_rope_head_dim == 0
    # Budget is reassigned: total head dim preserved
    assert cfg.qk_nope_head_dim == base.qk_nope_head_dim + base.qk_rope_head_dim
    assert cfg.dim == base.dim
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_nope.py -v -k "mythos_1b_scoped or mythos_1b_partial"`
Expected: FAIL with `ImportError: cannot import name 'mythos_1b_scoped_nope' from 'open_mythos.variants'`.

- [ ] **Step 3: Add the helpers to `open_mythos/variants.py`**

Append to `open_mythos/variants.py` (after `mythos_1b()`, before `mythos_3b()`):

```python
def mythos_1b_scoped_nope() -> MythosConfig:
    """Scoped NoPE variant of mythos_1b(): RoPE in prelude+coda, NoPE inside
    the recurrent block. See docs/superpowers/specs/2026-04-29-nope-for-recurrent-depth-design.md."""
    cfg = mythos_1b()
    cfg.pe_mode_recurrent = "nope"
    return cfg


def mythos_1b_partial_nope() -> MythosConfig:
    """Partial NoPE variant via MLA: qk_rope_head_dim=0, with the budget
    reassigned to qk_nope_head_dim so total per-head dim is preserved.
    See docs/superpowers/specs/2026-04-29-nope-for-recurrent-depth-design.md."""
    cfg = mythos_1b()
    # Reassign rope-head budget into nope-head (total head dim unchanged)
    cfg.qk_nope_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
    cfg.qk_rope_head_dim = 0
    return cfg
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_nope.py -v -k "mythos_1b_scoped or mythos_1b_partial"`
Expected: both tests PASS.

- [ ] **Step 5: Full suite regression check**

Run: `pytest tests/ -v`
Expected: all pre-existing tests PASS plus the new NoPE tests. If any pre-existing test fails, that's a regression — investigate and fix before proceeding.

- [ ] **Step 6: Commit**

```bash
git add open_mythos/variants.py tests/test_nope.py
git commit -m "feat(variants): add mythos_1b_scoped_nope and mythos_1b_partial_nope"
```

---

## Task 11: `NOPE_VARIANT` env knob in training script

**Files:**
- Modify: `training/1b_poc_fineweb.py` (around line 480 where `cfg = mythos_1b()` is; and the ClearML init section; and the checkpoint dir line)

- [ ] **Step 1: Find the exact current lines**

Run: `grep -n "mythos_1b\|EXPERIMENT_NAME\|CLEARML_PROJECT\|OUTPUT_DIR.*checkpoints" training/1b_poc_fineweb.py`

Record the line numbers for: the `cfg = mythos_1b()` call, the ClearML task name variable, and the checkpoint dir construction.

- [ ] **Step 2: Replace the config loader with a variant-aware selector**

In `training/1b_poc_fineweb.py`, find the line `cfg = mythos_1b()` (around line 480) and replace it with:

```python
    NOPE_VARIANT = os.environ.get("NOPE_VARIANT", "baseline")
    if NOPE_VARIANT == "baseline":
        cfg = mythos_1b()
    elif NOPE_VARIANT == "scoped":
        cfg = mythos_1b_scoped_nope()
    elif NOPE_VARIANT == "partial":
        cfg = mythos_1b_partial_nope()
    else:
        raise ValueError(
            f"NOPE_VARIANT must be 'baseline', 'scoped', or 'partial'; got {NOPE_VARIANT!r}"
        )
```

- [ ] **Step 3: Update the imports at the top of the file**

Find the line `from open_mythos.variants import mythos_1b` and change it to:

```python
from open_mythos.variants import (
    mythos_1b,
    mythos_1b_partial_nope,
    mythos_1b_scoped_nope,
)
```

- [ ] **Step 4: Override ClearML task name and checkpoint subdir based on variant**

Find where `EXPERIMENT_NAME` is read (from env, defaulting to `"1b-10b-tokens"`). Below that line, add:

```python
    # When running a NoPE ablation variant, override the ClearML task name and
    # checkpoint subdir so the three runs don't overwrite each other.
    if NOPE_VARIANT != "baseline" or os.environ.get("NOPE_VARIANT_TAG_BASELINE"):
        EXPERIMENT_NAME = f"nope-ablation/{NOPE_VARIANT}"
```

Find the `ckpt_dir` or equivalent variable that constructs the checkpoint directory. Add a variant-specific suffix:

```python
    if NOPE_VARIANT != "baseline" or os.environ.get("NOPE_VARIANT_TAG_BASELINE"):
        ckpt_dir = os.path.join(
            os.environ.get("OUTPUT_DIR", "./output/experiments"),
            "checkpoints",
            "nope-ablation",
            NOPE_VARIANT,
        )
```

(If the existing code puts checkpoints under a different path, mirror that structure with a `nope-ablation/{variant}` subdir.)

- [ ] **Step 5: Log the variant at training start**

Find the first `logger.info(...)` call inside `main()`, and add above it:

```python
    logger.info(f"NOPE_VARIANT = {NOPE_VARIANT}")
    logger.info(
        f"pe_mode_prelude={cfg.pe_mode_prelude} "
        f"pe_mode_coda={cfg.pe_mode_coda} "
        f"pe_mode_recurrent={cfg.pe_mode_recurrent} "
        f"qk_rope_head_dim={cfg.qk_rope_head_dim} "
        f"qk_nope_head_dim={cfg.qk_nope_head_dim}"
    )
```

- [ ] **Step 6: Smoke test — syntax check and import check**

Run: `source .venv/bin/activate && python -m py_compile training/1b_poc_fineweb.py`
Expected: no errors.

Run: `python -c "import ast; ast.parse(open('training/1b_poc_fineweb.py').read()); print('parse OK')"`
Expected: `parse OK`.

- [ ] **Step 7: Commit**

```bash
git add training/1b_poc_fineweb.py
git commit -m "feat(training): NOPE_VARIANT env knob selects the ablation config"
```

---

## Task 12: bsub submission wrapper for 3 variants

**Files:**
- Create: `deploy/bluevela/run_nope_ablation.sh`
- Create: `deploy/bluevela/bsub_nope_ablation.sh`

- [ ] **Step 1: Write the inner runner script**

Create `deploy/bluevela/run_nope_ablation.sh` with exactly this content:

```bash
#!/usr/bin/env bash
# Inner runner for a single NoPE ablation variant. Invoked by bsub_nope_ablation.sh.
# Expects conda env openmythos activated and CWD at the repo root.
# Required env: NOPE_VARIANT (baseline|scoped|partial), GPUS_PER_NODE

set -euo pipefail

: "${NOPE_VARIANT:?NOPE_VARIANT must be set by the bsub wrapper}"
: "${GPUS_PER_NODE:?GPUS_PER_NODE must be set by the bsub wrapper}"

echo "==============================================="
echo "NoPE ablation: NOPE_VARIANT=${NOPE_VARIANT}"
echo "GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "==============================================="

torchrun --nproc_per_node="${GPUS_PER_NODE}" training/1b_poc_fineweb.py
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x deploy/bluevela/run_nope_ablation.sh
```

- [ ] **Step 3: Write the bsub wrapper**

Create `deploy/bluevela/bsub_nope_ablation.sh` with exactly this content:

```bash
#!/usr/bin/env bash
# Submits the 3-variant NoPE ablation to BlueVela LSF.
# Each variant runs as an independent 4-GPU single-node job on the preemptable queue.
#
# Usage:
#   bash deploy/bluevela/bsub_nope_ablation.sh [baseline|scoped|partial|all]
# Default: all three variants submitted.
#
# Required env: CLEARML_API_HOST, CLEARML_API_ACCESS_KEY, CLEARML_API_SECRET_KEY, HF_TOKEN

set -euo pipefail

REQUIRED_VARS=(CLEARML_API_HOST CLEARML_API_ACCESS_KEY CLEARML_API_SECRET_KEY HF_TOKEN)
MISSING=()
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var:-}" ]; then MISSING+=("$var"); fi
done
if [ ${#MISSING[@]} -gt 0 ]; then
    echo "ERROR: Missing required environment variables:"
    for var in "${MISSING[@]}"; do echo "  - $var"; done
    exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-/u/pzerfos/data/granite-mythos/output/experiments}"
TARGET_TOKENS="${TARGET_TOKENS:-1}"   # 1 B per variant per the spec
CLEARML_PROJECT="${CLEARML_PROJECT:-granite-mythos}"
DATASET_PATH="${DATASET_PATH:-/proj/datasets/pzerfos/fineweb-edu-100B/sample/100BT}"

NUM_NODES=1
GPUS_PER_NODE=4
QUEUE=preemptable
BSUB_GROUP=grp_preemptable

SELECT="${1:-all}"
case "$SELECT" in
    baseline|scoped|partial) VARIANTS=("$SELECT") ;;
    all) VARIANTS=(baseline scoped partial) ;;
    *) echo "Usage: $0 [baseline|scoped|partial|all]"; exit 1 ;;
esac

umask 0002
DATE=$(date "+%Y-%m-%d-%H-%M")
LOG_DIR="${OUTPUT_DIR}/errs_and_logs"
mkdir -p "$LOG_DIR"

for variant in "${VARIANTS[@]}"; do
    JOB_NAME="pz-mythos-nope-${variant}"
    LOG_FILE="${LOG_DIR}/nope-${variant}-${DATE}.log"
    ERR_FILE="${LOG_DIR}/nope-${variant}-${DATE}.err"

    echo "========================================="
    echo "  Submitting NoPE ablation: ${variant}"
    echo "========================================="
    echo "  GPUs:         ${GPUS_PER_NODE} (single node)"
    echo "  Variant:      ${variant}"
    echo "  Tokens/var:   ${TARGET_TOKENS} B"
    echo "  Log:          ${LOG_FILE}"
    echo "========================================="

    bsub \
        -J "${JOB_NAME}" \
        -q "${QUEUE}" \
        -o "${LOG_FILE}" \
        -e "${ERR_FILE}" \
        -n "${NUM_NODES}" \
        -gpu "num=${GPUS_PER_NODE}/task:mode=exclusive_process" \
        -G "${BSUB_GROUP}" \
        blaunch \
        PYTHONUNBUFFERED=1 \
        NOPE_VARIANT="${variant}" \
        GPUS_PER_NODE="${GPUS_PER_NODE}" \
        CLEARML_API_HOST="${CLEARML_API_HOST}" \
        CLEARML_API_ACCESS_KEY="${CLEARML_API_ACCESS_KEY}" \
        CLEARML_API_SECRET_KEY="${CLEARML_API_SECRET_KEY}" \
        HF_TOKEN="${HF_TOKEN}" \
        CLEARML_PROJECT="${CLEARML_PROJECT}" \
        OUTPUT_DIR="${OUTPUT_DIR}" \
        TARGET_TOKENS="${TARGET_TOKENS}" \
        DATASET_PATH="${DATASET_PATH}" \
        bash -c "
            source \$(conda info --base)/etc/profile.d/conda.sh &&
            conda activate openmythos &&
            cd ${REPO_DIR} &&
            bash deploy/bluevela/run_nope_ablation.sh
        "
done
```

- [ ] **Step 4: Make the wrapper executable and syntax-check both**

```bash
chmod +x deploy/bluevela/bsub_nope_ablation.sh
bash -n deploy/bluevela/bsub_nope_ablation.sh
bash -n deploy/bluevela/run_nope_ablation.sh
```

Expected: no output (both scripts syntactically valid).

- [ ] **Step 5: Commit**

```bash
git add deploy/bluevela/bsub_nope_ablation.sh deploy/bluevela/run_nope_ablation.sh
git commit -m "feat(deploy): bsub wrapper for 3-variant NoPE ablation"
```

---

## Task 13: Length-generalization eval script

**Files:**
- Create: `evaluations/eval_length_gen.py`

- [ ] **Step 1: Read the existing eval_checkpoint.py to understand the scaffolding pattern**

Run: `head -60 evaluations/eval_checkpoint.py`

Note the pattern: how it loads the config, instantiates the model, loads state_dict, runs the forward pass. Reuse that pattern.

- [ ] **Step 2: Write the length-gen eval script**

Create `evaluations/eval_length_gen.py`:

```python
"""Length generalization sweep for the NoPE ablation.

Loads a trained checkpoint (baseline / scoped / partial) and evaluates
perplexity on a held-out FineWeb-Edu shard at seq_len in {2048, 4096, 8192, 16384}.
For lengths > the training max_seq_len, regenerates freqs_cis on-the-fly.

See docs/superpowers/specs/2026-04-29-nope-for-recurrent-depth-design.md §6.2.
"""

import argparse
import math
import os

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
```

- [ ] **Step 3: Syntax and import check**

```bash
python -m py_compile evaluations/eval_length_gen.py
```

Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add evaluations/eval_length_gen.py
git commit -m "feat(eval): length generalization sweep for NoPE ablation"
```

---

## Task 14: Depth-generalization eval script

**Files:**
- Create: `evaluations/eval_depth_gen.py`

- [ ] **Step 1: Write the depth-gen eval script**

Create `evaluations/eval_depth_gen.py`:

```python
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
    p.add_argument("--variant", required=True, choices=["baseline", "scoped", "partial"])
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
```

- [ ] **Step 2: Syntax check**

```bash
python -m py_compile evaluations/eval_depth_gen.py
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add evaluations/eval_depth_gen.py
git commit -m "feat(eval): depth generalization sweep for NoPE ablation"
```

---

## Task 15: Logbook entries

**Files:**
- Create: `docs/logbook/2026-04-29-nope-ablation-queued.md`
- Modify: `docs/logbook/2026-04-28-option-b-and-upstream-pr.md` (add NoPE ablation as an open item; soften FSDP2 item language per project feedback — the migration is not "queued"/"decided", only that the feasibility evaluation is complete with results leaning toward FSDP2)

- [ ] **Step 1: Create the queued-ablation logbook**

Create `docs/logbook/2026-04-29-nope-ablation-queued.md`:

```markdown
# NoPE Ablation — Queued for Launch on or after 2026-05-01

**Date:** 2026-04-29
**Status:** Code prepared on branch `feat/nope-ablation`. Launch held pending BlueVela load reduction.

---

## Summary

Ablation study evaluating whether NoPE (no positional embedding) is feasible
and beneficial for OpenMythos's recurrent-depth architecture. Three variants:

- **Baseline** (full RoPE) — `mythos_1b()`
- **Scoped NoPE** — RoPE in prelude/coda, NoPE inside the recurrent block; `mythos_1b_scoped_nope()`
- **Partial NoPE (MLA)** — `qk_rope_head_dim=0`, budget reassigned to `qk_nope_head_dim`; `mythos_1b_partial_nope()`

Each variant: 1 B tokens, 4 × H100 single node, preemptable queue, ~20 h wall
clock. Three jobs in parallel on separate nodes → ~20 h total, ~240 GPU-hours
compute.

Design rationale and pre-registered decision thresholds:
`docs/superpowers/specs/2026-04-29-nope-for-recurrent-depth-design.md`.
Implementation plan:
`docs/superpowers/plans/2026-04-29-nope-ablation.md`.

## Why this is queued, not running

BlueVela is currently overloaded: job 71939 (the ongoing 10 B run) occupies
one preemptable slot, and the general preemptable queue depth is high. Do
not submit until:

1. `bjobs -u pzerfos` shows no conflicting pzerfos jobs pending from earlier
   decisions, AND
2. A spot check of overall queue utilization on BlueVela indicates submitting
   three 4-GPU jobs won't preempt other users' critical work.

Earliest reasonable launch: **on or after 2026-05-01**.

## Launch procedure (when load permits)

```bash
# On BlueVela, from a fresh /u/pzerfos/OpenMythos-nope worktree:
ssh pzerfos@login4.bluevela.rmf.ibm.com
cd /u/pzerfos/OpenMythos
git fetch origin feat/nope-ablation
git worktree add /u/pzerfos/OpenMythos-nope feat/nope-ablation
cd /u/pzerfos/OpenMythos-nope

# Submit all three variants as independent 4-GPU jobs:
bash deploy/bluevela/bsub_nope_ablation.sh all
```

Expected: three PEND jobs named `pz-mythos-nope-{baseline,scoped,partial}`.

## Post-training eval

```bash
# Run per-variant on the final checkpoint:
python evaluations/eval_length_gen.py \
    --checkpoint /proj/checkpoints/pzerfos/openmythos/nope-ablation/scoped/step_NNNNN.pt \
    --variant scoped
python evaluations/eval_depth_gen.py \
    --checkpoint /proj/checkpoints/pzerfos/openmythos/nope-ablation/scoped/step_NNNNN.pt \
    --variant scoped
```

Grep `RESULT` in the log output for the tabulated NLL / PPL at each (seq_len, n_loops) point.

## Decision matrix reference

See spec §7. Summary:

- **Scoped wins** on all three axes → migrate production to Scoped NoPE.
- **MLA partial wins** on parity + comparable extrapolation → adopt the one-line `qk_rope_head_dim=0` config flip.
- **Mixed or regression** → keep full RoPE as default; publish as ablation-negative.
```

- [ ] **Step 2: Update the 2026-04-28 roadmap**

Open `docs/logbook/2026-04-28-option-b-and-upstream-pr.md` and find the "Remaining Open Items" section. Two edits:

**(a) Soften the FSDP2 language** — find the entry for item #4 (starts with "~~**FSDP1 → FSDP2 migration**~~"). Replace it with:

```markdown
4. **FSDP1 → FSDP2 migration** (pzerfos/OpenMythos#6) — **feasibility evaluation completed 2026-04-29**: bench results in `docs/logbook/2026-04-29-fsdp2-feasibility-results.md` show fsdp2 (no compile) delivers ~15% speed-up and ~6% peak-memory reduction on the production (stochastic_depth) recipe vs fsdp1; compile path is unreliable due to RoPE complex-op graph breaks. **Migration itself is not yet decided** — the numbers suggest it's worth doing, but scheduling and sequencing relative to other work is open. Bench code on `bench/fsdp2-feasibility` remains available for re-runs.
```

**(b) Add the NoPE ablation as a new open item** — append to the "Remaining Open Items" list:

```markdown
7. **NoPE ablation study** — 3-variant comparison (baseline RoPE / Scoped NoPE / Partial NoPE via MLA) at 1B scale, 1B tokens each, 4 GPUs per variant. Spec: `docs/superpowers/specs/2026-04-29-nope-for-recurrent-depth-design.md`. Plan: `docs/superpowers/plans/2026-04-29-nope-ablation.md`. Code on `feat/nope-ablation`. **Launch held** until BlueVela load drops; earliest 2026-05-01. Tracks a literature gap (recurrent-depth × NoPE is unstudied).
```

- [ ] **Step 3: Commit both changes**

```bash
git add docs/logbook/2026-04-29-nope-ablation-queued.md \
        docs/logbook/2026-04-28-option-b-and-upstream-pr.md
git commit -m "docs(logbook): queue NoPE ablation; soften FSDP2 status language"
```

---

## Task 16: Full test suite + push branch

**Files:**
- No new file changes. Regression check and push.

- [ ] **Step 1: Run the full test suite**

```bash
source .venv/bin/activate
pytest tests/ -v
```

Expected: all tests PASS. The baseline count from `2026-04-29` was 334 passing; with the new NoPE tests we should have ~344+. If any PRE-EXISTING test fails, stop and investigate — the NoPE plumbing should be backward-compatible because all `pe_mode` parameters default to `"rope"`.

- [ ] **Step 2: Run linters**

```bash
black --check .
ruff check .
```

If either reports issues on the files this plan touched (`open_mythos/main.py`, `open_mythos/variants.py`, `training/1b_poc_fineweb.py`, `tests/test_nope.py`, the two eval scripts), fix with `black .` and `ruff check --fix .` and commit as `style: apply black + ruff`.

- [ ] **Step 3: Verify commit history is clean**

```bash
git log --oneline main..HEAD
```

Expected: 12–15 commits, one per task, descriptive messages, no merge commits.

- [ ] **Step 4: Push the branch**

```bash
git push -u origin feat/nope-ablation
```

Expected: `Branch 'feat/nope-ablation' set up to track 'origin/feat/nope-ablation'.`

- [ ] **Step 5: Optional — open a draft PR for visibility**

```bash
gh pr create --draft --title "feat: NoPE ablation (3 variants, queued for 2026-05-01 launch)" \
    --body "Implements the NoPE ablation described in docs/superpowers/specs/2026-04-29-nope-for-recurrent-depth-design.md. Code complete, tests passing; training jobs NOT YET SUBMITTED — held pending BlueVela load. See docs/logbook/2026-04-29-nope-ablation-queued.md for the launch procedure."
```

This is optional. Skip if the project prefers to land PRs only at merge time.

---

## Remaining work (not in this plan)

- **Training launches** — user-triggered once BlueVela is free (earliest 2026-05-01). See `docs/logbook/2026-04-29-nope-ablation-queued.md` for the procedure.
- **Results writeup** — a follow-up logbook entry with the three eval tables, the pre-registered decision matrix, and the migrate-or-not recommendation. Mirrors `docs/logbook/2026-04-29-fsdp2-feasibility-results.md`'s format.
- **Follow-up: RoPE real-valued cos/sin refactor** — only if NoPE fails to beat RoPE AND we want to revisit `torch.compile` independently.
