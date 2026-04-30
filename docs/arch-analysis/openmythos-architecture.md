# OpenMythos Architecture

## ★ Updates Since Original Doc

The bullets below summarize changes merged on `main` since this diagram was
first drafted. Each ★ item also recurs in **Key Design Points** at the end of
this file, so the "what changed" list and the "what's true now" description
stay in sync. The ASCII diagram itself is unchanged — the bullets reference
diagram steps by number where relevant.

- **★ 2026-04-28 — Stochastic-depth training (PR #7).**
  `recurrent_mode ∈ {"act", "stochastic_depth"}`; **default is now
  `stochastic_depth`**. When enabled, `n_loops` is sampled uniformly from
  `[stochastic_depth_min, stochastic_depth_max]` (defaults `1..32`) on rank 0
  and broadcast, **ACT halting (step 6 in the diagram) is bypassed**, and
  `h_out = h_{n_loops−1}` replaces the ACT-weighted sum at the loop exit.
  Checkpoints are cross-mode compatible — `ACTHalting` weights stay in the
  `state_dict` and simply receive no gradient under `bypass_act=True`. Spec:
  `docs/superpowers/specs/2026-04-27-stochastic-depth-training-design.md`.
- **★ 2026-04-29 — DeepSeek-V3 aux-loss-free load balancing (PRs #8/#9).**
  Every MoE FFN (step 3 in the diagram, inside the recurrent block) adds a
  per-expert `router_bias` term to its gate scores and updates it each
  optimizer step by `−rate · sign(count − target)` with an FSDP-safe
  `all_reduce`. Default `router_bias_update_rate = 1e-3` — live on the 10B run
  since step 156,000 on 2026-04-29. Emits
  `router_imbalance_{max_over_mean,stddev_over_mean,ratio}` ClearML metrics
  when active; set to `0.0` to disable. See
  `docs/logbook/2026-04-29-router-bias-load-balancing.md`.
- **★ 2026-04-29 — NoPE ablation variants (PR #10).** Positional encoding is
  now selectable **per region** via three flags
  `pe_mode_prelude / pe_mode_recurrent / pe_mode_coda ∈ {"rope", "nope"}`,
  plus MLA's `qk_rope_head_dim=0` for the "partial" case. Three preset
  variants shipped: `baseline` (full RoPE everywhere — unchanged default),
  `scoped` (NoPE inside the recurrent block only), and `partial` (MLA without
  the RoPE sub-head). Selected at submission time via
  `python training/1b_poc_fineweb.py --variant {baseline,scoped,partial}`.
  Spec: `docs/superpowers/specs/2026-04-29-nope-for-recurrent-depth-design.md`.
- **★ 2026-04-29 — LTI stability clamp tightened** (step 5 in the diagram).
  `log_dt + log_A` clamped to `[-13, 20]` keeps `ρ(A) < 1` **strictly** under
  fp32 (the prior boundary could round to exactly 1.0 and break the stability
  guarantee).
- **Not pictured — perf & FSDP-safety (2026-04-23 → 29).** MoE dispatch is now
  a grouped sort+batch (~6.7× over the naive nested loop, no behavioral
  change). `n_loops` sampling and every MoE `all_reduce` must execute on every
  rank in the same order — rank-local early-exits cause NCCL-watchdog
  deadlock. Pinned `INVARIANT:` comments in `MoEFFN.update_router_bias` and
  `RecurrentBlock.forward` guard against regressions; canonical case analysis
  in `docs/logbook/2026-04-23-act-fsdp-deadlock.md`.

---

```
                        ┌─────────────────────┐
                        │     Input Tokens     │
                        │      (B, T)          │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │   Token Embedding    │
                        │   nn.Embedding       │
                        │   (B, T) → (B,T,D)  │
                        └──────────┬──────────┘
                                   │
         ══════════════════════════╪══════════════════════════
                      P R E L U D E  (run once)
         ══════════════════════════╪══════════════════════════
                                   │
                        ┌──────────▼──────────┐
                        │  TransformerBlock 0  │  ─┐
                        │  ┌───────────────┐   │   │
                        │  │ RMSNorm→Attn  │   │   │  prelude_layers
                        │  │ (GQA or MLA)  │   │   │  (default: 2)
                        │  │ + residual    │   │   │
                        │  ├───────────────┤   │   │  Dense SwiGLU FFN
                        │  │ RMSNorm→FFN   │   │   │  (no MoE)
                        │  │ (Expert)      │   │   │
                        │  │ + residual    │   │   │
                        │  └───────────────┘   │   │
                        ├──────────────────────┤   │
                        │  TransformerBlock 1  │  ─┘
                        └──────────┬──────────┘
                                   │
                              e = x (frozen copy for injection)
                                   │
         ══════════════════════════╪══════════════════════════
                  R E C U R R E N T   B L O C K
                  (single block, looped T times)
         ══════════════════════════╪══════════════════════════
                                   │
              ┌────────────────────▼────────────────────┐
              │  for t in range(n_loops):               │
              │                                         │
              │   ┌───────────────────────────────┐     │
              │   │ 1. Loop-Index Embedding       │     │
              │   │    sinusoidal signal → h       │     │
              │   └───────────────┬───────────────┘     │
              │                   ▼                     │
              │   ┌───────────────────────────────┐     │
              │   │ 2. RMSNorm(h + e)             │     │
              │   │    ↳ re-inject original input  │     │
              │   └───────────────┬───────────────┘     │
              │                   ▼                     │
              │   ┌───────────────────────────────┐     │
              │   │ 3. TransformerBlock (MoE)     │     │
              │   │    ┌──────────────────────┐   │     │
              │   │    │ Attn (GQA or MLA)    │   │     │
              │   │    ├──────────────────────┤   │     │
              │   │    │ MoE FFN              │   │     │
              │   │    │ shared + routed      │   │     │
              │   │    │ experts              │   │     │
              │   │    └──────────────────────┘   │     │
              │   └───────────────┬───────────────┘     │
              │                   ▼                     │
              │   ┌───────────────────────────────┐     │
              │   │ 4. LoRA Adapter (depth-wise)  │     │
              │   │    delta = (down(x)*scale[t])@B│     │
              │   │    per-loop scale, shared A/B  │     │
              │   └───────────────┬───────────────┘     │
              │                   ▼                     │
              │   ┌───────────────────────────────┐     │
              │   │ 5. LTI Injection              │     │
              │   │    h = A·h + B·e + trans_out  │◄──── e
              │   │    ρ(A) < 1 guaranteed (ZOH)  │     │
              │   └───────────────┬───────────────┘     │
              │                   ▼                     │
              │   ┌───────────────────────────────┐     │
              │   │ 6. ACT Halting                │     │
              │   │    p = σ(Linear(h))           │     │
              │   │    weight·h → h_out           │     │
              │   │    if Σp ≥ threshold: halt    │     │
              │   └───────────────┬───────────────┘     │
              │                   │                     │
              │              ─────┘  (loop back         │
              │                       or exit)          │
              └────────────────────┬───────────────────┘
                                   │
                                   │  h_out = Σ(weight_t · h_t)
                                   │  (ACT-weighted sum)
                                   │
         ══════════════════════════╪══════════════════════════
                         C O D A  (run once)
         ══════════════════════════╪══════════════════════════
                                   │
                        ┌──────────▼──────────┐
                        │  TransformerBlock 0  │  ─┐
                        │  (Attn + Dense FFN)  │   │  coda_layers
                        ├──────────────────────┤   │  (default: 2)
                        │  TransformerBlock 1  │   │  Dense SwiGLU FFN
                        │  (Attn + Dense FFN)  │  ─┘  (no MoE)
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │      RMSNorm        │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │   LM Head (Linear)  │
                        │   weight-tied with   │
                        │   Token Embedding    │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │   Output Logits     │
                        │   (B, T, vocab_size)│
                        └─────────────────────┘
```

## Key Design Points

- **Prelude & Coda** use dense SwiGLU FFN (`Expert`); only the **Recurrent Block** uses MoE FFN
- **Attention** is runtime-switchable between GQA and MLA via `cfg.attn_type`
- **★ Positional encoding is per-region selectable** via
  `pe_mode_prelude / pe_mode_recurrent / pe_mode_coda ∈ {"rope", "nope"}`;
  for MLA, `qk_rope_head_dim=0` produces the "partial NoPE" configuration.
  `variant="baseline"` = full RoPE everywhere (the unchanged default).
- The **same weights** are reused every loop iteration -- differentiated only by the sinusoidal loop-index embedding and per-loop LoRA scale vectors
- **LTI injection** re-injects the frozen prelude output `e` at every loop
  step, preventing hidden-state drift, with guaranteed stability (ρ(A) < 1).
  **★** `log_dt + log_A` is clamped to `[-13, 20]` so ρ(A) stays strictly
  below 1 in fp32.
- **ACT halting** allows early exit per-position: easy tokens stop early,
  hard tokens keep looping -- the final output is a probability-weighted sum
  across all iterations. **★ Bypassed by default now** — under
  `recurrent_mode="stochastic_depth"` (the current default) the halting step
  is skipped entirely and the final iteration's hidden state is returned
  directly. ACT remains available as an opt-in via `recurrent_mode="act"`;
  checkpoints are cross-mode compatible.
- **★ MoE load balancing is aux-loss-free** — a per-expert `router_bias`
  term is added to gate scores and shifted each optimizer step by
  `−rate · sign(count − target)` (DeepSeek-V3 Algorithm 1), with an
  FSDP-safe `all_reduce`. Default `router_bias_update_rate = 1e-3`; set to
  `0.0` to disable.
- More loops at inference = deeper reasoning, with no extra parameters
  (depth extrapolation). **★** Stochastic-depth training explicitly exercises
  the full `[1..32]` loop range during training, which is what produces the
  monotonic PPL-vs-depth curve that the ACT-only recipe did not.
