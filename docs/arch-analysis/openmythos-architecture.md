# OpenMythos Architecture

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
- The **same weights** are reused every loop iteration -- differentiated only by the sinusoidal loop-index embedding and per-loop LoRA scale vectors
- **LTI injection** re-injects the frozen prelude output `e` at every loop step, preventing hidden-state drift, with guaranteed stability (ρ(A) < 1)
- **ACT halting** allows early exit per-position: easy tokens stop early, hard tokens keep looping -- the final output is a probability-weighted sum across all iterations
- More loops at inference = deeper reasoning, with no extra parameters (depth extrapolation)
