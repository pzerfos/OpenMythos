# BlueVela SSH Connection Audit

**Date:** 2026-04-24
**Purpose:** Quantify the SSH connections Claude Code made to BlueVela over the OpenMythos training bringup campaign (Apr 22-24, 2026).

---

## Background

The OpenMythos 1B PoC training required iterative debugging on BlueVela — a
remote GPU cluster with no internet on compute nodes, IBM LSF job scheduling,
and conda-based environments. Every code change followed the cycle:

    edit locally -> commit -> push -> SSH git pull -> SSH bsub -> SSH bpeek
    -> SSH read error log -> diagnose -> repeat

Claude Code session transcripts (`.jsonl` files) record every tool invocation,
allowing an exact count of SSH connections.

---

## Per-Session Breakdown

Data sourced from Claude Code session transcripts in
`.claude/projects/.../`.

| Session ID | Date(s) | Purpose | Direct SSH | Monitor loops | Est. loop iterations |
|------------|---------|---------|--------:|--------:|---------:|
| `de7fc69e` | Apr 22-23 | Training bringup: env setup, dataset download, FSDP dtype fixes, 15+ job submissions | 113 | 0 | 0 |
| `0217b802` | Apr 23 | MoE dispatch optimization, ACT deadlock fix, checkpoint relocation, job monitoring | 48 | 13 | ~200 |
| `1d3f525a` | Apr 22 | Dataset pre-download (FineWeb-Edu, OpenHermes) | 4 | 0 | 0 |
| `435fc847` | Apr 24 | Job status checks, kill training run, eval script, depth sweep | 33 | 4 | ~60 |
| Subagents | Various | Spawned by above sessions | ~16 | 0 | 0 |

## Totals

| Category | Count |
|----------|------:|
| Direct SSH commands (`ssh ... "command"`) | ~214 |
| Monitor/until loops (poll every 10-15s) | 17 loops |
| Estimated SSH connections from loops (~15 iterations avg) | ~260 |
| **Grand total SSH connections** | **~475** |

## What Those ~475 Connections Were Doing

| Category | Est. count | Examples |
|----------|--------:|---------|
| Job monitoring (`bjobs`, `bpeek`) | ~120 | Status checks across ~20 distinct LSF jobs |
| Log/error inspection (`tail`, `cat`) | ~80 | Reading stdout/stderr for crash diagnosis |
| Code deploy + job submission (`git pull`, `bsub`) | ~60 | Push-pull-submit cycle per fix |
| Filesystem operations (`ls`, `find`, `du`) | ~50 | Checkpoint management, path discovery |
| Environment setup (`conda`, `pip`, verification) | ~40 | One-time setup + debugging |
| Dataset operations (`hf download`, `ls`) | ~30 | Pre-downloading FineWeb-Edu (267GB) |
| Monitor loop polling (`bjobs` status checks) | ~260 | Waiting for jobs to complete/fail |

## Jobs Submitted

20 distinct LSF jobs were submitted across all sessions:

| Job ID | Date | Outcome | Purpose |
|--------|------|---------|---------|
| 21640 | Apr 22 | EXIT | Early bringup attempt |
| 21642 | Apr 22 | EXIT | Early bringup attempt |
| 21700 | Apr 22 | EXIT | Early bringup attempt |
| 25650 | Apr 23 | EXIT | Dataset/env debugging |
| 26165 | Apr 23 | EXIT | FSDP dtype fix iteration |
| 26274 | Apr 23 | EXIT | FSDP dtype fix iteration |
| 26324 | Apr 23 | EXIT | FSDP dtype fix iteration |
| 26340 | Apr 23 | EXIT | FSDP dtype fix iteration |
| 26360 | Apr 23 | EXIT | FSDP dtype fix iteration |
| 26369 | Apr 23 | EXIT | OOM debugging |
| 26373 | Apr 23 | EXIT | OOM debugging |
| 26374 | Apr 23 | EXIT | Micro-batch tuning |
| 26420 | Apr 23 | EXIT | Convergence test |
| 26518 | Apr 23 | EXIT | **First convergence confirmed** (preempted at step 31) |
| 29260 | Apr 23 | EXIT | Flash attention test |
| 31808 | Apr 23 | EXIT | Pre-MoE-optimization (deadlocked at step 32) |
| 33050 | Apr 23 | EXIT | Post-MoE-optimization (deadlocked at step 33) |
| 33841 | Apr 23 | EXIT | **ACT fix validated** (ran to step 1,004, killed manually) |
| 34019 | Apr 23 | EXIT | **1B token run** (ran to step 31,225, killed after target met) |
| 41981 | Apr 24 | EXIT | Eval run (crashed on temperature=0.0 greedy decode) |
| 41996 | Apr 24 | DONE | **Eval run** (generation samples + depth sweep completed) |

## Heaviest Session: `de7fc69e` (Apr 22-23)

113 direct SSH commands across ~12 hours. This session handled the full bringup:

1. Conda environment setup (system Python too old for project)
2. `conda activate` failure in non-interactive LSF shells
3. FineWeb-Edu streaming hang (no internet on compute nodes)
4. HuggingFace `datasets` 17,000x slower than pyarrow for local parquet
5. ClearML `Task.init()` 10-minute hang (added 30s SIGALRM timeout)
6. 5 rounds of FSDP mixed-precision dtype fixes (RMSNorm, softmax, nn.Linear, LM head)
7. OOM at micro_batch=4 and micro_batch=2 (settled on micro_batch=1)
8. NCCL timeout on 2-GPU (resolved by dtype fixes, scaled to 4-GPU)

Each bug required 5-10 SSH round-trips to diagnose and verify the fix.

## Observations

- The SSH-heavy pattern is inherent to remote cluster debugging with no
  interactive access to compute nodes. Every hypothesis requires a full
  submit-wait-read cycle.
- Monitor loops account for ~55% of total connections but are low-cost
  (small `bjobs` status checks every 10-15s).
- The bringup session (`de7fc69e`) accounts for ~50% of all direct SSH
  commands. Once training was stable, subsequent sessions needed far fewer
  connections.
- Total wall-clock time across all sessions: ~36 hours. Average: ~13 SSH
  connections per hour, or ~1 every 4.5 minutes.
