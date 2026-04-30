# A Transferable Playbook for AI-Assisted AI-Model R&D on Remote Clusters

**Date:** 2026-04-30
**Status:** Initial draft — expected to iterate with feedback.
**Source material:** The OpenMythos repo end-to-end: `CLAUDE.md`, `docs/logbook/`,
`docs/arch-analysis/`, `docs/superpowers/{specs,plans}/`, `deploy/bluevela/`,
and the two earlier reports in this directory
(`2026-04-30-sandbox-ssh-savings-and-safety.md`).

> If we were to replicate the OpenMythos approach — a single researcher plus
> a coding agent bringing up a novel architecture on a remote GPU cluster —
> for a new model on a different cluster, what would we hand the user on
> day one? This report abstracts the methodology, scaffolding, agentic
> capabilities, and cluster-side services that made OpenMythos work, and
> identifies which of those are reusable vs. project-specific.

> **Attribution.** Additions marked **`[aif]`** in this document originate
> from the sibling `aif-experiment-toolkit` repo
> (`../aif-experiment-toolkit/`), which pursued a parallel effort aimed at
> similar goals. Where aif has a working artifact that maps onto a gap in
> this playbook, we adopt the pattern and credit it inline. The broader
> architecture (MCP-mediated safety model, spec → plan → logbook
> discipline, `superpowers:*` skill pack) remains the OpenMythos
> contribution; aif's contributions are largely in the operational
> discipline around experiments — telemetry, checklists, per-experiment
> directory structure, monitoring modes.
>
> **Pending revision.** The aif-derived content is an in-flight integration.
> Further context on the per-experiment directory layout (numbered
> `experiments/NNN-<slug>/` + append-only `notebook.md` + living `tldr.md`)
> is expected shortly and has been intentionally deferred from this pass.
> Other `[aif]` sections (Launch-readiness checklist, Experiment-monitoring
> modes, shared-dataset registry, telemetry format) are included but may
> be refined as the integration matures.

---

## 1. What actually worked on OpenMythos

Five disciplines stacked to make the bringup possible, and each is
independently transferable:

1. **Two-environment separation.** Local dev (`.venv`, CPU pytest) vs. remote
   cluster (conda env, 8-GPU LSF jobs) with the boundary made explicit in
   `CLAUDE.md`. The agent never confused "local" and "remote" work.
2. **Dated logbook as canonical state-of-world.** One entry per significant
   event (merge, launch, failure, decision), with the latest entry serving
   as the single source of truth. `CLAUDE.md` tells the agent to read it
   first on every session.
3. **Spec → Plan → Execute → Logbook** for non-trivial changes. Spec captures
   motivation + scope + invariants + success criteria; plan captures
   task-by-task execution (with checkbox syntax); subagent-driven
   implementation carries plans into code with per-task reviewers; logbook
   closes the loop.
4. **Pinned invariants in code.** Comments like
   `# INVARIANT: this all_reduce must run on every rank in the same order
   (see docs/logbook/2026-04-23-act-fsdp-deadlock.md)` outlive the people
   who discovered them. Future readers (human or agent) don't "optimize
   them away".
5. **CPU-testable unit suite.** A small-config `pytest` suite runs in <2 min
   locally and catches ~80% of non-distributed regressions before a single
   `bsub`. This is the layer that makes iteration cheap.

Everything in the rest of this report is in service of making those five
disciplines easy to inherit in a new project.

---

## 2. The new project's repo scaffold

What the repo should look like on day one, before any model code is
written:

```
<pkg>/                         # Model code (project-specific)
    main.py                    # Keep it concentrated; resist sprawl
    variants.py                # Pre-configured scales (1B, 3B, …)
tests/                         # CPU-only pytest suite, small configs
training/                      # Training entry points; per-scale
evaluations/                   # Standalone eval scripts
deploy/<cluster>/              # Cluster-specific scheduler wrappers
    setup_env.sh
    bsub_<experiment>.sh       # Outer submission (thin)
    run_<experiment>.sh        # Inner runner (body of the job)
docs/
    logbook/                   # Dated canonical log (YYYY-MM-DD-slug.md)
    superpowers/
        specs/                 # Design specs, one per non-trivial change
        plans/                 # Task-by-task implementation plans
    arch-analysis/             # Architecture diagram + design points
    reports/                   # Ad-hoc analyses
CLAUDE.md                      # Workflow, env discipline, pinned invariants
README.md                      # Human-facing intro
pyproject.toml                 # Deps + tooling (black, ruff, pytest)
```

### Why the `bsub_X.sh` / `run_X.sh` split

Hard-won on OpenMythos. The outer wrapper is thin (env-var validation, job
parameters, scheduler invocation); the inner runner is the body of the
work. This split exists because LSF's single-quote wrapping and
`blaunch`'s env-prefix failure mode break inline multi-line scripts in
subtle ways (OpenMythos job 56133 died from exactly that). The same
pattern carries over to SLURM's `sbatch`. Ship the pair even in the
template.

### Why `CLAUDE.md` is load-bearing

It's the first thing the agent reads on every new session. Ours is ~100
lines across: project overview, experimentation workflow, environments
(local vs. cluster), testing commands, training knobs, architecture key
classes, key design patterns / invariants. The **structure** carries over
to new projects unchanged — the content is refilled. A `CLAUDE.md` that
tells the agent where the logbook lives, where the spec/plan directories
are, and which invariants matter is the on-ramp that collapses a half-day
of "where is everything" into a single file.

### Spec + Plan templates

The spec format used for `2026-04-29-nope-for-recurrent-depth-design.md`
has a structural skeleton worth preserving verbatim:

- Motivation & literature grounding
- Scope & non-goals
- Invariants preserved
- Architecture changes
- Test plan
- Success criteria (decision matrix: Green / Partial-win / Mixed / Red)

The plan format (e.g. `2026-04-29-nope-ablation.md`) is a checkbox-driven
task list with file-change callouts per task. Both should be templated in
`docs/superpowers/{specs,plans}/_TEMPLATE.md` files in the starter repo.

### Logbook discipline

- One dated entry per significant event.
- Latest entry is the canonical state-of-world; `CLAUDE.md` makes this
  load-bearing.
- Capture **why**, not just what. Future-you and future-agent need
  reasoning, not just the diff.
- Link prior entries so the agent can crawl backward.
- Open / closed items tracked explicitly in the canonical roadmap entry
  (`docs/logbook/2026-04-28-option-b-and-upstream-pr.md` is a strong
  template for that specific subtype).

---

## 3. Agentic capabilities (what sandbox-Claude needs installed)

### Inherit the `superpowers:*` skill pack unchanged

These are cluster-agnostic and model-agnostic, and they carried the
OpenMythos bringup:

| Skill | Role |
|---|---|
| `superpowers:brainstorming` | Triggered before non-trivial features; writes a spec |
| `superpowers:writing-plans` | Spec → task-by-task plan |
| `superpowers:executing-plans` / `subagent-driven-development` | Plan → code, with per-task reviewers |
| `superpowers:systematic-debugging` | Bug-triage discipline |
| `superpowers:verification-before-completion` | No "it works" without running it |
| `superpowers:test-driven-development` | For any new code path |
| `code-review:code-review`, `pr-review-toolkit:*` | Pre-merge checks |

A new project should inherit all of these unchanged. They're installed
once in the sandbox and work the same across projects.

### Add project-specific skills

Things to author per-project:

- **Cluster-interaction skill.** Wraps the project's MCP verbs for "submit a
  job", "check status", "fetch log tail", "cancel job". Replaces ad-hoc
  SSH. Crucial — see §5.
- **Launch-readiness skill.** **`[aif]`** Checklist gate before any `bsub`,
  drawing on `aif-experiment-toolkit/practices/pre_launch_checklist.md`.
  Runs in order, and refuses to emit the `submit_job` MCP call if any
  step fails — failing upstream beats eating a 5-minute bsub plus a
  failed-bsub diagnosis cycle:
  - **Git cleanliness.** Hard-fail on uncommitted changes; the
    corresponding job record captures the HEAD commit SHA for
    reproducibility.
  - **Zombie-job scan.** Surface stale `PEND` / hung `RUN` jobs from
    earlier sessions before submitting new work.
  - **Resource preflight.** Validate GPU count vs. model size, memory
    budget vs. batch / sequence length, wall-time estimate vs. requested
    wall time (sanity-check the ratio).
  - **Path / config validation.** Upload paths exist, checkpoint dir is
    correct, experiment name is set, required env vars present, CLI args
    parse, referenced config exists.
  - **Test + lint gates.** CPU test suite passes; ruff / black clean.
  - **Invariants honored.** `CLAUDE.md`-pinned invariants (FSDP
    collective ordering, buffer dtype rules, etc.) still hold in the
    diff being submitted.
- **Experiment-monitoring skill.** **`[aif]`** Supervising long-running
  jobs has three distinct modes, adopted from
  `aif-experiment-toolkit/practices/hyperparameter_sweep.md` and
  `architecture_comparison.md`. Mode is chosen per experiment at launch
  time, not globally, and is recorded in the experiment's plan:
  - **Run-to-completion.** All jobs finish; no intervention. Appropriate
    for fast sweeps, well-understood recipes, or overnight runs.
  - **Interactive exploration.** Sandbox-Claude checks in on a cadence
    proportional to job duration (every ~5% of wall time is a decent
    default), summarizes telemetry, and asks the user whether to kill,
    continue, or branch. Appropriate for risky runs or novel
    architectures.
  - **Autonomous.** Sandbox-Claude makes kill / replace decisions
    against a pre-registered decision matrix (from the experiment's
    plan). Appropriate only when criteria are unambiguous —
    `loss == NaN → kill`, `gnorm > 10× baseline for > N steps → kill`,
    `wall-time budget exceeded → kill`.
  This skill is orthogonal to the production-monitoring / sandbox-
  mediator distinction in §5 and Obs 6 of the sandbox-safety report.
  Both can apply to the same run: autonomous kill-on-NaN here, canonical
  dashboard observation there.
- **Logbook-entry skill.** Opinionated template that prompts for *why*,
  *relationship to prior entries*, *open-items delta*. Reduces drift.

---

## 4. Cluster-side services (owned by ops, not by any project)

This is the layer that doesn't currently exist anywhere in the OpenMythos
ecosystem, and it's the highest-leverage piece of new infrastructure for
future projects. A **cluster onboarding bundle**, separate from any single
project's repo:

1. **Preconditions document.** One page covering: firewall whitelist (what
   dashboards / artifact stores are reachable), airgap status of compute
   nodes, job-scheduler quirks, port-collision patterns, shared-storage
   mount points, auth model. This is the ~20–25 SSH bucket identified in
   the safety report that should never be rediscovered per-project.

2. **Cluster-config module** shipped as a pip-installable package, so
   projects import the preconditions as **code**:
   ```python
   from bluevela_config import CHECKPOINT_ROOT, DATASETS_ROOT, PREEMPTABLE_QUEUE
   ```
   Moves "BlueVela truths" out of every training script.

3. **Reference scheduler templates** (`bsub_template.sh`, `sbatch_template.sh`)
   that already bake in the known hazards: torchrun port pattern
   (`29500 + LSB_JOBID % 1000`), env-var validation block, conda
   activation in non-interactive shells, no-internet-on-compute
   workarounds, `bsub`/`run` split.

4. **MCP orchestrator service.** The server-side piece of the
   sandbox-Claude safety story (see §5 and the sandbox-safety report for
   the full rationale). One deployment per cluster, not per project.
   Exposes narrow verbs (`submit_job`, `get_job_status`, `fetch_log_tail`,
   `list_checkpoints`, `cancel_job`), logs every call, implements rate
   limits and authz.

5. **Dashboard pointers** — ClearML / W&B / Grafana — canonical
   observability, so sandbox-Claude never has to be the sole monitor.

6. **Shared dataset staging + registry.** **`[aif]`** (registry pattern).
   A well-known filesystem path for large datasets so each project doesn't
   re-download (OpenMythos' FineWeb-Edu is 267 GB — non-trivial). Expose
   the registry as a small Python module (modeled on
   `aif-experiment-toolkit/aif/data.py` and `aif/checkpoints.py`) so
   projects consume datasets and checkpoints as rich objects, not bare
   paths:
   ```python
   from <cluster>_config.data import get_dataset
   ds = get_dataset("fineweb-edu-100B")
   ds.path, ds.tokenizer, ds.num_parts   # metadata attached
   ```
   Same pattern for pre-tokenized checkpoints. Removes a class of "wrong
   path / wrong tokenizer / wrong shard count" bugs that would otherwise
   recur per-project.

7. **Telemetry format.** **`[aif]`** A standard on-cluster streaming
   format, adopted from `aif-experiment-toolkit/aif/telemetry.py`:
   - **Append-only JSONL, one file per metric.** E.g.
     `<experiment>/train_loss.jsonl`, `<experiment>/val_perplexity.jsonl`,
     `<experiment>/grad_norm.jsonl`. Each line:
     `{"step": int, "value": float, "timestamp": "ISO8601"}`.
   - **Metric naming convention** — `{category}/{name}`:
     - `train/*` — training-loop metrics (`train/loss`, `train/gnorm`,
       `train/lr`)
     - `val/*` — validation (`val/perplexity`, `val/token_accuracy`)
     - `grad/*` — gradient statistics (`grad/norm`, `grad/variance`)
     - `perf/*` — performance (`perf/tokens_per_sec`, `perf/gpu_util`)
     - Project-specific families as needed (`router/*`, `lti/*` on
       OpenMythos).
   - **Optional forwarders.** The same logger mirrors to ClearML / W&B /
     Grafana. JSONL remains the source-of-truth — local, grep-able,
     outlives any single dashboard, survives a ClearML outage.
   - **Why this matters for the agent.** Sandbox-Claude reads telemetry
     by having the MCP orchestrator `fetch_log_tail` on the JSONL files,
     not by calling ClearML APIs. One fewer credential for the agent to
     hold; one more layer of containment (§5). ClearML stays available
     for human operators as the canonical dashboard channel.

---

## 5. Safety: sandbox-Claude never holds raw SSH keys

The containment property. Full rationale is in
`docs/reports/2026-04-30-sandbox-ssh-savings-and-safety.md`; in brief:

- Sandbox-Claude runs in a persistent cloud sandbox (0 / 1 / 2 GPUs
  depending on how much of the distributed-systems debugging should move
  off the cluster — 2-GPU is the right target for FSDP/NCCL work).
- **All** cluster interactions route through the MCP orchestrator.
  Sandbox-Claude holds a token to the MCP server; the MCP server holds
  the SSH credentials. Verbs are narrow, observable, auditable,
  revocable.
- Canonical production monitoring stays on human/ops-side channels
  (ClearML dashboards, ops tooling). Sandbox-Claude can mediate on demand
  but is never the sole authority.

This separation is what makes the sandbox worth having. Without it, a
sandbox is just "a cloud machine with SSH keys", which is strictly more
dangerous than a laptop because it's persistent and not tied to a
physically-present human.

---

## 6. The day-one handoff kit

Concretely, what gets given to the user when a new project starts:

1. **Template repo.** Git template or cookiecutter, pre-populated with the
   scaffold in §2. Fill-in `CLAUDE.md` with project-specific blanks
   clearly marked. Example spec, example plan, example logbook entry
   included as references (the OpenMythos NoPE ablation makes a good
   exemplar — spec, plan, launched-logbook, and post-merge fix all on
   record).
2. **Cluster preconditions document** (§4).
3. **MCP orchestrator URL + auth token** for their sandbox.
4. **Pointer to the skill pack** (§3) and instructions to install.
5. **A "first-PR" guided walkthrough.** Small feature, end-to-end
   spec → plan → subagent execution → logbook entry, so the user sees the
   cycle before tackling the real model.
6. **The sandbox-safety report + one-page TL;DR**, so the user understands
   *why* the MCP separation exists and doesn't try to "just SSH in for a
   sec".

---

## 7. Deliberately not in the handoff

- **The model itself.** Every AI-model R&D project has a unique
  architecture; templating past `tests/`-pattern and `variants.py`-pattern
  is counterproductive.
- **Training hyperparameters.** Project-specific.
- **Dataset handling.** Varies too much; a short `docs/datasets.md` note is
  enough.
- **Cluster-native bugs** (LSF quoting, `blaunch`, scheduler corner cases).
  These live in the cluster preconditions document, not in any project's
  `CLAUDE.md`.

---

## 8. Three-layer summary

The transferable content splits cleanly:

| Layer | Owner | Artifact |
|---|---|---|
| **Project repo** | Researcher | Template + filled-in `CLAUDE.md` + scaffold (§2, §3-project-specific) |
| **Cluster infra** | Cluster ops | Preconditions doc, config module, scheduler templates, MCP orchestrator, shared storage, dashboards (§4) |
| **Agent capabilities** | Sandbox platform | `superpowers:*` skill pack, MCP auth token, sandbox runtime (§3-inherited, §5) |

If a new project starts with all three layers in place, the Apr 22–24
OpenMythos bringup — the 113-SSH saga that took a week — compresses to
"clone the template, fill in the model, push, watch ClearML". The
sandbox-safety report already quantified the raw SSH savings (~72–78% on
a 2-GPU sandbox); the repo / skills / cluster split above is how those
savings are **realized** in practice. The SSH number is the scoreboard;
the three-layer split is the playbook.

---

## 9. What to prototype first

Given where the OpenMythos ecosystem currently stands, the highest-leverage
next step is building the **cluster-side services (§4)** — specifically:

- **MCP orchestrator + auth model.** This is the keystone of the safety
  story and doesn't exist yet.
- **BlueVela preconditions document + `bluevela_config` pip package.**
  Codifies the ~20–25 SSH bucket of cluster truths so the next project
  doesn't relearn them.

The **project-repo template (§2)** is largely a straight extraction from
the current OpenMythos repo — mostly deletion of project-specific content
plus a `_TEMPLATE.md` pair under `docs/superpowers/`. Lower priority
because the marginal value of templating is small until a second project
actually exists.

The **skill pack (§3)** is already mature — the `superpowers:*` skills
used throughout OpenMythos apply unchanged. The only per-project additions
(cluster-interaction, launch-readiness, logbook-entry) are short and
cheap to author once the MCP orchestrator exists.

---

## 10. Open questions for follow-up

- **What's the auth model for the MCP orchestrator?** Per-user tokens?
  Per-sandbox tokens with TTL? Delegated auth from an identity provider?
  This is the security-design bottleneck.
- **Should the template repo carry a minimal "hello-world" model** (e.g.,
  a tiny 10M-param transformer) so the first-PR walkthrough has something
  to land against? Or is that contrary to §7?
- **How opinionated should the template be about FSDP vs FSDP2?** The
  FSDP2 feasibility work suggests migration is worth it; baking it in
  saves the next project from the FSDP1 dtype-fix saga entirely.
- **Is the canonical logbook format itself a generalizable primitive** —
  i.e., worth lifting into a project-independent `logbook` tool that
  enforces structure? Or does handrolled Markdown remain the right
  trade-off?
