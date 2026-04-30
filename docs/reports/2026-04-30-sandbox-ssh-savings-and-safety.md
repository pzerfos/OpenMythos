# Sandbox GPU Count, SSH Savings, and Cluster Safety — A Short Analysis

> A coding agent running inside a sandbox can do less damage to a shared
> cluster than one that has direct SSH into it. The SSH-connection savings
> quantified below are one measurable consequence of that separation; the
> safety benefit — keeping a dev agent from cheaply acting on (and cheaply
> breaking) live production infrastructure — is the other, and arguably the
> more important one.

**Date:** 2026-04-30
**Source audit:** `docs/logbook/2026-04-24-bluevela-ssh-audit.md` (~475 SSH
connections from Claude Code to BlueVela across the Apr 22–24 OpenMythos 1B PoC
training bringup campaign)

---

## Question

If Claude Code lived inside a development sandbox with a GPU attached, how
many of the ~475 BlueVela SSH connections from the Apr 22–24 bringup campaign
would have been avoidable? How does the answer change with 0, 1, or 2 GPUs in
the sandbox?

## Scope

- **Prep / dev / debug belongs in the sandbox** — code, environment readiness,
  model debugging, dataset prep, single-job smoke tests, eval-script
  iteration.
- **Scaled experiments (>2 GPU) stay on BlueVela** — the 1B convergence run,
  the 10B production run, any multi-node work, any eval on
  production-sized checkpoints.
- Claude Code runs inside the sandbox, replacing today's laptop `.venv`.
  "Local" = sandbox.
- Savings are measured as "SSH connections avoided" against the audit's
  ~475 total; bringup-heavy, not steady-state.

## Terminology note

Throughout this report, "SSH" is used as shorthand for "remote operation
against the cluster". In the target architecture described in Obs 6 and the
Recommendation, sandbox-Claude performs these operations via **narrow,
MCP-exposed, orchestrator-backed tooling** — not direct SSH keys to the
cluster. Handing raw SSH credentials to a dev agent would defeat the
containment property the sandbox is meant to provide. The audit's raw SSH
counts remain the right unit for quantifying *how much remote cluster
interaction happens*; the architectural question of *which mechanism
performs it* (direct SSH, MCP-over-orchestrator, user-driven) is separate
and sits on top.

## Methodology

Walk each of the 7 categories and each of the 20 bsubs in the audit and
classify:

1. What kind of bug it was (env / dataset / pure-Python / single-GPU kernel /
   FSDP-sharding / NCCL collective-ordering / OOM / production-scale).
2. What minimum hardware reproduces that bug class.
3. Whether a pre-submission smoke test in the sandbox would have caught it,
   thereby eliminating the failed-bsub diagnosis cycle (~10–20 SSHs per
   cycle) and its attached poll loop (~10–20 poll iterations per cycle).

## Per-sandbox analysis

### 0 GPU — Linux + internet + disk, CPU only

Beyond today's laptop `.venv`, a 0-GPU sandbox adds:

- A Linux environment matching BlueVela's, catching env-install issues the
  darwin laptop misses.
- Enough disk for the 267 GB FineWeb-Edu dataset, so dataset-format debugging
  happens on the real data.
- Containerized isolation + continuous availability.

It catches: env iteration, dataset-format bugs, pure-Python bugs (the
`temperature=0.0` greedy-decode crash from job 41981), algorithmic unit tests,
bsub-script syntax pre-flight.

It does not catch: anything that requires a CUDA kernel path — Flash-Attn,
LoRA/LTI on CUDA, FSDP, NCCL, OOM, MoE performance, collective ordering.

Failed bsubs pre-screened: ~4–5 of 20 (21640, 21642, 21700, 25650, 41981).

The ~15–20% SSH savings are real and worth counting, but they are one
yardstick among several — not the primary one for the 0-GPU tier.
Persistence and resumability are the larger drivers here. See Obs 3.

### 1 GPU — adds single-GPU CUDA paths

Adds on top of 0-GPU: single-GPU training smoke (real-weight forward/backward),
Flash-Attn single-GPU path, LoRA/LTI on CUDA, greedy-decode on the real model,
model-fit OOM checks.

Still does not catch: FSDP sharding, NCCL collectives, ACT/MoE collective-
ordering deadlocks, rank-0 broadcast semantics, multi-GPU OOM. FSDP requires
≥2 ranks to do anything meaningful; NCCL collective-ordering bugs manifest
only when there are actual collectives.

Failed bsubs pre-screened: ~6–7 of 20 (above + 29260 Flash-Attn test, partial
26369/26373 OOM-at-load).

### 2 GPU — crosses the distributed threshold

2 is the minimum GPU count at which FSDP and NCCL do anything real. This is
a phase change, not a linear improvement. Adds on top of 1-GPU: the entire
**FSDP dtype-fix saga** (5 jobs, ~50 SSHs of diagnosis), the **MoE all-reduce
collective-ordering bugs** (2 jobs), the **ACT collective-ordering deadlock**
(1 job plus its lineage), 2-GPU NCCL timeouts, sharded OOM (qualitatively).

Still does not catch: true-scale production (≥4 GPU; ops-side concern, see
Obs 6), BlueVela-native topology bugs (LSF non-interactive shell, `blaunch`,
ClearML firewall, compute-node-no-internet, torchrun port 29500 TIME_WAIT —
most of which belong in onboarding preconditions rather than sandbox-Claude's
budget; see Obs 4), and scaling quirks that only surface at 8 GPU (HSDP
sharding dimensions, per-rank memory at real batch, gradient-accumulation
timing).

Failed bsubs pre-screened: ~13–15 of 20 (above + 26165/274/324/340/360 FSDP
dtype, 31808/33050 MoE deadlock, 33841 ACT-lineage).

## Summary table

| Sandbox | Bug classes caught pre-submission | Bug classes still require SSH | Failed bsubs pre-screened (of 20) | Est. SSH saved | % of ~475 |
|---|---|---|:---:|:---:|:---:|
| **0 GPU** | Env iteration, dataset-format, pure-Python bugs, algorithmic unit tests, bsub syntax | Any GPU kernel path, Flash-Attn, FSDP, NCCL, OOM, collective ordering, MoE perf | ~4–5 | ~60–100 | ~15–20% |
| **1 GPU** | 0-GPU set **+ single-GPU training smoke**, Flash-Attn, LoRA/LTI on CUDA, greedy-decode on real model, model-fit OOM | FSDP sharding, NCCL collectives, ACT/MoE deadlocks, rank-0 broadcast semantics, multi-GPU OOM | ~6–7 | ~180–230 | ~38–48% |
| **2 GPU** | 1-GPU set **+ FSDP dtype fixes, NCCL collective ordering, ACT collective-ordering deadlock, MoE all-reduce ordering**, 2-GPU NCCL timeouts, sharded OOM | True-scale production (ops-side, Obs 6); LSF shell / queue / `blaunch` genuine residue (~10–15 SSHs, Obs 4; bulk of the old "BlueVela-native" list belongs in onboarding preconditions); 2→8 GPU scaling quirks | ~13–15 | ~340–370 | ~72–78% |

## Key observations

1. **1 → 2 GPU is the phase change, not 0 → 1.** Going from 0 to 1 GPU roughly
   doubles savings by adding CUDA-path bugs. Going from 1 to 2 GPU doubles
   again, and every SSH saved at that step is attributable to one bug family:
   **distributed-systems bugs that only manifest under real collectives**. Any
   sandbox below 2 GPU leaves the single largest bug family in the audit
   (FSDP / NCCL / deadlock) out of scope and therefore still BlueVela-bound.

2. **Diminishing returns after 2 GPU.** 4 or 8 GPUs would catch an additional
   tail of scaling-specific bugs (HSDP sharding dimensions, gradient-
   accumulation timing at real batch, LSF multi-node), but the marginal
   gain is ~5–10 SSHs per bringup. The 2-GPU step directly saves ~72–78%
   of the audit; the remaining ~22–28% is not a "better sandbox" problem
   and splits into three different buckets (detailed in Obs 4, 6, and the
   Recommendation): cluster preconditions that belong in onboarding docs
   (~20–25 SSHs), ops-side monitoring that belongs to a different actor
   entirely (~125 SSHs), and a small genuine first-encounter residue on
   the order of ~10–15 SSHs that falls on sandbox-Claude.

3. **0 GPU: SSH savings are one yardstick; persistence is the bigger one.**
   Against the SSH-count metric, a 0-GPU sandbox saves ~15–20% — real and
   worth counting, but modestly above today's laptop `.venv`. What the
   SSH-count metric understates is that the point of putting Claude in a
   0-GPU cloud sandbox isn't "be better than the laptop on GPU work" (it
   has no GPU to be better with); it's a broader set of properties that
   the SSH lens doesn't measure directly:
   - **Persistence.** The agent keeps running across laptop sleep, battery
     death, wifi loss, commute, office-to-home transitions. Long-running
     tasks — dataset downloads, training-log tailing, multi-hour ClearML
     polling, overnight batch jobs — don't die with the developer's
     physical environment.
   - **Resumability.** Reconnect hours or days later from any machine and
     pick up exactly where the agent left off. Workspace, running
     processes, scheduled follow-ups, and open investigations all survive.
   - **Continuous availability.** Notifications, `/loop`-style periodic
     checks (PR monitoring, `bjobs` polling, deploy watches) fire when
     they're supposed to fire, instead of only when the laptop is open.
   - **Consistent Linux env + dataset-scale disk.** Env-install issues the
     darwin laptop hides surface here; and there's room for the full 267 GB
     FineWeb-Edu dataset locally.

   For OpenMythos-class workflows where the dev agent often needs to watch
   a remote BlueVela job for hours or days, the persistence/resumability
   axis sits alongside — and often outweighs — the SSH-count axis. Both
   matter. But a 0-GPU cloud sandbox is worth doing on persistence grounds
   alone, even before the ~15–20% SSH savings are counted.

4. **Most BlueVela-native "bugs" are really documentation gaps.** The
   ~35-SSH bucket labelled "BlueVela-native topology bugs" needs to be split.
   A large share of it was discovered the expensive way during Apr 22–24,
   but the majority of those items are not fundamental discoveries — they
   are **preconditions of the cluster** that ought to be supplied upfront as
   assumptions or handled by cluster operators, not rediscovered by every
   new dev agent:

   - **ClearML firewall to `clearml-ext.sl.res.ibm.com:8008`** — an ops-side
     action (the firewall was opened by cluster ops on 2026-04-27). A dev
     agent given "ClearML is reachable on port 8008" as a precondition
     spends 0 SSHs on this.
   - **Compute-node-no-internet** — a standard property of many HPC
     clusters. A 2-line onboarding note ("compute nodes are airgapped;
     stage datasets and artifacts to shared FS before submission") turns
     this from a several-SSH discovery into a design constraint the agent
     respects from commit one.
   - **Torchrun port 29500 TIME_WAIT collisions** — once encountered, the
     fix is a standard pattern (`--master_port = 29500 + (LSB_JOBID % 1000)`,
     landed in `897c9e5`). Baking this into the project's bsub template
     eliminates the class entirely for future projects.

   After moving these items into preconditions, the **genuinely irreducible**
   first-encounter residue — non-interactive `conda activate` under LSF
   shells, LSF queue-specific behavior, `blaunch` quirks if multi-node is
   ever re-enabled — is closer to **~10–15 SSHs**, not ~35. And even that
   shrinks with each project that inherits the accumulated preconditions
   list. The lesson: the cost of "a sandbox hides cluster-topology bugs"
   is **proportional to how poorly the cluster is documented for incoming
   dev agents**, not to anything intrinsic about the sandbox.

5. **Bringup vs steady-state.** The Apr 22–24 window was a one-time bringup
   saturated with distributed-systems debugging, so a 2-GPU sandbox hits
   disproportionately hard there (~72–78% savings). For steady-state
   production monitoring — the current 10B run at step 218k/305k is a good
   example — the sandbox provides little extra lift because the bulk of SSHs
   are `bjobs` polling on a healthy real-scale job, which stays on BlueVela by
   scope. Call it ~25–35% steady-state savings vs ~72–78% bringup savings.
   The bringup-heavy case is the right one to optimize against, because
   bringup is where the pain lives.

6. **Sandbox-Claude can mediate production monitoring, but need not own
   it.** The ~125-SSH production-monitoring bucket could be framed as
   "irreducible because scaled runs live on BlueVela", but that framing
   quietly assumes sandbox-Claude is the one doing all the monitoring.
   The sandbox's role is narrower: **bridge development-to-operations
   quickly and safely**, i.e., get a well-tested job from "idea" to "bsub
   submitted on BlueVela" without risking the cluster. Once the job is
   running at scale, monitoring can and often should be shared across
   multiple channels:
   - **Human operators watching ClearML dashboards** directly — canonical
     source of truth, no SSH, no dev-agent involvement.
   - **Ops-facing tools and dashboards** that ingest BlueVela state
     without round-tripping through a dev agent.
   - **A separate, ops-scoped Claude session** with a different permission
     boundary and a different risk profile, when more agent autonomy is
     needed on the operations side.
   - **Sandbox-Claude as an assistive mediator** — answering user-triggered
     "what's the status of job X?" questions, summarizing log tails,
     drafting a diagnosis for a human to verify, flagging anomalies. In
     this mode sandbox-Claude uses **MCP-exposed tooling backed by a
     job-submission-and-monitoring orchestrator** (narrow verbs: submit,
     query status, fetch log tails), rather than raw SSH keys to the
     cluster. This distinction is the whole point: **an agent holding a
     direct SSH key to the cluster is not a contained agent**. MCP tools
     expose only the verbs we intend to grant, each call is observable and
     auditable, and revocation is a single server-side change. This is a
     UX / convenience role, not an authority role.

   The safety property is the same under all four modes: **sandbox-Claude
   is never the sole authority on production state**. The user retains
   direct visibility via ClearML and can verify anything the agent reports.
   A dev agent that can cheaply act on a live cluster is a liability
   precisely when it's the only thing watching; when it's one channel among
   several, the blast radius of a confused agent is bounded by the human
   who can still see the truth directly. Under the mediator role, the
   ~125-SSH bucket doesn't disappear from sandbox-Claude's accounting —
   it shrinks to whatever on-demand checks the user asks for, and the rest
   moves to the canonical channels.

## Recommendation

The sandbox's role is **development-to-operations handoff**: take a job from
"idea" to "bsub submitted with high confidence it will run cleanly", nothing
more. Three recommendations follow from that:

**1. Pick the GPU tier by goal, not by default.**
- **If the primary goal is persistence and resumability** (continuous
  availability for long-running investigations, `/loop` checks, `bjobs`
  polling that survives laptop sleep) — **a 0-GPU sandbox is already a
  substantial improvement** over a laptop workflow, regardless of its
  modest direct SSH savings. See Obs 3.
- **If the primary goal is minimizing SSH-per-bringup** on distributed-
  systems debugging across future OpenMythos-class projects — **a 2-GPU
  sandbox is the right target**. 1 GPU leaves the dominant bug family
  (FSDP / NCCL / collective-ordering) on BlueVela; 4+ GPUs are expensive
  and only add marginal catch rate. 2 is the threshold at which FSDP and
  NCCL do real work.

**2. Invest in preconditions, not in a larger sandbox.**
The old ~35-SSH "BlueVela-native topology bugs" bucket splits into ~20–25
SSHs of **cluster preconditions** (ClearML firewall is open on port 8008,
compute nodes are airgapped, torchrun port pattern `29500 + LSB_JOBID%1000`,
etc.) and ~10–15 SSHs of genuine first-encounter residue. The preconditions
belong in onboarding docs supplied by cluster operators, not in each dev
agent's debug loop. Well-documented preconditions shrink the residue toward
zero across subsequent projects; see Obs 4.

**3. Share production-scale monitoring across channels — via MCP, not raw
SSH.** The ~125-SSH production-monitoring bucket (`bjobs` polling, `bpeek`,
log tails on running scaled jobs) is not a bucket sandbox-Claude should own
as sole authority. Canonical monitoring belongs on ClearML dashboards and
ops-side tooling; sandbox-Claude can act as an **assistive mediator** —
user-triggered status questions, log-tail summaries, anomaly flags for
human review — but through **narrow, auditable MCP verbs backed by an
orchestrator**, not direct SSH keys. A dev agent holding raw SSH
credentials to the cluster is not a contained agent, and can cheaply act
on (and break) live infrastructure. Routing its cluster interactions
through an MCP layer, and keeping it one channel among several rather than
the sole observer, is what preserves the containment property. See Obs 6.

---

## Final accounting

Putting all the reframes together, the ~475 SSHs in the Apr 22–24 audit
decompose as follows under a 2-GPU sandbox with Claude inside and the
division of labor above:

| Bucket | SSHs | Who handles it |
|---|:---:|---|
| Directly saved by sandbox-Claude (prep, dev, debug of single-GPU and FSDP/NCCL bugs reproducible at 2 ranks) | ~340–370 | sandbox-Claude, locally, 0 SSH |
| Cluster preconditions (ClearML firewall, compute-node-no-internet, torchrun port pattern, …) | ~20–25 | cluster operators / onboarding docs, once per cluster |
| Production-scale monitoring (`bjobs`, `bpeek`, log tails on running scaled jobs) | ~125 | primarily ClearML dashboards, ops-side tooling, humans, or a separately-scoped ops agent; sandbox-Claude only as **assistive mediator via MCP verbs** (user-triggered, never sole authority, no raw SSH keys) |
| Genuine sandbox-Claude first-encounter residue (LSF shell, `blaunch` quirks if multi-node returns, queue specifics) | ~10–15 | sandbox-Claude, shrinks project-to-project as preconditions list grows |
| **Total** | **~475** | |

The headline "~72–78% avoidable" is the directly-saved number alone. If the
architectural separation is taken seriously — sandbox for dev, ops tooling
for ops, onboarding docs for cluster facts, and **MCP-mediated cluster
access rather than raw SSH keys** in sandbox-Claude's hands — sandbox-
Claude's actual residual SSH burden on the next bringup of this kind lands
in the **~10–15 SSH range**, not ~160. That is the real prize, and most of
it is a safety property (an agent with narrower remote reach is an agent
with narrower blast radius) rather than a productivity one.
