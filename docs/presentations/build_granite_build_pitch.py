"""Generate a 3-slide PowerPoint pitching Granite.build Agentic as a mediation
layer for new-model R&D, grounded in the OpenMythos/BlueVela experience.

Usage:
    python docs/presentations/build_granite_build_pitch.py
"""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

OUTPUT = Path(__file__).parent / "2026-04-28-openmythos-on-BlueVela.pptx"

NAVY = RGBColor(0x0B, 0x2A, 0x4A)
ACCENT = RGBColor(0xC9, 0x3C, 0x20)
GREY = RGBColor(0x55, 0x55, 0x55)
LIGHT = RGBColor(0xF2, 0xF2, 0xF2)
BLACK = RGBColor(0x11, 0x11, 0x11)


def add_title(slide, text, subtitle=None):
    box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(1.0))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = text
    r.font.size = Pt(30)
    r.font.bold = True
    r.font.color.rgb = NAVY
    if subtitle:
        p2 = tf.add_paragraph()
        r2 = p2.add_run()
        r2.text = subtitle
        r2.font.size = Pt(14)
        r2.font.color.rgb = GREY
        r2.font.italic = True


def add_bullets(slide, left, top, width, height, items, *, title=None,
                title_color=NAVY, body_size=14, bullet_char="•"):
    """items: list[str] or list[tuple(text, indent_level)]."""
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    first = True
    if title:
        p = tf.paragraphs[0]
        r = p.add_run()
        r.text = title
        r.font.size = Pt(16)
        r.font.bold = True
        r.font.color.rgb = title_color
        first = False
    for item in items:
        if isinstance(item, tuple):
            text, indent = item
        else:
            text, indent = item, 0
        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()
        p.level = indent
        r = p.add_run()
        prefix = "   " * indent + ("◦ " if indent > 0 else f"{bullet_char} ")
        r.text = prefix + text
        r.font.size = Pt(body_size)
        r.font.color.rgb = BLACK


def add_footer_band(slide, text):
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(6.9), Inches(13.33), Inches(0.6)
    )
    bar.line.fill.background()
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    tf = bar.text_frame
    tf.margin_left = Inches(0.3)
    tf.margin_right = Inches(0.3)
    p = tf.paragraphs[0]
    p.alignment = 1  # center
    r = p.add_run()
    r.text = text
    r.font.size = Pt(14)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)


def add_workflow_band(slide, top):
    steps = [
        "edit locally",
        "git commit/push",
        "SSH git pull",
        "SSH bsub",
        "SSH bpeek / tail logs",
        "diagnose",
        "repeat",
    ]
    left = Inches(0.5)
    width = Inches(1.72)
    gap = Inches(0.05)
    for i, s in enumerate(steps):
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, Inches(0.55))
        box.fill.solid()
        box.fill.fore_color.rgb = LIGHT if i % 2 == 0 else NAVY
        box.line.color.rgb = NAVY
        tf = box.text_frame
        tf.margin_left = Inches(0.05)
        tf.margin_right = Inches(0.05)
        p = tf.paragraphs[0]
        p.alignment = 1
        r = p.add_run()
        r.text = s
        r.font.size = Pt(11)
        r.font.bold = True
        r.font.color.rgb = NAVY if i % 2 == 0 else RGBColor(0xFF, 0xFF, 0xFF)
        left = left + width + gap


def slide1(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    add_title(
        slide,
        "OpenMythos on BlueVela: the cost of remote R&D",
        "Recurrent-Depth Transformer bringup — Apr 22–24, 2026",
    )

    add_bullets(
        slide,
        Inches(0.5), Inches(1.35), Inches(12.3), Inches(1.0),
        [
            "OpenMythos: open reconstruction of a Recurrent-Depth Transformer "
            "(Prelude → looped Recurrent block → Coda) with ACT halting, depth-wise "
            "LoRA, fine-grained MoE, switchable GQA/MLA attention.",
            "Current effort: develop the model and bring a 1B-param PoC to reliable "
            "training on BlueVela (remote GPU cluster, LSF scheduler, no internet on "
            "compute nodes).",
        ],
        body_size=13,
    )

    # Workflow band
    wf_label = slide.shapes.add_textbox(Inches(0.5), Inches(2.7), Inches(6), Inches(0.3))
    p = wf_label.text_frame.paragraphs[0]
    r = p.add_run()
    r.text = "Inner loop — every code change:"
    r.font.size = Pt(13)
    r.font.bold = True
    r.font.color.rgb = GREY
    add_workflow_band(slide, Inches(3.05))

    # Pain pane — left
    add_bullets(
        slide,
        Inches(0.5), Inches(3.95), Inches(6.2), Inches(2.8),
        [
            "~475 SSH connections in 3 days across 4 Claude Code sessions",
            "  ~214 direct commands + ~260 monitor-loop polls",
            "20 LSF jobs submitted; most failed iteratively:",
            "  5 rounds of FSDP mixed-precision dtype fixes",
            "  OOM at micro_batch=4, 2 → settled on 1",
            "  NCCL timeout, ACT deadlock, ClearML 10-min hang",
            "Heaviest session: 113 SSH in ~12 hrs — ~1 SSH every 4.5 min",
        ],
        title="The comms overhead",
        title_color=ACCENT,
        body_size=12,
    )

    # Pain pane — right
    add_bullets(
        slide,
        Inches(6.95), Inches(3.95), Inches(6.0), Inches(2.8),
        [
            "Each hypothesis = a full submit → wait → read-log cycle",
            "  5–10 SSH round-trips per bug to diagnose + verify the fix",
            "Remote cluster has no interactive compute-node access",
            "Monitor loops poll bjobs/bpeek every 10–15 s just to detect state change",
            "Engineer time is dominated by plumbing — not modeling",
        ],
        title="Why it hurts",
        title_color=ACCENT,
        body_size=12,
    )

    add_footer_band(slide, "Hundreds of SSH hops to move a single idea from laptop to a running GPU job.")


def slide2(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(
        slide,
        "Granite.build Agentic — safe sandbox + scale mediation",
        "Collapse the submit-SSH-tail loop into submit → poll → inspect",
    )

    # Left column: sandbox
    add_bullets(
        slide,
        Inches(0.5), Inches(1.45), Inches(6.2), Inches(5.2),
        [
            "Persistent container (OpenShell) with Claude Code, Git, Docker/Podman, Python",
            "Optional local GPU for prototyping before scaling out",
            "Persists across sessions — decoupled from the laptop; no laptop ↔ cluster tunnel",
            "Safe execution: preflight validation, space-scoped credentials and artifacts",
            "Agent Guide tells Claude Code which MCP tools exist and when to use each",
        ],
        title="a) Safe, persistent agent sandbox with a GPU",
        body_size=13,
    )

    # Right column: scale mediation
    add_bullets(
        slide,
        Inches(6.95), Inches(1.45), Inches(6.0), Inches(5.2),
        [
            "Control-plane-only: build specs reference commit SHAs + content-addressed images — no data moves through the agent",
            "Deterministic, reproducible runs with artifact lineage and provenance",
            "Compute backends (Bash / Docker / RunPod; later K8s/LSF) fetch code, images, and artifacts themselves",
            "10 MCP tools let the agent discover steps, environments, quotas, and submit jobs directly",
            "Preflight validation blocks bad specs before a GPU is ever allocated",
        ],
        title="b) Granite.build as the scale mediation layer",
        body_size=13,
    )

    add_footer_band(
        slide,
        "Replace hundreds of SSH hops with a typed control plane — the agent submits, Granite.build runs it.",
    )


def slide3(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(
        slide,
        "Flight plans + logbooks — external memory for experiments",
        "The intent, the methodology, and the daily record — all durable and searchable",
    )

    # Left column: flight plans
    add_bullets(
        slide,
        Inches(0.5), Inches(1.45), Inches(6.2), Inches(5.2),
        [
            "Co-authored with the agent in the sandbox; captures the what and the why",
            "Level 1 — Intent (human narrative)",
            "Level 2 — Methodology (structured steps, dependencies)",
            "Level 3 — Execution (build.yaml, derived at submit time)",
            "Append-only revision history; one plan spawns many builds as you iterate",
            "Every job carries plan_id → build → artifact — provenance by construction",
        ],
        title="Flight plans: durable intent",
        body_size=13,
    )

    # Right column: logbooks
    add_bullets(
        slide,
        Inches(6.95), Inches(1.45), Inches(6.0), Inches(5.2),
        [
            "Daily record of what was achieved, decided, or ruled out — not raw transcripts",
            "Searchable, shareable, persistent external record of agent sessions",
            "Future retrieval: answer months later why micro_batch=1 or why MLA over GQA",
            "Example: the 475-SSH audit was reconstructed from session transcripts — logbooks make this first-class",
            "Pairs with flight plans: the plan says what we meant to do, the logbook says what actually happened",
        ],
        title="Logbooks: durable history",
        body_size=13,
    )

    add_footer_band(
        slide,
        "Conversations end. Flight plans and logbooks remain — and can be searched, shared, and resumed.",
    )


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)
    slide1(prs)
    slide2(prs)
    slide3(prs)
    prs.save(OUTPUT)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
