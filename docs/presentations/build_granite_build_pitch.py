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


def add_container_box(slide, left, top, width, height, header, items,
                      header_color=NAVY, body_color=LIGHT):
    """Labeled container with a colored header strip and stacked inner rows."""
    outer = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    outer.fill.solid()
    outer.fill.fore_color.rgb = RGBColor(0xFA, 0xFA, 0xFA)
    outer.line.color.rgb = header_color
    outer.line.width = Pt(1.5)
    outer.text_frame.text = ""

    header_h = Inches(0.4)
    hdr = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, header_h)
    hdr.fill.solid()
    hdr.fill.fore_color.rgb = header_color
    hdr.line.fill.background()
    tf = hdr.text_frame
    tf.margin_top = Inches(0.03)
    tf.margin_bottom = Inches(0.03)
    p = tf.paragraphs[0]
    p.alignment = 1
    r = p.add_run()
    r.text = header
    r.font.size = Pt(13)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    inner_top = top + header_h + Inches(0.08)
    inner_left = left + Inches(0.12)
    inner_w = width - Inches(0.24)
    avail_h = (top + height) - inner_top - Inches(0.08)
    gap = Inches(0.06)
    item_h = (avail_h - gap * (len(items) - 1)) / len(items)
    for item in items:
        ib = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, inner_left, inner_top, inner_w, item_h
        )
        ib.fill.solid()
        ib.fill.fore_color.rgb = body_color
        ib.line.color.rgb = header_color
        ib.line.width = Pt(0.5)
        tfi = ib.text_frame
        tfi.word_wrap = True
        tfi.margin_left = Inches(0.08)
        tfi.margin_right = Inches(0.08)
        pi = tfi.paragraphs[0]
        pi.alignment = 1
        ri = pi.add_run()
        ri.text = item
        ri.font.size = Pt(11)
        ri.font.bold = True
        ri.font.color.rgb = BLACK
        inner_top = inner_top + item_h + gap


def add_labeled_arrow(slide, left, top, width, height, label, color=ACCENT):
    arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, left, top, width, height)
    arrow.fill.solid()
    arrow.fill.fore_color.rgb = color
    arrow.line.fill.background()
    lbl = slide.shapes.add_textbox(
        left - Inches(0.1), top - Inches(0.32), width + Inches(0.2), Inches(0.28)
    )
    p = lbl.text_frame.paragraphs[0]
    p.alignment = 1
    r = p.add_run()
    r.text = label
    r.font.size = Pt(10)
    r.font.bold = True
    r.font.italic = True
    r.font.color.rgb = color


def add_banner(slide, left, top, width, height, text,
               color=NAVY, text_color=None, font_size=13, italic=False):
    if text_color is None:
        text_color = RGBColor(0xFF, 0xFF, 0xFF)
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    box.fill.solid()
    box.fill.fore_color.rgb = color
    box.line.fill.background()
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = 1
    r = p.add_run()
    r.text = text
    r.font.size = Pt(font_size)
    r.font.bold = True
    r.font.italic = italic
    r.font.color.rgb = text_color


def add_down_arrow(slide, left, top, width, height, color=ACCENT):
    arrow = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, left, top, width, height)
    arrow.fill.solid()
    arrow.fill.fore_color.rgb = color
    arrow.line.fill.background()


def add_schematic_caption(slide, left, top, width, text):
    box = slide.shapes.add_textbox(left, top, width, Inches(0.3))
    p = box.text_frame.paragraphs[0]
    p.alignment = 1
    r = p.add_run()
    r.text = text
    r.font.size = Pt(11)
    r.font.italic = True
    r.font.color.rgb = GREY


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

    # --- Schematic: Sandbox → gbserver → Backends (design doc §2) ---
    sch_top = Inches(1.45)
    sch_h = Inches(2.55)

    add_container_box(
        slide, Inches(0.4), sch_top, Inches(4.2), sch_h,
        "Sandbox (OpenShell · k8s-sig)",
        [
            "Claude Code + MCP client",
            "Git · Docker/Podman · Python",
            "Local GPU (optional, for dev)",
        ],
    )
    add_labeled_arrow(
        slide, Inches(4.7), sch_top + Inches(1.05), Inches(0.55), Inches(0.4), "MCP"
    )
    add_container_box(
        slide, Inches(5.35), sch_top, Inches(5.3), sch_h,
        "Granite.build (gbserver)",
        [
            "MCP Server — 10 tools (submit, status, discover, plans)",
            "BuildWatch · BuildRunner · preflight validation",
            "Storage · Artifacts · Lineage · Provenance",
        ],
    )
    add_labeled_arrow(
        slide,
        Inches(10.75), sch_top + Inches(1.05), Inches(0.55), Inches(0.4),
        "dispatch",
    )
    add_container_box(
        slide, Inches(11.4), sch_top, Inches(1.55), sch_h,
        "Backends",
        [
            "Docker / Podman",
            "RunPod (cloud)",
            "SkyPilot · k8s · Slurm",
        ],
    )

    add_schematic_caption(
        slide,
        Inches(0.4), Inches(4.05), Inches(12.6),
        "Backends fetch code (git), images (registry), and artifacts "
        "(S3 · HF · COS) directly — never through MCP or the agent.",
    )

    # --- Condensed bullets below schematic ---
    add_bullets(
        slide,
        Inches(0.5), Inches(4.45), Inches(6.2), Inches(2.35),
        [
            "Persistent container, decoupled from laptop — survives across sessions",
            "Local GPU for prototyping before scaling out",
            "Safe execution: preflight checks, space-scoped credentials + artifacts",
        ],
        title="a) Safe, persistent agent sandbox",
        body_size=12,
    )
    add_bullets(
        slide,
        Inches(6.95), Inches(4.45), Inches(6.0), Inches(2.35),
        [
            "Control plane only — specs pin commit SHAs + image URIs",
            "10 MCP tools: discover steps/envs/quotas, submit, poll, retrieve",
            "Deterministic runs; artifact lineage and provenance by construction",
        ],
        title="b) Granite.build as scale mediation",
        body_size=12,
    )

    add_footer_band(
        slide,
        "Replace hundreds of SSH hops with a typed control plane — the agent submits, Granite.build runs it.",
    )


def slide3(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(
        slide,
        "From sandbox to at-scale — packaging + submission",
        "Claude Code helps with development, packaging, AND the build.yaml submission",
    )

    # --- Shared starting banner ---
    add_banner(
        slide,
        Inches(3.17), Inches(1.45), Inches(7.0), Inches(0.55),
        "Local dev iterations complete in the sandbox — code runs end-to-end",
    )

    # Fork arrows into the two path columns
    add_down_arrow(slide, Inches(3.3), Inches(2.1), Inches(0.35), Inches(0.3))
    add_down_arrow(slide, Inches(9.7), Inches(2.1), Inches(0.35), Inches(0.3))

    # --- Path A: Git push → Bring-Your-Own-Step ---
    add_container_box(
        slide, Inches(0.4), Inches(2.5), Inches(6.2), Inches(2.7),
        "Path A — Git push → Bring-Your-Own-Step (BYOS)",
        [
            "Commit & push to GitHub (pin commit SHA, not a branch ref)",
            "Claude Code authors a custom_code step (BYOS) referencing github_url",
            "Backend clones repo, runs setup_command then start_command",
        ],
    )

    # --- Path B: Docker image → Bring-Your-Own-Image ---
    add_container_box(
        slide, Inches(6.73), Inches(2.5), Inches(6.2), Inches(2.7),
        "Path B — Docker image → Bring-Your-Own-Image (BYOI)",
        [
            "Claude Code writes Dockerfile — faithful conda-env reproduction",
            "Build + push to registry with content-addressed tag (not :latest)",
            "Claude Code authors a custom_code step (BYOI) — backend pulls the image",
        ],
    )

    # Merge arrows from path boxes into the submit banner
    add_down_arrow(slide, Inches(3.3), Inches(5.25), Inches(0.35), Inches(0.3))
    add_down_arrow(slide, Inches(9.7), Inches(5.25), Inches(0.35), Inches(0.3))

    # --- Converging banner: Claude Code writes build.yaml, submits via gbserver ---
    add_container_box(
        slide, Inches(1.4), Inches(5.65), Inches(10.53), Inches(1.15),
        "Claude Code constructs the build.yaml → submit_job (Granite.build)",
        [
            "At-scale execution on backends (Docker · RunPod · SkyPilot · k8s · Slurm)",
            "Artifacts registered automatically with lineage + provenance",
        ],
    )

    add_footer_band(
        slide,
        "Two paths, one control plane — Claude Code assists from the first commit to the last artifact.",
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
