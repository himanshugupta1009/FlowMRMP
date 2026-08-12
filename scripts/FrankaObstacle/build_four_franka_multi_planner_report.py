#!/usr/bin/env python3
"""Build the concise PDF documenting the four-Franka planner integration.

The report is intentionally implementation-focused: exact changes needed to
make Prioritized Planning, CRRT/CRRT-EB, and KCBS work with current Franka
VanillaRRT/FlowEBRRT and the shared 295-sphere cuRobo world model.  It also
includes the fully audited 360-trial outcome table for traceability.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "results" / "franka_obstacle" / "four_franka_multi_planner_20seeds_final_v3_20260812"
PROBLEM_DIR = ROOT / "results" / "franka_obstacle" / "four_franka_multi_planner_problems_final_v3_20260812"
OUTPUT = ROOT / "output" / "pdf" / "four_franka_multi_planner_implementation_report.pdf"

NAVY = colors.HexColor("#17243B")
BLUE = colors.HexColor("#2563A6")
TEAL = colors.HexColor("#087F8C")
PALE = colors.HexColor("#EAF1F8")
INK = colors.HexColor("#1D2733")
MUTED = colors.HexColor("#526171")
GREEN = colors.HexColor("#DFF4E6")


def fmt(value, digits=2):
    if value in (None, ""):
        return "—"
    return f"{float(value):.{digits}f}"


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#C8D4E1"))
    canvas.line(0.55 * inch, 0.40 * inch, 10.45 * inch, 0.40 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.58 * inch, 0.22 * inch, "Four-Franka coordination · exact 295-sphere audit")
    canvas.drawRightString(10.42 * inch, 0.22 * inch, f"{doc.page}")
    canvas.restoreState()


def build():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    aggregate = list(csv.DictReader((RUN_DIR / "aggregate.csv").open(encoding="utf-8")))
    manifest = json.loads((PROBLEM_DIR / "manifest.json").read_text(encoding="utf-8"))
    overall = [row for row in aggregate if row["difficulty"] == "all"]
    tiers = [row for row in aggregate if row["difficulty"] != "all"]

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "Title", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=23, leading=26, textColor=NAVY, alignment=TA_CENTER,
        spaceAfter=10,
    )
    subtitle = ParagraphStyle(
        "Subtitle", parent=styles["Normal"], fontSize=10.5, leading=14,
        textColor=MUTED, alignment=TA_CENTER, spaceAfter=14,
    )
    h1 = ParagraphStyle(
        "H1", parent=styles["Heading1"], fontName="Helvetica-Bold",
        fontSize=15, leading=18, textColor=BLUE, spaceBefore=7, spaceAfter=6,
    )
    body = ParagraphStyle(
        "Body", parent=styles["BodyText"], fontSize=9.2, leading=12.2,
        textColor=INK, spaceAfter=5,
    )
    small = ParagraphStyle(
        "Small", parent=body, fontSize=7.4, leading=9.2, spaceAfter=0,
    )
    callout = ParagraphStyle(
        "Callout", parent=body, backColor=PALE, borderColor=BLUE,
        borderWidth=0.7, borderPadding=8, textColor=NAVY, leading=13,
        spaceBefore=4, spaceAfter=8,
    )

    doc = BaseDocTemplate(
        str(OUTPUT), pagesize=landscape(letter),
        leftMargin=0.55 * inch, rightMargin=0.55 * inch,
        topMargin=0.45 * inch, bottomMargin=0.52 * inch,
        title="Four-Franka Multi-Planner Implementation Report",
        author="FrankaFM benchmark workflow",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="pages", frames=[frame], onPage=footer)])

    story = [
        Paragraph("Four-Franka Multi-Planner Integration", title),
        Paragraph(
            "Prioritized Planning · centralized CRRT/CRRT-EB · KCBS · current VanillaRRT and FlowEBRRT",
            subtitle,
        ),
        Paragraph(
            "Outcome: all six planner/base combinations now share the existing four-arm table world, "
            "the 295-sphere MorphIt/cuRobo robot model, world-from-base transforms, and one exact "
            "collision/audit definition. PyBullet was not used for planning or evaluation.",
            callout,
        ),
        Paragraph("What was changed", h1),
    ]

    changes = [
        ["Area", "Implementation"],
        ["Shared geometry", Paragraph(
            "Reused the four cuRobo checkers, table/pedestal/mount cuboids, fixed world-from-base transforms, "
            "and 295 world-frame spheres per arm. Added flattened joint-path collision and synchronized first-conflict callbacks.", small)],
        ["Prioritized", Paragraph(
            "Reused the existing dynamic-obstacle adapter. Each lower-priority arm checks its cuRobo edge against cached sphere trajectories of higher-priority arms.", small)],
        ["CRRT", Paragraph(
            "Added optional articulated metric and exact joint-path callbacks to the legacy centralized planner. Goal-biased parent/extension selection uses each Franka agent's 7D position policy; ordinary parent selection remains 14D.", small)],
        ["CRRT-EB / Flow", Paragraph(
            "Added a dynamic centralized Flow variant using the current best_inference.pt model. It generates 32 sequence edges per active arm, ranks by normalized 7D position, physically propagates each sequence, repairs colliding joint candidates, then falls back to random controls.", small)],
        ["KCBS", Paragraph(
            "Added an exact conflict-detector callback and exact time-indexed constraint builder. Low-level Vanilla/Flow replanning checks 295-sphere overlaps against the constrained arm state at the matching timestep; legacy point-radius geometry is bypassed.", small)],
        ["Budgeting", Paragraph(
            "Enforced one 30 s total budget per four-arm trial. Added pre-extension deadline checks so a zero-remaining-budget RRT/CRRT call cannot execute one expensive rollout. KCBS passes only the remaining budget to every low-level call.", small)],
        ["Reproducibility", Paragraph(
            "Created a resumable 360-trial runner. Every trial writes one JSON record; audited successes also store four trajectories. Aggregates are regenerated from raw records.", small)],
    ]
    table = Table(changes, colWidths=[1.25 * inch, 8.55 * inch], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8.5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#BCC9D6")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F6F9FC")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.extend([table, Spacer(1, 8), Paragraph("Scenario calibration and protocol", h1)])
    problem_rows = [["Tier", "Direct invalid", "Arm pairs", "Mean normalized q motion", "Purpose"]]
    purpose = {
        "easy": "Short, nontrivial; each arm starts outside radius 0.40",
        "medium": "Longer motion with one exact inter-arm bottleneck",
        "hard": "Large motion with two exact inter-arm bottlenecks",
    }
    for problem in manifest["problems"]:
        problem_rows.append([
            problem["label"].title(),
            f"{problem['invalid_interior_samples']}/39",
            str(problem["colliding_pair_count"]),
            fmt(problem["mean_normalized_q_distance"], 3),
            purpose[problem["label"]],
        ])
    problem_table = Table(problem_rows, colWidths=[0.8*inch, 1.2*inch, 0.8*inch, 1.55*inch, 5.45*inch])
    problem_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), TEAL), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 8),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#BBCBD0")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F2FAFA")]),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 5), ("RIGHTPADDING", (0,0), (-1,-1), 5),
    ]))
    story.extend([
        problem_table,
        Paragraph(
            "Protocol: 20 evaluation seeds per tier and planner/base combination; 3 tiers × 6 combinations = 360 trials. "
            "Success requires planner completion and a post-run audit of joint/velocity limits, cuRobo self/static collision, "
            "and synchronized world-frame inter-arm sphere overlap. Timing statistics below include successful trials only.",
            body,
        ),
        Paragraph("Files and reproducibility", h1),
        Paragraph(
            "Adapters: scripts/FrankaObstacle/four_franka_multi_planner.py and franka_prioritized_adapter.py. "
            "Runner: benchmark_four_franka_multi_planner.py. Scenario generator: create_four_franka_benchmark_problems.py. "
            "Legacy integration points: mrmp_with_kite_extend/src/cRRT.py, kcbs.py, and rrt.py. Raw JSONs, trajectories, "
            "CSVs, summary, and hashes are in results/franka_obstacle/four_franka_multi_planner_20seeds_final_v3_20260812/.",
            small,
        ),
        Paragraph("Overall results (60 trials per row)", h1),
    ])

    overall_rows = [[
        "Coordination", "Base", "Success", "Plan mean / median (s)",
        "Makespan mean / median (s)", "Nodes total", "Waypoints total",
    ]]
    for row in overall:
        overall_rows.append([
            row["coordination"].title(), row["base_planner"].title(),
            f"{row['successes']}/60 ({float(row['success_percent']):.1f}%)",
            f"{fmt(row['successful_planning_mean_seconds'])} / {fmt(row['successful_planning_median_seconds'])}",
            f"{fmt(row['successful_path_makespan_mean_seconds'])} / {fmt(row['successful_path_makespan_median_seconds'])}",
            f"{int(row['all_trial_tree_nodes_total']):,}",
            f"{int(row['all_trial_checked_waypoints_total']):,}",
        ])
    overall_table = Table(overall_rows, colWidths=[1.25*inch, 0.8*inch, 1.35*inch, 1.65*inch, 1.9*inch, 1.2*inch, 1.45*inch], repeatRows=1)
    overall_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), NAVY), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 7.6),
        ("ALIGN", (2,1), (-1,-1), "CENTER"), ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#B9C6D3")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, PALE]),
        ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story.extend([overall_table, Spacer(1, 8), Paragraph("Per-tier success and successful-trial timing", h1)])

    display_name = {"prioritized": "Prioritized", "crrt": "CRRT", "kcbs": "KCBS"}
    tier_rows = [["Coordination + base", "Tier", "Success", "Plan mean / med (s)", "Makespan mean / med (s)"]]
    for row in tiers:
        tier_rows.append([
            f"{display_name[row['coordination']]} + {row['base_planner'].title()}",
            row["difficulty"].title(),
            f"{row['successes']}/20 ({float(row['success_percent']):.0f}%)",
            f"{fmt(row['successful_planning_mean_seconds'])} / {fmt(row['successful_planning_median_seconds'])}",
            f"{fmt(row['successful_path_makespan_mean_seconds'])} / {fmt(row['successful_path_makespan_median_seconds'])}",
        ])
    tier_table = Table(tier_rows, colWidths=[2.1*inch, 0.75*inch, 1.2*inch, 1.55*inch, 1.8*inch], repeatRows=1)
    tier_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), BLUE), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 6.2),
        ("ALIGN", (1,1), (-1,-1), "CENTER"), ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#C5D0DC")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F7F9FC")]),
        ("TOPPADDING", (0,0), (-1,-1), 1.1), ("BOTTOMPADDING", (0,0), (-1,-1), 1.1),
    ]))
    story.extend([
        tier_table,
        Spacer(1, 7),
        Paragraph(
            "Interpretation: excluded pilots confirmed easy. Vanilla prioritized/KCBS were fastest and 100% reliable there. "
            "Flow produced the only medium/hard successes for Prioritized and CRRT, at much larger node/waypoint cost. "
            "Centralized CRRT was least reliable within 30 s. All 84 successes passed exact post-run audit.",
            callout,
        ),
    ])
    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    build()
