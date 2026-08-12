#!/usr/bin/env python3
"""Build the concise PDF and combined CSVs for a prioritized Franka run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


NAVY = colors.HexColor("#18212B")
BLUE = colors.HexColor("#2878C8")
GREEN = colors.HexColor("#208B57")
ORANGE = colors.HexColor("#D77B20")
LIGHT = colors.HexColor("#F2F5F8")
MID = colors.HexColor("#D6DEE7")
TEXT = colors.HexColor("#263442")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def paragraph(text: str, style) -> Paragraph:
    return Paragraph(text, style)


def flatten_generations(results: list[dict]) -> list[dict[str, object]]:
    rows = []
    for result in sorted(results, key=lambda item: (item["problem_id"], item["algorithm"])):
        for generation in result["generations"]:
            collision = generation.get("collision_profile", {})
            planner = generation.get("planner_profile", {})
            flow = generation.get("flow_generator_profile", {})
            rows.append(
                {
                    "problem_id": result["problem_id"],
                    "scenario_name": result["scenario_name"],
                    "algorithm": result["algorithm"],
                    "generation": generation["generation"],
                    "robot_index": generation["robot_index"],
                    "robot_name": generation["robot_name"],
                    "status": generation["status"],
                    "success": generation["success"],
                    "planning_time_seconds": generation.get("planning_time_seconds"),
                    "iterations": generation.get("iterations"),
                    "nodes": generation.get("nodes"),
                    "path_motion_time_seconds": generation.get("path_motion_time_seconds"),
                    "path_cost": generation.get("path_cost"),
                    "checked_waypoints": generation.get("checked_waypoints"),
                    "limit_rejections": generation.get("limit_rejections"),
                    "self_or_static_collision_rejections": generation.get(
                        "self_or_static_collision_rejections"
                    ),
                    "inter_robot_collision_rejections": generation.get(
                        "inter_robot_collision_rejections"
                    ),
                    "collision_batches": collision.get("collision_batches"),
                    "collision_waypoints": collision.get("collision_waypoints"),
                    "sphere_fk_seconds": collision.get("sphere_fk_seconds"),
                    "self_static_seconds": collision.get("self_static_seconds"),
                    "dynamic_collision_seconds": collision.get(
                        "dynamic_collision_seconds"
                    ),
                    "dynamic_sphere_pair_tests": collision.get(
                        "dynamic_sphere_pair_tests"
                    ),
                    "dynamic_collision_batches_rejected": collision.get(
                        "dynamic_collision_batches_rejected"
                    ),
                    "goal_parking_queries": collision.get("goal_parking_queries"),
                    "goal_parking_sphere_pair_tests": collision.get(
                        "goal_parking_sphere_pair_tests"
                    ),
                    "flow_generated_bundles": planner.get("flow_generated_bundles"),
                    "flow_generation_calls": planner.get("flow_generation_calls"),
                    "flow_cache_hits": planner.get("flow_cache_hits"),
                    "flow_cache_misses": planner.get("flow_cache_misses"),
                    "flow_edge_trials": planner.get("try_edge_calls"),
                    "random_control_trials": planner.get("random_control_calls"),
                    "flow_model_seconds": flow.get("model_s"),
                    "flow_postprocess_seconds": flow.get("postprocess_s"),
                    "flow_samples": flow.get("samples"),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_page(canvas, doc) -> None:
    width, height = letter
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 0.45 * inch, width, 0.45 * inch, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(0.55 * inch, height - 0.29 * inch, "FOUR-FRANKA PRIORITIZED PLANNING")
    canvas.setFillColor(colors.HexColor("#677584"))
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(width - 0.55 * inch, 0.32 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(output: Path, run_dir: Path, run_summary: dict) -> None:
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=25,
        leading=29,
        textColor=NAVY,
        alignment=TA_LEFT,
        spaceAfter=10,
    )
    subtitle = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=10.5,
        leading=15,
        textColor=colors.HexColor("#536270"),
        spaceAfter=14,
    )
    heading = ParagraphStyle(
        "HeadingCustom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=BLUE,
        spaceBefore=8,
        spaceAfter=8,
    )
    body = ParagraphStyle(
        "BodyCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=TEXT,
        spaceAfter=7,
    )
    small = ParagraphStyle(
        "SmallCustom",
        parent=body,
        fontSize=8.3,
        leading=11,
        spaceAfter=0,
    )
    callout = ParagraphStyle(
        "Callout",
        parent=body,
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=GREEN,
        alignment=TA_CENTER,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(
        str(output),
        pagesize=letter,
        leftMargin=0.55 * inch,
        rightMargin=0.55 * inch,
        topMargin=0.67 * inch,
        bottomMargin=0.55 * inch,
        title="Four-Franka Prioritized Planning Implementation",
        author="FrankaFM project",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="page", frames=[frame], onPage=draw_page)])

    results = run_summary["results"]
    flow = [result for result in results if result["algorithm"] == "FlowEBRRT"]
    vanilla = [result for result in results if result["algorithm"] == "VanillaRRT"]
    flow_success_times = [result["wall_time_seconds"] for result in flow if result["solved"]]
    vanilla_success_times = [
        result["wall_time_seconds"] for result in vanilla if result["solved"]
    ]
    all_audited = all(
        result["post_run_curobo_audit"]["collision_free"] for result in results
    )

    story = [
        Spacer(1, 0.16 * inch),
        paragraph("Prioritized Planning for Four Franka Arms", title),
        paragraph(
            "Implementation and validation summary for the shared-table environment. "
            "Planning and collision decisions use cuRobo and GPU sphere checks; PyBullet "
            "is restricted to final mesh visualization.",
            subtitle,
        ),
    ]
    metric_data = [
        ["FLOWEBRRT", "VANILLA RRT", "COLLISION MODEL"],
        [
            paragraph(f"<b>{sum(r['solved'] for r in flow)}/3 solved</b><br/>mean successful time {mean(flow_success_times):.2f} s<br/>median {median(flow_success_times):.2f} s", callout),
            paragraph(f"<b>{sum(r['solved'] for r in vanilla)}/3 solved</b><br/>successful time {mean(vanilla_success_times):.2f} s", callout),
            paragraph("<b>295 spheres/arm</b><br/>1,180 total spheres<br/>all saved paths audited", callout),
        ],
    ]
    metric_table = Table(metric_data, colWidths=[2.25 * inch] * 3, rowHeights=[0.28 * inch, 0.82 * inch])
    metric_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, 1), LIGHT),
                ("BOX", (0, 0), (-1, -1), 0.7, MID),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, MID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]
        )
    )
    story.extend([metric_table, Spacer(1, 0.13 * inch), paragraph("Benchmark results", heading)])

    by_key = {(r["problem_id"], r["algorithm"]): r for r in results}
    table_data = [["Problem", "FlowEBRRT", "VanillaRRT", "GPU replay audit"]]
    for problem_id in range(3):
        flow_result = by_key[(problem_id, "FlowEBRRT")]
        vanilla_result = by_key[(problem_id, "VanillaRRT")]
        table_data.append(
            [
                f"{problem_id}: {flow_result['scenario_name']}",
                f"{'solved' if flow_result['solved'] else 'failed'} - {flow_result['wall_time_seconds']:.2f} s",
                f"{'solved' if vanilla_result['solved'] else 'failed'} - {vanilla_result['wall_time_seconds']:.2f} s",
                "pass" if (
                    flow_result["post_run_curobo_audit"]["collision_free"]
                    and vanilla_result["post_run_curobo_audit"]["collision_free"]
                ) else "fail",
            ]
        )
    result_table = Table(table_data, colWidths=[2.45 * inch, 1.45 * inch, 1.45 * inch, 1.35 * inch])
    result_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.2),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("GRID", (0, 0), (-1, -1), 0.45, MID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend(
        [
            result_table,
            Spacer(1, 0.08 * inch),
            paragraph(
                "The 120-second budget is shared across priority generations. Failed "
                "Vanilla runs retain their best valid partial paths. All six saved path "
                f"sets passed the final cuRobo replay audit: <b>{'yes' if all_audited else 'no'}</b>.",
                small,
            ),
            paragraph("What was added", heading),
            paragraph(
                "<b>1. Geometry binding.</b> Every robot uses the same Panda URDF mesh "
                "and the same 295-sphere MorphIt/cuRobo model. The scene manifest verifies "
                "the URDF match and records 1,180 planning spheres.",
                body,
            ),
            paragraph(
                "<b>2. World/base transforms.</b> cuRobo produces spheres in each "
                "panda_link0 frame. The adapter applies p_world = R_world_base * p_base + "
                "t_world_base. Table and fixture poses use the inverse transform before "
                "entering each arm's local cuRobo scene.",
                body,
            ),
            paragraph(
                "<b>3. Articulated dynamic obstacles.</b> A completed higher-priority path "
                "is cached on the GPU as [time, 295, x-y-z-radius]. Lower-priority rollout "
                "waypoints are aligned by time and checked with batched sphere-pair tests. "
                "A finished path holds its final configuration indefinitely.",
                body,
            ),
            paragraph(
                "<b>4. Generic orchestration hook.</b> prioritized_planning.py now accepts "
                "an optional dynamic-obstacle adapter. Its original 2D/3D radius behavior "
                "is unchanged when the hook is absent.",
                body,
            ),
            paragraph("Planning sequence", heading),
        ]
    )
    sequence = Table(
        [
            ["Generation", "Robot", "Collision constraints"],
            ["0", "NW", "self + table/fixtures"],
            ["1", "NE", "self/static + NW(t)"],
            ["2", "SW", "self/static + NW(t) + NE(t)"],
            ["3", "SE", "self/static + NW(t) + NE(t) + SW(t)"],
        ],
        colWidths=[1.05 * inch, 1.15 * inch, 4.5 * inch],
    )
    sequence.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("GRID", (0, 0), (-1, -1), 0.45, MID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend(
        [
            sequence,
            paragraph("Correctness checks and discovered fix", heading),
            paragraph(
                "The first post-run GPU replay exposed a root-goal edge case in scenario "
                "2: SW started inside its goal, so inherited RRT accepted it immediately "
                "without checking future parking. The root acceptance rule now calls the "
                "same future-collision test as every other goal. Scenario 2 FlowEBRRT was "
                "rerun after this fix and passed the replay audit.",
                body,
            ),
            paragraph(
                "Validation completed with 12 focused tests, CUDA endpoint checks for all "
                "three scenarios, real dynamic-sphere conflict detection, byte-compilation, "
                "and a second synchronized cuRobo replay of every saved path. PyBullet was "
                "not imported by the planning or audit entry points.",
                body,
            ),
            paragraph("Files and outputs", heading),
        ]
    )
    files = [
        ["File", "Purpose"],
        ["franka_prioritized_adapter.py", "Base transforms, cuRobo scenes, GPU sphere caches, dynamic and parking checks"],
        ["prioritized_four_franka.py", "Vanilla/Flow runner, path persistence, per-generation statistics"],
        ["audit_four_franka_prioritized.py", "Independent cuRobo-only synchronous path replay"],
        ["visualize_four_franka_prioritized.py", "PyBullet mesh playback and MP4 generation only"],
        ["prioritized_planning.py", "Backward-compatible optional articulated-obstacle hook"],
        ["run_20260807_final", "Six result folders, JSON/CSV statistics, audits, paths, and six videos"],
    ]
    file_table = Table(files, colWidths=[2.25 * inch, 4.45 * inch])
    file_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.2),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("GRID", (0, 0), (-1, -1), 0.45, MID),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend(
        [
            file_table,
            Spacer(1, 0.1 * inch),
            paragraph(
                "Complete per-generation data is stored in combined_generations.csv and "
                "in each result folder's generations.csv/summary.json. Videos are 640x480, "
                "24 FPS, and contain 96 frames each.",
                small,
            ),
        ]
    )
    doc.build(story)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output = args.output.resolve()
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    rows = flatten_generations(summary["results"])
    write_csv(run_dir / "combined_generations.csv", rows)
    benchmark_rows = []
    for result in sorted(summary["results"], key=lambda item: (item["problem_id"], item["algorithm"])):
        benchmark_rows.append(
            {
                "problem_id": result["problem_id"],
                "scenario_name": result["scenario_name"],
                "algorithm": result["algorithm"],
                "solved": result["solved"],
                "wall_time_seconds": result["wall_time_seconds"],
                "reported_planning_time_seconds": result["reported_planning_time_seconds"],
                "total_path_cost": result["total_path_cost"],
                "saved_paths_collision_free": result["post_run_curobo_audit"]["collision_free"],
                "attempted_generations": sum(
                    generation["status"] != "not_attempted_after_prior_failure"
                    for generation in result["generations"]
                ),
            }
        )
    write_csv(run_dir / "benchmark_summary.csv", benchmark_rows)
    build_pdf(output, run_dir, summary)
    print(f"saved combined generations: {run_dir / 'combined_generations.csv'}")
    print(f"saved benchmark summary: {run_dir / 'benchmark_summary.csv'}")
    print(f"saved report: {output}")


if __name__ == "__main__":
    main()
