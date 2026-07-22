#!/usr/bin/env python3
"""Create the short Franka edge-bundle dataset design and format PDF."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output" / "pdf" / "franka_edge_bundle_dataset_pipeline.pdf"

NAVY = colors.HexColor("#19324D")
BLUE = colors.HexColor("#2878B5")
LIGHT_BLUE = colors.HexColor("#EAF3F9")
LIGHT_GRAY = colors.HexColor("#F4F6F8")
MID_GRAY = colors.HexColor("#65717E")
GRID = colors.HexColor("#CCD5DE")


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.5)
    canvas.line(0.7 * inch, 0.55 * inch, 7.8 * inch, 0.55 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MID_GRAY)
    canvas.drawString(0.7 * inch, 0.35 * inch, "Franka edge-bundle dataset")
    canvas.drawRightString(7.8 * inch, 0.35 * inch, f"Page {doc.page}")
    canvas.restoreState()


def make_styles():
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCustom",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=23,
            leading=27,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=12,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleCustom",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=16,
            textColor=MID_GRAY,
            spaceAfter=18,
        ),
        "h1": ParagraphStyle(
            "H1Custom",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            textColor=NAVY,
            spaceBefore=10,
            spaceAfter=7,
        ),
        "h2": ParagraphStyle(
            "H2Custom",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=15,
            textColor=BLUE,
            spaceBefore=8,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "BodyCustom",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.4,
            leading=13.3,
            textColor=colors.HexColor("#26323D"),
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "SmallCustom",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.3,
            leading=11.3,
            textColor=colors.HexColor("#34414D"),
        ),
        "callout": ParagraphStyle(
            "CalloutCustom",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10.2,
            leading=14.2,
            textColor=NAVY,
        ),
        "code": ParagraphStyle(
            "CodeCustom",
            fontName="Courier",
            fontSize=7.4,
            leading=10.2,
            textColor=colors.HexColor("#22313F"),
            leftIndent=5,
        ),
        "table_head": ParagraphStyle(
            "TableHead",
            fontName="Helvetica-Bold",
            fontSize=8.3,
            leading=10.5,
            textColor=colors.white,
            alignment=TA_LEFT,
        ),
        "table": ParagraphStyle(
            "TableBody",
            fontName="Helvetica",
            fontSize=7.8,
            leading=10.2,
            textColor=colors.HexColor("#26323D"),
        ),
    }


def P(text, style):
    return Paragraph(text, style)


def bullet(text, styles):
    return Paragraph(
        f"<bullet color='#2878B5'>&bull;</bullet>{text}",
        ParagraphStyle(
            "Bullet",
            parent=styles["body"],
            leftIndent=13,
            firstLineIndent=-8,
            bulletIndent=0,
            spaceAfter=4,
        ),
    )


def styled_table(rows, widths, styles, *, header=True):
    converted = []
    for row_idx, row in enumerate(rows):
        style = styles["table_head"] if header and row_idx == 0 else styles["table"]
        converted.append([P(str(cell), style) for cell in row])
    table = Table(converted, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.4, GRID),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
            ]
        )
    table.setStyle(TableStyle(commands))
    return table


def build_pdf():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = make_styles()
    doc = BaseDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.72 * inch,
        title="Franka Edge-Bundle Dataset Pipeline",
        author="OpenAI Codex",
        subject="Construction and HDF5 format for a self-contained Franka edge-bundle dataset",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates(PageTemplate(id="standard", frames=[frame], onPage=footer))

    story = []
    story.append(P("Franka Edge-Bundle Dataset", styles["title"]))
    story.append(
        P(
            "Brief construction pipeline, confirmed decisions, and self-contained HDF5 format",
            styles["subtitle"],
        )
    )
    callout = Table(
        [[P(
            "Result: 100,000 bundles, 32 variable-length edges per bundle, a 95/5 split, "
            "and every source trajectory stored inside one verified HDF5 file.",
            styles["callout"],
        )]],
        colWidths=[7.05 * inch],
    )
    callout.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.8, BLUE),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.extend([callout, Spacer(1, 10)])

    story.append(P("1. Brief pipeline", styles["h1"]))
    for item in [
        "Treat every non-terminal trajectory timestep as a possible raw edge start. The starting condition is the 14D state c = (q, q_dot), with seven positions and seven velocities.",
        "Sample query conditions from real states already present in datasetFull.h5. This keeps the initial collection phase on the demonstrated data support.",
        "Normalize q by the CuRobo joint ranges and q_dot by the CuRobo velocity limits. Search the normalized 14D space with a KD-tree.",
        "Accept a query when at least 256 raw edge starts lie within the fixed radius. Uniformly cap the neighborhood at 256 candidates when it is larger.",
        "A raw edge is the complete remaining suffix of its source trajectory. Its q, q_dot, and q_ddot sequences all have shape (L, 7), where L is allowed to vary.",
        "Run farthest-point sampling on normalized duration and endpoint features, choose 32 diverse suffixes, and save their raw edge IDs as one bundle.",
        "Randomly split the 100,000 completed bundles into 95,000 training bundles and 5,000 validation bundles.",
    ]:
        story.append(bullet(item, styles))

    story.append(P("2. Confirmed design choices", styles["h1"]))
    decisions = [
        ["Decision", "Value"],
        ["State condition", "(q, q_dot), 14 normalized values"],
        ["Per-step action", "q_ddot, 7 values"],
        ["Edge duration", "Full remaining source suffix; duration bias retained intentionally"],
        ["Edge length", "Variable L; no padding or fixed horizon in this dataset"],
        ["Candidate pool", "256 radius neighbors"],
        ["Bundle size", "32 edges selected by farthest-point sampling"],
        ["Bundle count", "100,000"],
        ["Split", "95,000 train / 5,000 validation"],
        ["Limits", "CuRobo limits used by datasetFull.h5: acceleration 15 and jerk 500 per joint"],
        ["Additional filtering", "None; existing trajectory data is accepted as provided"],
        ["Storage", "One self-contained HDF5 file; no external HDF5 links"],
    ]
    story.append(styled_table(decisions, [2.0 * inch, 5.05 * inch], styles))

    story.append(PageBreak())
    story.append(P("3. Radius measurement and bundle construction", styles["h1"]))
    story.append(
        P(
            "The radius was measured on 20,000 real raw-edge starts. For each sample, the "
            "distance to its 256th nearest neighbor was computed in normalized 14D space. "
            "The fixed radius is the median of those distances:",
            styles["body"],
        )
    )
    radius_box = Table(
        [[P("Median radius", styles["small"]), P("0.933788681848", styles["callout"])],
         [P("Distance metric", styles["small"]), P("Euclidean distance on normalized q and q_dot", styles["small"])],
         [P("Coverage for 256 neighbors", styles["small"]), P("50.0% by construction of the median", styles["small"])],
         [P("Coverage for 32 neighbors", styles["small"]), P("99.84%", styles["small"])]],
        colWidths=[2.2 * inch, 4.85 * inch],
    )
    radius_box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.6, BLUE),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, GRID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend([radius_box, Spacer(1, 8)])
    story.append(
        P(
            "The builder therefore samples unique real states and rejects any query with fewer "
            "than 256 neighbors inside this radius. It accepted 100,000 bundles after 201,031 "
            "attempts, an acceptance rate of 49.74%.",
            styles["body"],
        )
    )
    percentiles = [
        ["Quantity", "5th percentile", "Median", "95th percentile"],
        ["256th-neighbor radius", "0.6566", "0.9338", "1.2073"],
        ["Accepted neighborhood size before cap", "284", "987", "3,472"],
        ["Selected edge length L", "27", "107", "173"],
        ["Selected edge duration (seconds)", "0.52", "2.12", "3.44"],
    ]
    story.append(styled_table(percentiles, [2.5 * inch, 1.52 * inch, 1.52 * inch, 1.52 * inch], styles))

    story.append(P("Farthest-point sampling features", styles["h2"]))
    story.append(
        P(
            "FPS preserves the SOC bundle-selection structure while adapting the feature vector "
            "to variable-length Franka suffixes. It uses 0.5 x normalized duration, seven "
            "normalized delta-q values, and seven normalized final-velocity values. The 32 "
            "selected raw edge IDs are sorted numerically for deterministic storage.",
            styles["body"],
        )
    )

    story.append(P("Raw edge definition", styles["h2"]))
    raw_edge_rows = [
        ["Stored field", "Shape", "Meaning"],
        ["start_q", "(7,)", "Matched joint configuration"],
        ["start_dq", "(7,)", "Matched joint velocity"],
        ["q suffix", "(L, 7)", "positions[start_index:]"],
        ["dq suffix", "(L, 7)", "velocities[start_index:]"],
        ["ddq suffix", "(L, 7)", "accelerations[start_index:]"],
        ["duration", "scalar", "(L - 1) x 0.02 seconds"],
        ["delta_q", "(7,)", "final_q - start_q"],
        ["final_q / final_dq", "(7,) each", "Suffix endpoint state"],
        ["source IDs", "scalars", "Trajectory index and starting timestep"],
    ]
    story.append(styled_table(raw_edge_rows, [1.7 * inch, 1.15 * inch, 4.2 * inch], styles))

    story.append(PageBreak())
    story.append(P("4. Self-contained HDF5 format", styles["h1"]))
    story.append(
        P(
            "Output file: FlowMRMP/data/franka_edge_bundle_k32_n100000.h5. The source "
            "trajectories are copied once, and every bundle refers to them through compact raw "
            "edge IDs. This retains variable L without duplicating the same suffix arrays across "
            "many bundles.",
            styles["body"],
        )
    )
    schema = """/
|-- conds_train                         float32 (95000, 14)
|-- conds_raw_train                     float32 (95000, 14)
|-- query_source_edge_ids_train         int64   (95000,)
|-- source_edge_ids_train               int64   (95000, 32)
|-- conds_val                           float32 (5000, 14)
|-- conds_raw_val                       float32 (5000, 14)
|-- query_source_edge_ids_val           int64   (5000,)
|-- source_edge_ids_val                 int64   (5000, 32)
|-- metadata_json                       scalar UTF-8 JSON
|-- raw_edges/
|   |-- trajectory_index                int32   (1211710,)
|   |-- start_index                     int32   (1211710,)
|   |-- trajectory_length               int32   (1211710,)
|   |-- duration                        float32 (1211710,)
|   |-- start_q, start_dq               float32 (1211710, 7)
|   |-- delta_q                         float32 (1211710, 7)
|   `-- final_q, final_dq               float32 (1211710, 7)
`-- source_trajectories/
    |-- trajectory_names                UTF-8   (10000,)
    `-- 000000 ... 011877/
        |-- positions                   float32 (T, 7)
        |-- velocities                  float32 (T, 7)
        `-- accelerations               float32 (T, 7)"""
    code_box = Table(
        [[Preformatted(schema, styles["code"])]] ,
        colWidths=[7.05 * inch],
    )
    code_box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GRAY),
                ("BOX", (0, 0), (-1, -1), 0.5, GRID),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(code_box)

    story.append(P("How one bundle edge is resolved", styles["h2"]))
    resolution = """edge_id = source_edge_ids_train[bundle_index, edge_slot]
traj_i  = raw_edges/trajectory_index[edge_id]
start   = raw_edges/start_index[edge_id]
name    = source_trajectories/trajectory_names[traj_i]

q_edge   = source_trajectories/name/positions[start:]
dq_edge  = source_trajectories/name/velocities[start:]
ddq_edge = source_trajectories/name/accelerations[start:]"""
    story.append(Preformatted(resolution, styles["code"]))
    story.append(
        P(
            "All referenced arrays live in the same HDF5 file. The original datasetFull.h5 is "
            "not needed after the new file has been created.",
            styles["body"],
        )
    )

    story.append(P("Normalization and metadata", styles["h2"]))
    story.append(
        P(
            "conds_train and conds_val use q_norm = 2(q - q_lower)/(q_upper - q_lower) - 1 "
            "and dq_norm = dq/dq_max. The raw conditions are also retained. metadata_json records "
            "all limits, formulas, radius statistics, FPS features, random seed, split sizes, and "
            "bundle-generation statistics.",
            styles["body"],
        )
    )

    story.append(PageBreak())
    story.append(P("5. Verification completed", styles["h1"]))
    story.append(
        P(
            "The completed file was reopened and checked independently after generation. "
            "The checks below cover structure, reference resolution, neighborhood locality, "
            "metadata consistency, split integrity, and file independence.",
            styles["body"],
        )
    )
    verification = [
        ["Check", "Result"],
        ["File size", "273.025 MiB"],
        ["Raw variable-length edges", "1,211,710"],
        ["Selected-edge locality", "All sampled selected starts were within radius; max excess 3.5e-8"],
        ["Suffix resolution", "Exact q, q_dot, q_ddot shapes and starts for 2,000 sampled edges"],
        ["Endpoint metadata", "Zero sampled endpoint error"],
        ["Duration metadata", "Maximum sampled floating-point error 1.15e-7 seconds"],
        ["Train/validation queries", "100,000 unique query states; no overlap between splits"],
        ["External HDF5 links", "0"],
        ["SHA-256", "7364fce48d6c7901175e86dd532b9a58a21a92ae988ae0a4fff8b177c4c8d360"],
    ]
    story.append(styled_table(verification, [2.0 * inch, 5.05 * inch], styles))

    story.append(P("Delivered artifacts", styles["h2"]))
    artifacts = [
        ["Artifact", "Path"],
        ["Edge-bundle data", "FlowMRMP/data/franka_edge_bundle_k32_n100000.h5"],
        ["Dataset generator", "FlowMRMP/scripts/create_franka_edge_bundle_dataset.py"],
        ["This document", "output/pdf/franka_edge_bundle_dataset_pipeline.pdf"],
    ]
    story.append(styled_table(artifacts, [1.65 * inch, 5.4 * inch], styles))

    story.append(P("Use notes", styles["h2"]))
    for item in [
        "Variable L is preserved. No padding, truncation, or fixed training representation has been imposed.",
        "A bundle is read through source_edge_ids_train or source_edge_ids_val, then resolved through raw_edges and source_trajectories.",
        "The copied source trajectories make the output independent of the original datasetFull.h5 file.",
        "Future model code may choose its own batching or sequence representation without regenerating this dataset.",
    ]:
        story.append(bullet(item, styles))
    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    build_pdf()
