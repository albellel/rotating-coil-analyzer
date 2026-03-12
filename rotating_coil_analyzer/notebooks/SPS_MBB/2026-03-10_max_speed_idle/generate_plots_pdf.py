"""
Extract all images, tables, and text summaries from executed notebooks
and compile into a single PDF.

Usage:
    python generate_plots_pdf.py
"""
from pathlib import Path
from datetime import datetime
import json
import base64
import io
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT_PDF = HERE / "hysteresis_report.pdf"

NOTEBOOKS = [
    HERE / "hysteresis_analysis_body.ipynb",
    HERE / "hysteresis_analysis_fringe.ipynb",
    HERE / "hysteresis_analysis_comparison.ipynb",
    HERE / "transfer_function.ipynb",
]

NB_TITLES = {
    "hysteresis_analysis_body.ipynb": "Body Segment",
    "hysteresis_analysis_fringe.ipynb": "Fringe Segment",
    "hysteresis_analysis_comparison.ipynb": "Body vs Fringe Comparison",
    "transfer_function.ipynb": "Transfer Function (Body + Fringe)",
}

# Page size (A4 landscape)
PAGE_W, PAGE_H = 11.69, 8.27


def _join(x):
    """Join list of strings if needed."""
    return "".join(x) if isinstance(x, list) else x


def _is_table_text(text: str) -> bool:
    """Heuristic: does this text output look like a summary table or stats block?"""
    lines = text.strip().splitlines()
    if len(lines) < 3:
        return False
    # Pandas text repr: columns aligned with spaces
    # Our printed blocks: contain "===" dividers, "Session", "Plateau", "B1", "b2", "sigma"
    indicators = [
        "B1", "b2", "b3", "sigma", "tau", "R2", "Plateau", "Session",
        "Accommodation", "Eddy", "settled", "Delta", "mean", "std",
        "SFTPRO", "LHC", "idle", "injection", "drift", "convergence",
        "===", "---", "turns", "units",
    ]
    score = sum(1 for ind in indicators if ind.lower() in text.lower())
    return score >= 3


def _html_table_to_text(html: str) -> str:
    """Extract text from a simple HTML table (pandas DataFrame display)."""
    # Remove style tags
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL)
    # Extract rows
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.DOTALL)
    table_rows = []
    for row_html in rows:
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row_html, flags=re.DOTALL)
        # Strip any remaining tags
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        table_rows.append(cells)
    if not table_rows:
        return ""

    # Compute column widths
    n_cols = max(len(r) for r in table_rows)
    col_widths = [0] * n_cols
    for row in table_rows:
        for j, cell in enumerate(row):
            col_widths[j] = max(col_widths[j], len(cell))

    # Format as aligned text
    lines = []
    for i, row in enumerate(table_rows):
        parts = []
        for j in range(n_cols):
            val = row[j] if j < len(row) else ""
            parts.append(val.rjust(col_widths[j]))
        lines.append("  ".join(parts))
        if i == 0:
            lines.append("  ".join("-" * w for w in col_widths))
    return "\n".join(lines)


def extract_outputs(nb_path: Path):
    """Extract all visual outputs from an executed notebook, in cell order.

    Returns a list of (type, data) tuples:
    - ("image", bytes)       -- PNG image
    - ("text", str)          -- text table / summary block
    - ("html_table", str)    -- converted pandas DataFrame
    """
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    outputs = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue

        cell_outputs = cell.get("outputs", [])

        # Collect all stream text for this cell
        stream_text = ""
        has_image = False
        has_html_table = False

        for output in cell_outputs:
            otype = output.get("output_type", "")
            data = output.get("data", {})

            # PNG images
            if "image/png" in data:
                b64 = _join(data["image/png"])
                outputs.append(("image", base64.b64decode(b64)))
                has_image = True

            # HTML tables (pandas DataFrames)
            elif "text/html" in data:
                html = _join(data["text/html"])
                if "<table" in html.lower():
                    text = _html_table_to_text(html)
                    if text.strip():
                        outputs.append(("html_table", text))
                        has_html_table = True

            # Stream output (print statements)
            elif otype == "stream":
                text = _join(output.get("text", ""))
                stream_text += text

        # Only include stream text if it looks like a table/summary
        # and the cell didn't produce images or HTML tables
        if stream_text.strip() and not has_image and _is_table_text(stream_text):
            outputs.append(("text", stream_text.strip()))

    return outputs


def render_text_page(pdf, text: str, title: str = "", fontsize: int = 7):
    """Render a text block as a monospace page in the PDF."""
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax = fig.add_subplot(111)
    ax.axis("off")

    # Title at top
    y_top = 0.97
    if title:
        ax.text(0.5, y_top, title, ha="center", va="top", fontsize=11,
                fontweight="bold", transform=ax.transAxes)
        y_top -= 0.04

    # Truncate very long text to fit on one page
    lines = text.splitlines()
    max_lines = int(PAGE_H * 72 / (fontsize * 1.6))  # approximate
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines)"]

    # Also truncate long lines
    max_chars = int(PAGE_W * 72 / (fontsize * 0.6))
    lines = [l[:max_chars] for l in lines]

    body = "\n".join(lines)
    ax.text(0.02, y_top - 0.01, body, ha="left", va="top", fontsize=fontsize,
            fontfamily="monospace", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", fc="#f8f8f8", ec="#cccccc",
                      alpha=0.9))

    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def render_image_page(pdf, img_bytes: bytes):
    """Render a PNG image as a full page in the PDF."""
    img = Image.open(io.BytesIO(img_bytes))
    w_px, h_px = img.size
    dpi = img.info.get("dpi", (100, 100))
    if isinstance(dpi, tuple):
        dpi_x, dpi_y = dpi
    else:
        dpi_x = dpi_y = dpi
    w_in = max(w_px / dpi_x, 8)
    h_in = max(h_px / dpi_y, 5)
    scale = min(PAGE_W / w_in, PAGE_H / h_in, 1.0)
    w_in *= scale
    h_in *= scale

    fig, ax = plt.subplots(figsize=(w_in, h_in))
    ax.imshow(img)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def main():
    counts = {"image": 0, "text": 0, "html_table": 0}

    # Try main path; if locked, use timestamped fallback
    pdf_path = OUT_PDF
    try:
        open(pdf_path, "wb").close()
    except PermissionError:
        ts = datetime.now().strftime("%H%M%S")
        pdf_path = OUT_PDF.with_stem(OUT_PDF.stem + f"_{ts}")
        print(f"  {OUT_PDF.name} is locked, writing to {pdf_path.name}")

    with PdfPages(str(pdf_path)) as pdf:
        # Global title page
        fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        fig.text(0.5, 0.60, "Hysteresis Analysis", ha="center",
                 fontsize=32, fontweight="bold")
        fig.text(0.5, 0.50, "A-B-B-A Protocol with Standardization",
                 ha="center", fontsize=18)
        fig.text(0.5, 0.40, "MBB max-speed campaign \u2014 2026-03-10",
                 ha="center", fontsize=14, color="grey")
        fig.text(0.5, 0.28,
                 "A1: 200 GeV (full MD1)  \u2192  B1: 26 GeV (flat MD1)\n"
                 "B2: 26 GeV (flat MD1)  \u2192  A2: 200 GeV (full MD1)\n\n"
                 "10\u00d7 standardization cycles (0 \u2192 5781 A \u2192 0) before each session\n"
                 "Body + Fringe segments  \u2022  ~176 RPM  \u2022  Encoder offset corrected",
                 ha="center", fontsize=11, linespacing=1.8)
        fig.text(0.5, 0.05, "Generated from hysteresis_analysis notebooks",
                 ha="center", fontsize=8, color="grey")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for nb_path in NOTEBOOKS:
            if not nb_path.exists():
                print(f"  SKIP (not found): {nb_path.name}")
                continue

            title = NB_TITLES.get(nb_path.name, nb_path.stem)
            items = extract_outputs(nb_path)

            n_img = sum(1 for t, _ in items if t == "image")
            n_tbl = sum(1 for t, _ in items if t in ("text", "html_table"))
            print(f"  {nb_path.name}: {n_img} images, {n_tbl} tables/summaries")

            if not items:
                continue

            # Section title page
            fig = plt.figure(figsize=(PAGE_W, PAGE_H))
            fig.text(0.5, 0.55, title, ha="center", va="center",
                     fontsize=28, fontweight="bold")
            fig.text(0.5, 0.42, f"MBB A-B-B-A Hysteresis \u2014 2026-03-10",
                     ha="center", va="center", fontsize=14, color="grey")
            fig.text(0.5, 0.34, f"{n_img} plots + {n_tbl} tables",
                     ha="center", va="center", fontsize=12, color="grey")
            pdf.savefig(fig)
            plt.close(fig)

            # Render all outputs in order
            for item_type, data in items:
                if item_type == "image":
                    render_image_page(pdf, data)
                    counts["image"] += 1
                elif item_type == "html_table":
                    render_text_page(pdf, data, title=f"{title} \u2014 Summary Table")
                    counts["html_table"] += 1
                elif item_type == "text":
                    render_text_page(pdf, data, title=title)
                    counts["text"] += 1

    total = sum(counts.values())
    print(f"\nPDF written: {pdf_path}")
    print(f"  Images: {counts['image']}")
    print(f"  Text summaries: {counts['text']}")
    print(f"  HTML tables: {counts['html_table']}")
    print(f"  Total pages: {total + 4}")  # +4 for title pages


if __name__ == "__main__":
    main()
