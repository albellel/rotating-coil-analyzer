#!/usr/bin/env python
"""Build the MC62 staircase presentation from executed notebooks.

Extracts PNG images from Jupyter notebooks that have already been
executed in-place (with outputs embedded), then assembles a PPTX
using the CERN template.

Usage
-----
    python build_presentation.py           # build PPTX
    python build_presentation.py --count   # just count images per notebook

Prerequisites
-------------
- python-pptx, Pillow
- Notebooks executed in-place (via nbconvert --execute --inplace)
"""
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN

# -- Paths -----------------------------------------------------------------
REPO = Path(r"C:\Users\albellel\python-projects\rotating-coil-analyzer")
NB_DIR = REPO / "rotating_coil_analyzer" / "notebooks" / "LEAR_MC62"
TEMPLATE_PPTX = NB_DIR / "MC62_2Hz_staircase_presentation.pptx"
OUTPUT_PPTX = NB_DIR / "MC62_staircase_presentation.pptx"

# Executed notebooks -- in-place in the repository
EXEC_NBS = {
    "analysis_01": NB_DIR / "analysis" / "2026-02-11_01_staircase_with_shims.ipynb",
    "analysis_02": NB_DIR / "analysis" / "2026-02-12_02_staircase_without_shims.ipynb",
    "analysis_03": NB_DIR / "analysis" / "2026-02-16_03_staircase_2Hz.ipynb",
    "analysis_04": NB_DIR / "analysis" / "2026-02-17_04_staircase_2Hz_morning.ipynb",
    "eddy_01": NB_DIR / "eddy_current" / "2026-02-11_01_staircase_with_shims.ipynb",
    "eddy_02": NB_DIR / "eddy_current" / "2026-02-12_02_staircase_without_shims.ipynb",
    "eddy_03": NB_DIR / "eddy_current" / "2026-02-16_03_staircase_2Hz.ipynb",
    "compare_01v02": NB_DIR / "comparison" / "2026-02-12_01_vs_02_shims_effect.ipynb",
    "compare_03v04": NB_DIR / "comparison" / "2026-02-17_03_vs_04_reproducibility.ipynb",
}


# -- Image extraction ------------------------------------------------------

def extract_images(nb_path: Path) -> list[bytes]:
    """Extract all PNG images from an executed notebook, in cell order."""
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    images = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            if "image/png" in data:
                b64 = data["image/png"]
                if isinstance(b64, list):
                    b64 = "".join(b64)
                images.append(base64.b64decode(b64))
    return images


def extract_text_output(nb_path: Path, cell_index: int) -> str:
    """Extract text output from a specific code cell (0-indexed among code cells)."""
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    code_idx = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        if code_idx == cell_index:
            texts = []
            for output in cell.get("outputs", []):
                if output.get("output_type") in ("stream", "execute_result"):
                    text = output.get("text", "")
                    if isinstance(text, list):
                        text = "".join(text)
                    texts.append(text)
                elif output.get("output_type") == "display_data":
                    text = output.get("data", {}).get("text/plain", "")
                    if isinstance(text, list):
                        text = "".join(text)
                    texts.append(text)
            return "\n".join(texts)
        code_idx += 1
    return ""


def count_images_all():
    """Print image counts for all notebooks (use with --count)."""
    for key, path in EXEC_NBS.items():
        if path.exists():
            imgs = extract_images(path)
            print(f"  {key}: {len(imgs)} images")
        else:
            print(f"  {key}: NOT FOUND")


# -- PPTX helpers ----------------------------------------------------------

# Slide dimensions (CERN template standard: 13.33" x 7.5")
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

# Layout indices (from template inspection)
LAYOUT_TITLE = 1       # 'Title Slide'
LAYOUT_CONTENT = 3     # 'Content'
LAYOUT_CHAPTER = 17    # 'Chapter Header'
LAYOUT_TITLE_ONLY = 18 # 'Title Only'
LAYOUT_BLANK = 19      # 'Blank'
LAYOUT_LAST = 20       # 'Last slide'


def add_title_slide(prs, title, subtitle):
    """Add a title slide."""
    layout = prs.slide_layouts[LAYOUT_TITLE]
    slide = prs.slides.add_slide(layout)
    for shape in slide.placeholders:
        if shape.placeholder_format.idx == 0:
            shape.text = title
        elif shape.placeholder_format.idx == 1:
            shape.text = subtitle
    return slide


def add_chapter_slide(prs, chapter_num, title, subtitle=""):
    """Add a chapter header slide."""
    layout = prs.slide_layouts[LAYOUT_CHAPTER]
    slide = prs.slides.add_slide(layout)
    for shape in slide.placeholders:
        if shape.placeholder_format.idx == 0:
            shape.text = f"{chapter_num}. {title}"
        elif shape.placeholder_format.idx == 1:
            shape.text = subtitle
    return slide


def add_image_slide(prs, title, image_bytes, subtitle=""):
    """Add a slide with a title and a large image."""
    layout = prs.slide_layouts[LAYOUT_TITLE_ONLY]
    slide = prs.slides.add_slide(layout)

    # Set title
    for shape in slide.placeholders:
        if shape.placeholder_format.idx == 0:
            shape.text = title

    # Add image -- fill most of the slide below the title
    img_stream = io.BytesIO(image_bytes)
    max_w = Inches(12.7)
    max_h = Inches(5.8)
    top = Inches(1.4)

    # Get image aspect ratio
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes))
    img_w, img_h = img.size
    aspect = img_w / img_h

    # Fit within bounds
    if max_w / max_h > aspect:
        height = max_h
        width = int(height * aspect)
    else:
        width = max_w
        height = int(width / aspect)

    # Center horizontally
    left = (SLIDE_W - width) // 2

    slide.shapes.add_picture(img_stream, left, top, width, height)

    # Add subtitle if provided
    if subtitle:
        txBox = slide.shapes.add_textbox(
            Inches(0.5), Inches(7.0), Inches(12), Inches(0.4),
        )
        tf = txBox.text_frame
        tf.text = subtitle
        for p in tf.paragraphs:
            p.font.size = Pt(10)
            p.font.italic = True
            p.alignment = PP_ALIGN.LEFT

    return slide


def add_text_slide(prs, title, body_text):
    """Add a content slide with title and body text."""
    layout = prs.slide_layouts[LAYOUT_CONTENT]
    slide = prs.slides.add_slide(layout)
    for shape in slide.placeholders:
        if shape.placeholder_format.idx == 0:
            shape.text = title
        elif shape.placeholder_format.idx == 1:
            shape.text = body_text
    return slide


def add_last_slide(prs):
    """Add the CERN 'last slide'."""
    layout = prs.slide_layouts[LAYOUT_LAST]
    prs.slides.add_slide(layout)


def _safe(imgs, idx):
    """Return image at index if it exists, else None."""
    return imgs[idx] if len(imgs) > idx else None


# -- Main build ------------------------------------------------------------

def build():
    """Build the complete presentation."""
    # Load template (use existing PPTX for the master/layouts)
    prs = Presentation(str(TEMPLATE_PPTX))

    # Remove all existing slides
    from lxml import etree
    while len(prs.slides._sldIdLst) > 0:
        sldId = prs.slides._sldIdLst[0]
        rId = sldId.get(etree.QName(
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships", "id"))
        if rId:
            prs.part.drop_rel(rId)
        prs.slides._sldIdLst.remove(sldId)

    # Extract images from executed notebooks
    print("Extracting images from executed notebooks...")
    images = {}
    for key, path in EXEC_NBS.items():
        if path.exists():
            imgs = extract_images(path)
            images[key] = imgs
            print(f"  {key}: {len(imgs)} images")
        else:
            images[key] = []
            print(f"  {key}: NOT FOUND at {path}")

    # ==================================================================
    # Slide 1: Title
    # ==================================================================
    add_title_slide(
        prs,
        "MC62 Rotating Coil Measurement Campaign",
        "LEAR C-shaped Dipole\n"
        "Staircase Tests -- Feb 11-17, 2026\n"
        "A. Bellelli -- CERN",
    )

    # ==================================================================
    # Chapter 1: Test 01 -- 1 Hz Staircase with Shims (Feb 11)
    # ==================================================================
    add_chapter_slide(prs, 1,
        "Test 01\nFeb 11, 2026 -- With Shims",
        "1 Hz staircase, 0->+200->0->-200->0 A in 20 A steps\n"
        "41 plateaus, 350 turns each at -60 rpm\n"
        "Integral (R45) + Central (DQ) PCBs")

    # analysis_01 (run-based, 9 images):
    # 0: current profile, 1: settling visualization
    # 2: B1 hysteresis, 3: b2 hysteresis, 4: b3 hysteresis, 5: TF hysteresis
    # 6: full individual hyst, 7: zoomed 2x2 (I!=0), 8: multipole spectrum
    a01 = images.get("analysis_01", [])
    if _safe(a01, 7):
        add_image_slide(prs, "Test 01 -- Harmonic Hysteresis (I != 0)",
                        a01[7],
                        "B1, b2, b3, TF vs current -- excluding I = 0 plateaus")
    if _safe(a01, 2):
        add_image_slide(prs, "Test 01 -- B1 vs Current (Full Range)",
                        a01[2])
    if _safe(a01, 8):
        add_image_slide(prs, "Test 01 -- Multipole Spectrum at Peak Current",
                        a01[8])

    # ==================================================================
    # Chapter 2: Test 02 -- 1 Hz Staircase without Shims (Feb 12)
    # ==================================================================
    add_chapter_slide(prs, 2,
        "Test 02\nFeb 12, 2026 -- Without Shims",
        "1 Hz staircase, same cycle as Test 01\n"
        "Shims removed between Test 01 and Test 02")

    # analysis_02: same layout as analysis_01 (9 images)
    a02 = images.get("analysis_02", [])
    if _safe(a02, 7):
        add_image_slide(prs, "Test 02 -- Harmonic Hysteresis (I != 0)",
                        a02[7],
                        "B1, b2, b3, TF vs current -- excluding I = 0 plateaus")
    if _safe(a02, 2):
        add_image_slide(prs, "Test 02 -- B1 vs Current (Full Range)",
                        a02[2])
    if _safe(a02, 8):
        add_image_slide(prs, "Test 02 -- Multipole Spectrum at Peak Current",
                        a02[8])

    # ==================================================================
    # Chapter 3: Shims Effect -- 01 vs 02
    # ==================================================================
    add_chapter_slide(prs, 3,
        "Shims Effect\nTest 01 vs Test 02",
        "Quantify the effect of removing iron shims\n"
        "on B1, b2, b3, and the transfer function")

    # compare_01v02 images:
    # 0: B1 overlay, 1: b2 overlay, 2: b3 overlay, 3: TF overlay
    # 4: difference bar charts (2x2), 5: multipole spectrum
    c12 = images.get("compare_01v02", [])
    if _safe(c12, 0):
        add_image_slide(prs, "Shims Effect -- B1 vs Current",
                        c12[0],
                        "Circles = with shims, Squares = without shims")
    if _safe(c12, 1):
        add_image_slide(prs, "Shims Effect -- b2 (Quadrupole) vs Current",
                        c12[1])
    if _safe(c12, 2):
        add_image_slide(prs, "Shims Effect -- b3 (Sextupole) vs Current",
                        c12[2])
    if _safe(c12, 4):
        add_image_slide(prs, "Shims Effect -- Per-Level Differences",
                        c12[4],
                        "\u0394 = Test 02 (no shims) \u2212 Test 01 (with shims)")
    if _safe(c12, 5):
        add_image_slide(prs, "Shims Effect -- Multipole Spectrum Comparison",
                        c12[5])

    # ==================================================================
    # Chapter 4: Test 03 -- 2 Hz Staircase (Feb 16 afternoon)
    # ==================================================================
    add_chapter_slide(prs, 4,
        "Test 03\nFeb 16, 2026 -- Afternoon",
        "2 Hz staircase (120 rpm), 512 samples/turn\n"
        "~800 turns per plateau, streaming binary format\n"
        "No shims, same current cycle")

    # analysis_03 (streaming, 13 images):
    # 0: per-turn statistics, 1: annotated current profile
    # 2: turn classification map, 3: harmonics vs turn (with classification)
    # 4: all-turns field-vs-time (plot 1), 5: all-turns (plot 2)
    # 6: B1 hyst, 7: b2 hyst, 8: b3 hyst, 9: TF hyst
    # 10: zoomed 2x2 (I!=0), 11: multipole spectrum, 12: FFMM parity
    a03 = images.get("analysis_03", [])
    if _safe(a03, 2):
        add_image_slide(prs, "Test 03 -- Turn Classification Map",
                        a03[2],
                        "Green = plateau, Orange = ramp, Grey = precycle")
    if _safe(a03, 10):
        add_image_slide(prs, "Test 03 -- Harmonic Hysteresis (I != 0)",
                        a03[10],
                        "B1, b2, b3, TF vs current -- excluding I = 0 plateaus")
    if _safe(a03, 1):
        add_image_slide(prs, "Test 03 -- Current Profile & Plateau Detection",
                        a03[1])
    if _safe(a03, 6):
        add_image_slide(prs, "Test 03 -- B1 vs Current (Full Range)",
                        a03[6])
    if _safe(a03, 11):
        add_image_slide(prs, "Test 03 -- Multipole Spectrum at Peak Current",
                        a03[11])
    if _safe(a03, 3):
        add_image_slide(prs, "Test 03 -- Harmonics vs Turn Number",
                        a03[3],
                        "B1, b2, b3 per turn with classification overlay")
    if _safe(a03, 4):
        add_image_slide(prs, "Test 03 -- All Turns: Field vs Time",
                        a03[4],
                        "Including ramp turns -- full current-harmonic correlation")

    # Eddy current results for test 03
    e03 = images.get("eddy_03", [])
    # eddy_03 (7 images): 0: settling curves, 1: representative fits,
    # 2: tau vs current, 3: settling bias (3-panel),
    # 4: sensitivity study, 5: precycle, 6: double-exp comparison
    if _safe(e03, 2):
        add_image_slide(prs, "Test 03 -- Eddy Current: \u03c4 vs Current",
                        e03[2],
                        "\u03c4 decreases with |I| (permeability effect)")
    if _safe(e03, 4):
        add_image_slide(prs, "Test 03 -- Sensitivity Study: N_LAST_TURNS",
                        e03[4])
    if _safe(e03, 6):
        add_image_slide(prs, "Test 03 -- Single vs Double Exponential Fit",
                        e03[6],
                        "R\u00b2 comparison, \u0394R\u00b2 distribution, "
                        "\u03c4\u2081 vs \u03c4\u2082")

    # ==================================================================
    # Chapter 5: Test 04 -- 2 Hz Staircase (Feb 17 morning)
    # ==================================================================
    add_chapter_slide(prs, 5,
        "Test 04\nFeb 17, 2026 -- Morning",
        "2 Hz staircase, repeat of Test 03\n"
        "Morning measurement for reproducibility check\n"
        "N_SKIP_END = 20 (trim last 20 turns)")

    # analysis_04 (streaming, 13 images): same layout as analysis_03
    a04 = images.get("analysis_04", [])
    if _safe(a04, 2):
        add_image_slide(prs, "Test 04 -- Turn Classification Map",
                        a04[2],
                        "Green = plateau, Orange = ramp, Grey = precycle")
    if _safe(a04, 10):
        add_image_slide(prs, "Test 04 -- Harmonic Hysteresis (I != 0)",
                        a04[10],
                        "B1, b2, b3, TF vs current -- excluding I = 0 plateaus")
    if _safe(a04, 6):
        add_image_slide(prs, "Test 04 -- B1 vs Current (Full Range)",
                        a04[6])
    if _safe(a04, 3):
        add_image_slide(prs, "Test 04 -- Harmonics vs Turn Number",
                        a04[3],
                        "B1, b2, b3 per turn with classification overlay")
    if _safe(a04, 4):
        add_image_slide(prs, "Test 04 -- All Turns: Field vs Time",
                        a04[4],
                        "Including ramp turns -- full current-harmonic correlation")

    # ==================================================================
    # Chapter 6: Reproducibility -- 03 vs 04
    # ==================================================================
    add_chapter_slide(prs, 6,
        "Reproducibility\nTest 03 vs Test 04",
        "Day-to-day: Feb 16 afternoon vs Feb 17 morning\n"
        "Identical hardware, same cycle, same settings")

    # compare_03v04 images:
    # 0: B1 overlay, 1: b2 overlay, 2: b3 overlay, 3: TF overlay
    # 4: difference bars, 5: multipole spectrum, 6: hysteresis width
    # 7: B1 noise scatter
    c34 = images.get("compare_03v04", [])
    if _safe(c34, 0):
        add_image_slide(prs, "Reproducibility -- B1 vs Current",
                        c34[0],
                        "Circles = Test 03, Squares = Test 04")
    if _safe(c34, 1):
        add_image_slide(prs, "Reproducibility -- b2 vs Current",
                        c34[1])
    if _safe(c34, 2):
        add_image_slide(prs, "Reproducibility -- b3 vs Current",
                        c34[2])
    if _safe(c34, 4):
        add_image_slide(prs, "Reproducibility -- Per-Level Differences",
                        c34[4],
                        "\u0394B1, \u0394b2, \u0394b3 (Test 04 \u2212 Test 03)")
    if _safe(c34, 6):
        add_image_slide(prs, "Reproducibility -- Hysteresis Width & Noise",
                        c34[6])

    # ==================================================================
    # Chapter 7: Eddy Current Comparison -- Single vs Double Tau
    # ==================================================================
    add_chapter_slide(prs, 7,
        "Eddy Current Analysis\nSingle vs Double Exponential",
        "Compare single-exp B1(t) = Binf + A exp(-t/\u03c4)\n"
        "vs double-exp with two time constants \u03c4_1, \u03c4_2")

    # eddy_01/02 (6 images each): 0: settling, 1: fits, 2: tau vs I,
    # 3: bias (3-panel), 4: sensitivity, 5: double-exp
    # eddy_03 (7 images): same + 5: precycle, 6: double-exp
    e01 = images.get("eddy_01", [])
    e02 = images.get("eddy_02", [])
    if _safe(e01, 2):
        add_image_slide(prs, "Test 01 (With Shims) -- \u03c4 vs Current",
                        e01[2])
    if _safe(e01, 5):
        add_image_slide(prs, "Test 01 -- Single vs Double Exp Fit",
                        e01[5],
                        "R\u00b2 comparison and time-constant analysis")
    if _safe(e02, 5):
        add_image_slide(prs, "Test 02 (No Shims) -- Single vs Double Exp Fit",
                        e02[5])
    if _safe(e03, 6):
        add_image_slide(prs, "Test 03 (2 Hz) -- Single vs Double Exp Fit",
                        e03[6])

    # ==================================================================
    # Chapter 8: Observations
    # ==================================================================
    add_chapter_slide(prs, 8,
        "Key Observations", "")

    add_text_slide(prs, "Shims Effect -- Key Findings",
        "Comparing Test 01 (with shims) vs Test 02 (without shims):\n\n"
        "\u2022 b2 (quadrupole): primary target of shims -- "
        "quantifiable difference in allowed harmonic\n"
        "\u2022 B1 (main field): small change, shims affect "
        "homogeneity not total flux\n"
        "\u2022 b3 (sextupole): indirect changes through "
        "saturation redistribution\n"
        "\u2022 Transfer function: minimal impact on TF\n\n"
        "cel/fed disabled for both tests (UNSAFE diagnostic)")

    add_text_slide(prs, "2 Hz Streaming -- Key Findings",
        "Streaming at 2 Hz (120 rpm) provides:\n\n"
        "\u2022 Full supercycle visibility -- every turn captured\n"
        "\u2022 Turn classification: automated plateau/ramp/precycle separation\n"
        "\u2022 Eddy-current transients resolved with 0.5 s time resolution\n"
        "\u2022 Harmonics vs time: real-time field quality evolution\n\n"
        "All-turns analysis (including ramp turns) reveals\n"
        "current-harmonic correlations invisible in plateau-only averages.")

    # ==================================================================
    # Chapter 9: Summary & Conclusions
    # ==================================================================
    add_chapter_slide(prs, 9,
        "Summary\n& Conclusions", "")

    add_text_slide(prs, "MC62 Measurement Campaign Summary",
        "4 staircase tests: 01 (with shims, 1 Hz), 02 (no shims, 1 Hz), "
        "03 (no shims, 2 Hz PM), 04 (no shims, 2 Hz AM)\n\n"
        "Key findings:\n"
        "\u2022 Shims effect: quantified via 01-vs-02 comparison\n"
        "\u2022 Reproducibility: excellent (03 vs 04, "
        "\u0394B1 < 62 \u00b5T, \u0394b3 < 0.05 units)\n"
        "\u2022 Eddy currents: \u03c4 depends on |I| "
        "(permeability effect, 2--40 s)\n"
        "\u2022 Pipeline validation: machine-precision parity with FFMM C++\n"
        "\u2022 cel/fed correctly auto-disabled for all 4 tests")

    # -- Last slide --
    add_last_slide(prs)

    # Save
    prs.save(str(OUTPUT_PPTX))
    print(f"\nPresentation saved to: {OUTPUT_PPTX}")
    print(f"Total slides: {len(prs.slides)}")


if __name__ == "__main__":
    if "--count" in sys.argv:
        print("Image counts per notebook:")
        count_images_all()
    else:
        build()
