"""The manuscript style sheet holds on paper: one typeface, a bold face, the specified sizes and width.

Renders `paperstyle.demo_figure()` (two panels with axis labels, maths, ticks, a legend and panel
letters), saves it with `paperstyle.save`, and reads the PDF back with tools that share no code with
matplotlib: `pdffonts` (poppler) for the embedded fonts and PyMuPDF for the size of every glyph run.
It fails if
  - the PDF embeds more than one font family (e.g. DejaVu Sans from mathtext, issue #40),
  - there is no bold face (issue #15),
  - a panel letter is not bold or not PANEL_LETTER_PT, an axis label not LABEL_PT, a tick label
    not TICK_PT, any glyph smaller than MIN_SCRIPT_PT (a subscript of tick-size text),
  - the page is not W_FULL_MM wide (the figure would not print at 100%),
  - a y-axis label sits closer than MIN_LABEL_GAP_MM (1 mm) to its tick labels, measured between
    PyMuPDF span boxes; the demo asks one label for labelpad=1, which `save` must raise
    (dead-relu-paper #63).

Given figure PDFs as arguments, it audits those instead, at the size each prints when included at
\\textwidth: the house family only (Helvetica on the Mac), a bold face, panel letters 8-10 pt, no glyph under MIN_SCRIPT_PT (the
acceptance tests of dead-relu-paper #15 and #40), every y-label at least 1 mm from its tick labels (#63). Glyph runs between MIN_SCRIPT_PT and MIN_PT are
counted, not failed: a PDF does not say which runs are sub- or superscripts.

Run: python tests/test_paperstyle.py [figure.pdf ...]
"""
import os
import re
import statistics
import subprocess
import sys
import tempfile

import pymupdf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "trainRNNbrain", "experiments_and_analysis"))

import paperstyle as ps  # noqa: E402  (needs the path above)

PT_PER_MM = 72 / 25.4
MIN_LABEL_GAP_MM = 1.0      # y-label to tick-label gap on the page (dead-relu-paper #63)


def pdf_font_names(path):
    """Base names of the fonts embedded in a PDF, as `pdffonts` lists them, subset prefix removed.

    Args:
        path: str, a PDF file.
    Returns:
        sorted list of str, e.g. ['Helvetica', 'Helvetica-Bold'].
    """
    out = subprocess.run(["pdffonts", path], capture_output=True, text=True, check=True).stdout
    names = [line.split()[0] for line in out.splitlines()[2:] if line.strip()]
    return sorted({n.split("+", 1)[-1] for n in names})


def family_of(font_name):
    """Family part of a PostScript font name: 'Helvetica-BoldOblique' -> 'Helvetica', 'Arial-BoldMT' -> 'Arial'."""
    fam = font_name.split("-", 1)[0]
    return re.sub(r"(MT|PS)$", "", fam)


def spans(path):
    """Every text run on the first page of a PDF, with its size and font.

    Args:
        path: str, a PDF file.
    Returns:
        (list of (text, size_pt, font_name), page_width_mm).
    """
    doc = pymupdf.open(path)
    page = doc[0]
    out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for s in line["spans"]:
                if s["text"].strip():
                    out.append((s["text"].strip(), s["size"], s["font"]))
    return out, page.rect.width / PT_PER_MM


def y_label_gaps(path, scale=1.0, search_mm=8.0):
    """Gap between every rotated (vertical) text line and the nearest horizontal text to its right.

    A y-axis label is a vertical line of text; its tick labels are the horizontal lines to its
    right that overlap it in height. Vertical lines with no horizontal text within `search_mm`
    (a label on an axis without tick labels, a rotated annotation) are skipped.

    Args:
        path: str, a PDF file; scale: printed size / PDF size; search_mm: how far right to look.
    Returns:
        list of (text, gap_mm at print size), one per vertical line that has a neighbour.
    """
    page = pymupdf.open(path)[0]
    vertical, horizontal = [], []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(s["text"] for s in line["spans"]).strip()
            if not text:
                continue
            (vertical if abs(line["dir"][0]) < 0.1 else horizontal).append((text, line["bbox"]))
    out = []
    for text, (x0, y0, x1, y1) in vertical:
        gaps = [hx0 - x1 for _, (hx0, hy0, hx1, hy1) in horizontal
                if hy1 > y0 and hy0 < y1 and -0.5 <= hx0 - x1 < search_mm * PT_PER_MM]
        if gaps:
            out.append((text, min(gaps) * scale / PT_PER_MM))
    return out


def audit(path, printed_width_mm=ps.W_FULL_MM):
    """Check one figure PDF against the style sheet at the size it prints when set at \\textwidth.

    Args:
        path: str, a figure PDF; printed_width_mm: the width LaTeX prints it at.
    Returns:
        (failures: list of str, summary: dict with fonts, scale, median and letter sizes in printed pt).
    """
    fonts = pdf_font_names(path)
    runs, width_mm = spans(path)
    scale = printed_width_mm / width_mm
    letters = [round(size * scale, 2) for text, size, font in runs
               if len(text) == 1 and text.islower() and "Bold" in font]
    sizes = [size * scale for _, size, _ in runs]
    families = sorted({family_of(f) for f in fonts})
    fail = []
    house = ps.register_fonts().replace(" ", "")
    if families != [house]:
        fail.append(f"font families {families}, want {house} only")
    if not any("Bold" in f for f in fonts):
        fail.append(f"no bold face among {fonts}")
    if not letters:
        fail.append("no bold panel letter found")
    elif min(letters) < 8 or max(letters) > 10:
        fail.append(f"panel letters {min(letters)}-{max(letters)} pt printed, outside 8-10 pt")
    if sizes and min(sizes) < ps.MIN_SCRIPT_PT - 0.05:
        fail.append(f"text at {min(sizes):.1f} pt printed, under {ps.MIN_SCRIPT_PT:.1f} pt")
    gaps = y_label_gaps(path, scale)
    for text, gap in gaps:
        if gap < MIN_LABEL_GAP_MM:
            fail.append(f"y-label '{text}' {gap:.2f} mm from its tick labels, under {MIN_LABEL_GAP_MM} mm")
    summary = dict(fonts=fonts, scale=round(scale, 3),
                   median_pt=round(statistics.median(sizes), 2) if sizes else None,
                   letters_pt=sorted(set(letters)),
                   runs_under_min_pt=sum(s < ps.MIN_PT - 0.05 for s in sizes),
                   min_ylabel_gap_mm=round(min(g for _, g in gaps), 2) if gaps else None)
    return fail, summary


def check_demo():
    """Render the demo figure, save it with paperstyle.save, and hold it to the exact style sheet.

    Returns:
        list of str, the failures (empty if the style sheet holds).
    """
    fig = ps.demo_figure()
    ax_a = fig.axes[0]
    fig.axes[1].set_ylabel(ps.LABELS["r2"], labelpad=1)    # the pad Fig. 4 asked for; save() must raise it
    fail = []
    # what matplotlib was asked to draw
    if ax_a.xaxis.label.get_size() != ps.LABEL_PT:
        fail.append(f"axis label asked at {ax_a.xaxis.label.get_size()} pt")
    with tempfile.TemporaryDirectory() as tmp:
        path = ps.save(fig, "paperstyle_demo", out_dir=tmp)
        print("pdffonts", os.path.basename(path))
        print(subprocess.run(["pdffonts", path], capture_output=True, text=True).stdout.rstrip())
        audit_fail, summary = audit(path)
        fail += audit_fail
        runs, width_mm = spans(path)
    print("summary", summary)
    if summary["min_ylabel_gap_mm"] is None:
        fail.append("no y-label to tick-label gap measured on the demo")
    # what landed in the PDF, read back independently
    if abs(width_mm - ps.W_FULL_MM) > 0.1:
        fail.append(f"page is {width_mm:.2f} mm wide, not {ps.W_FULL_MM} mm")
    families = {family_of(f) for f in summary["fonts"]}
    if families != {ps.FONT_FAMILY.replace(" ", "")}:
        fail.append(f"fonts {summary['fonts']} are not all {ps.FONT_FAMILY}")
    by_text = {}
    for text, size, font in runs:
        by_text.setdefault(text, []).append((round(size, 2), font))
    expect = {"active units": ps.LABEL_PT, "iteration": ps.LABEL_PT, "1,000": ps.TICK_PT,
              "200,000": ps.TICK_PT, "3BFF": ps.TICK_PT}
    for text, want in expect.items():
        got = by_text.get(text)
        if not got or any(abs(s - want) > 0.05 for s, _ in got):
            fail.append(f"'{text}' at {got}, want {want} pt")
    for letter in "ab":
        got = by_text.get(letter)
        if not got or any(abs(s - ps.PANEL_LETTER_PT) > 0.05 or "Bold" not in f for s, f in got):
            fail.append(f"panel letter '{letter}' is {got}, want bold {ps.PANEL_LETTER_PT} pt")
    return fail


def main():
    """Audit the PDFs given on the command line, or with none, check the demo figure. Exit 1 on failure."""
    paths = sys.argv[1:]
    if paths:
        bad = 0
        for path in paths:
            if not os.path.exists(path):
                print(f"FAIL {path}: no such file")
                bad += 1
                continue
            fail, summary = audit(path)
            print(f"{'FAIL' if fail else 'ok  '} {os.path.basename(path)} {summary}")
            for f in fail:
                print("     ", f)
            bad += bool(fail)
        sys.exit(1 if bad else 0)
    fail = check_demo()
    for f in fail:
        print("FAIL:", f)
    if fail:
        sys.exit(1)
    print(f"ok: one family ({ps.FONT_FAMILY}), bold panel letters at {ps.PANEL_LETTER_PT} pt, "
          f"labels {ps.LABEL_PT} pt, ticks {ps.TICK_PT} pt, page {ps.W_FULL_MM} mm")


if __name__ == "__main__":
    main()
