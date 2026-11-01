"""The manuscript style sheet holds on paper: one typeface, a bold face, the specified sizes and width.

Renders `paperstyle.demo_figure()` (two panels with axis labels, maths, ticks, a legend and panel
letters), saves it with `paperstyle.save`, and reads the PDF back with tools that share no code with
matplotlib: `pdffonts` (poppler) for the embedded fonts and PyMuPDF for the size of every glyph run.
It fails if
  - the PDF embeds more than one font family (e.g. DejaVu Sans from mathtext, issue #40),
  - there is no bold face (issue #15),
  - a panel letter is not bold or not PANEL_LETTER_PT, an axis label not LABEL_PT, a tick label
    not TICK_PT, any glyph smaller than MIN_SCRIPT_PT (a subscript of tick-size text),
  - the page is not W_FULL_MM wide (the figure would not print at 100%).

Given figure PDFs as arguments, it audits those instead, at the size each prints when included at
\\textwidth: the house family only (Helvetica on the Mac), a bold face, panel letters 8-10 pt, no glyph under MIN_SCRIPT_PT (the
acceptance tests of dead-relu-paper #15 and #40). Glyph runs between MIN_SCRIPT_PT and MIN_PT are
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
    summary = dict(fonts=fonts, scale=round(scale, 3),
                   median_pt=round(statistics.median(sizes), 2) if sizes else None,
                   letters_pt=sorted(set(letters)),
                   runs_under_min_pt=sum(s < ps.MIN_PT - 0.05 for s in sizes))
    return fail, summary


def check_demo():
    """Render the demo figure, save it with paperstyle.save, and hold it to the exact style sheet.

    Returns:
        list of str, the failures (empty if the style sheet holds).
    """
    fig = ps.demo_figure()
    ax_a = fig.axes[0]
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
