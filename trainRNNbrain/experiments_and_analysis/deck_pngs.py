#!/usr/bin/env python3
"""
Rasterise every figure `docs/presentation.md` links, so the deck can be read on GitHub.

WHY THE DECK LINKS PNG AND NOT SVG. The talk's own assets are vector -- `paperstyle.save` writes a
PDF and an SVG for every panel and that is what you present from. But GitHub's blob view will not
render a markdown file whose images are 35 resolvable SVGs: it returned HTTP 503 on
docs/presentation.md with them committed, while the SAME markdown with the SAME <img> tags rendered
fine on a branch where the images were absent, and GitHub's own /markdown API renders the file
without complaint. The difference is GitHub resolving and sanitising each SVG during page render.
Two of these panels are dense line plots -- every seed's full loss trace, 1,500 to 2,000 probes --
and weigh 1.14 MB and 0.90 MB as SVG with roughly a hundred thousand path nodes each.

A PNG is an opaque raster: nothing to parse, nothing to sanitise, and about a tenth the bytes for
exactly these plots (1.14 MB of SVG becomes 119 KB). So the markdown links PNG, the PDFs and SVGs
stay on disk for the talk, and `paperstyle.save` is untouched -- the vector-only rule it documents
is about the manuscript's panels and still holds.

The source is each panel's PDF rather than its SVG, because the PDF is the asset the project treats
as authoritative and `pdftoppm` renders it without a browser or an SVG library.

Usage:  python trainRNNbrain/experiments_and_analysis/deck_pngs.py [--dpi 200] [--check]
        (run from the repository root)
        --check reports what is missing or stale and writes nothing.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys

DECK = "docs/presentation.md"
FIG_DIR = "img/internal_figures"
DPI = 200          # 790 to 1180 px wide for this project's panel widths, against a 760 px display


def linked_stems(deck=DECK):
    """Every figure stem the deck links, in order of appearance, without duplicates.

    Args:
        deck: path to the talk's Markdown file.
    Returns:
        list of stems, e.g. ['slide_m1_model_organism', ...].
    """
    text = open(deck).read()
    out = []
    for stem in re.findall(rf'<img src="\.\./{FIG_DIR}/([^"]+)\.(?:png|svg)"', text):
        if stem not in out:
            out.append(stem)
    return out


def rasterise(stem, dpi=DPI):
    """Write `<stem>.png` beside `<stem>.pdf`. Returns the png path, or None if the pdf is absent."""
    pdf = os.path.join(FIG_DIR, f"{stem}.pdf")
    png = os.path.join(FIG_DIR, f"{stem}.png")
    if not os.path.exists(pdf):
        return None
    subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile", pdf, png[:-4]],
                   check=True, capture_output=True)
    return png


def main(dpi=DPI, check_only=False):
    """Rasterise or check every linked figure. Returns the number of problems found."""
    if not check_only and shutil.which("pdftoppm") is None:
        print("pdftoppm not found: install poppler (brew install poppler)")
        return 1
    stems = linked_stems()
    missing, stale, total = [], [], 0
    for stem in stems:
        pdf = os.path.join(FIG_DIR, f"{stem}.pdf")
        png = os.path.join(FIG_DIR, f"{stem}.png")
        if not os.path.exists(pdf):
            missing.append(stem)
            continue
        if check_only:
            if not os.path.exists(png):
                missing.append(stem)
            elif os.path.getmtime(png) < os.path.getmtime(pdf):
                stale.append(stem)
            else:
                total += os.path.getsize(png)
            continue
        rasterise(stem, dpi)
        total += os.path.getsize(png)
    verb = "checked" if check_only else f"wrote at {dpi} dpi"
    print(f"{verb}: {len(stems) - len(missing)} of {len(stems)} linked figures, "
          f"{total / 1e6:.1f} MB total")
    for s in missing:
        print(f"  MISSING pdf or png: {s}")
    for s in stale:
        print(f"  STALE png, pdf is newer: {s}")
    return len(missing) + len(stale)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dpi", type=int, default=DPI, help="raster resolution")
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    a = ap.parse_args()
    sys.exit(1 if main(a.dpi, a.check) else 0)
