#!/usr/bin/env python3
"""
Check `docs/presentation.md` against the four rules the talk is built on. Exits non-zero on failure.

The deck was rewritten to a standard that is easy to state and easy to lose: the picture carries the
claim, the prose names the task and stops, nothing points at a slide the audience cannot see, and no
code identifier reaches the screen. Each of those is one grep, so they are one script, and the
script runs in a second -- which is the difference between a standard and an intention.

  1. every figure the deck links exists on disk (the links are PNG, written by deck_pngs.py)
  2. the TALK BODY (everything before the appendix) contains no cross-reference of the form
     "slide 12" -- an audience cannot flip back, so a pointer is a dead end
  3. the talk body contains no code identifier, folder name or config key
  4. no slide carries more than MAX_WORDS words of prose under its figure
  5. every slide figure on disk is either in the talk or named in the appendix with a reason -- a
     panel that is simply absent is a panel nobody decided to cut
  6. every slide that shows DATA names the training budget it was read at, in its own figure, its
     own caption, or the intro of the section it sits in. The deck reads 40,000 iterations in its
     results sections and 150,000 or 200,000 in its opening, so a panel that says only "trained"
     lets a reader carry the wrong number from one section into the next -- which happened

Rule 4 has a deliberate ceiling rather than a target. A slide at the ceiling is usually one whose
figure is not finished: if the picture needed fifty words of help, the picture is the thing to fix.

Usage:  python trainRNNbrain/experiments_and_analysis/check_presentation.py [path]
        (run from the repository root)
"""

import glob
import os
import re
import sys

DECK = "docs/presentation.md"
APPENDIX = "# APPENDIX"
MAX_WORDS = 55

# Identifiers that mean something to this repository and nothing to an audience. The deck may use
# them inside an <img> path -- a filename is not something anyone reads off the screen -- so the
# prose is tested after the image tags are stripped.
# A slide whose figure is a pure schematic has no budget to name. These are listed rather than
# detected, because "has no data" is not something a .svg can be asked.
SCHEMATIC = ("slide_m1_model_organism", "fig_supp_tasks", "slide_rules", "slide_mech_dropout",
             "slide_mech_duplicate", "slide_mech_rescale", "slide_mech_synnoise",
             "slide_mech_penalty", "slide_wh_zero_gradient", "slide_33_bottom_line",
             "slide_06_scaling")
BUDGET = r"iteration|iters\b|\b\d{2,3},\d{3}\b|\b\d{2,3}k\b|own loss floor"

FORBIDDEN = ("copy_noise", "cap_fr", "rescale_target_frac", "rescale_alpha", "EqType", "DATA_DIR",
             "reinit_mode", "input_row_norm", "LmbdMet", "LmbdFR", "LmbdRWS", "sigma_w",
             "q_0.9", "tPR", "n_units=", "_sweep", "paper_grid", "frm_args")


def prose_of(block):
    """The words a listener reads under one slide: the block with its image tags removed.

    ⚠️ STOPS AT THE NEXT HEADING OR RULE. A slide that is the last in its section is followed by the
    section break and the next section's title, and counting those put the last slide of every
    section 30 to 40 words over its real length -- which this check then reported as a defect in the
    slide. The first version of this function did exactly that.

    Args:
        block: the Markdown of one slide, heading line excluded.
    Returns:
        str of the remaining prose: image tags, blank lines and anything from the next heading or
        horizontal rule onwards dropped.
    """
    out = []
    for line in block.split("\n"):
        s = line.strip()
        if s.startswith("#") or s == "---":
            break
        if s and not s.startswith("<p align"):
            out.append(s)
    return " ".join(out)


def check(path=DECK):
    """Run every rule over the deck.

    Args:
        path: the deck's Markdown file, relative to the repository root.
    Returns:
        list of failure strings, empty when the deck passes.
    """
    text = open(path).read()
    body = text.split(APPENDIX)[0]
    fails = []

    for m in re.finditer(r'src="\.\./(img/[^"]+)"', text):
        if not os.path.exists(m.group(1)):
            fails.append(f"figure does not exist: {m.group(1)}")

    for m in re.finditer(r"\bslides? \d+\w*\b", body, re.I):
        fails.append(f"cross-reference in the talk body: {m.group(0)!r}")

    # RULE 5. A figure that is on disk and in neither place was dropped by accident rather than by
    # decision, which is how the previous deck came to carry panels nobody could account for.
    appendix = text.split(APPENDIX)[1] if APPENDIX in text else ""
    shown = set(re.findall(r'src="\.\./img/internal_figures/([^"]+)\.(?:png|svg)"', body))
    named = set(re.findall(r"`(slide_[a-z0-9_]+)`", appendix))
    for path in sorted(glob.glob("img/internal_figures/slide_*.svg")):
        stem = os.path.basename(path)[:-4]
        if stem not in shown and stem not in named:
            fails.append(f"figure neither shown nor accounted for in the appendix: {stem}")

    # RULE 6 needs the section each slide sits in and whether that section's INTRO names a budget,
    # so the body is walked in document order. A section's intro is the text between its "# " line
    # and its first "### " slide.
    section_budget, slide_section, section = {}, {}, ""
    for m in re.finditer(r"^(#{1,3}) (.+)$", body, flags=re.M):
        level, title = len(m.group(1)), m.group(2).strip()
        if level == 1:
            section = title
            intro = body[m.end():]
            intro = intro[:intro.find("\n### ")] if "\n### " in intro else intro
            section_budget[section] = bool(re.search(BUDGET, intro, re.I))
        elif level == 3:
            slide_section[title] = section

    blocks = re.split(r"^### ", body, flags=re.M)[1:]
    for b in blocks:
        head, *rest = b.split("\n")
        prose = prose_of("\n".join(rest))
        n = len(prose.split())
        if n > MAX_WORDS:
            fails.append(f"{n} words of prose (limit {MAX_WORDS}) under: {head[:62]}")
        stems = re.findall(r"internal_figures/([^\"]+)\.(?:png|svg)", b)
        if stems and stems[0] not in SCHEMATIC:
            svg = f"img/internal_figures/{stems[0]}.svg"
            inside = (open(svg, encoding="utf-8", errors="ignore").read()
                      if os.path.exists(svg) else "")
            named = (re.search(BUDGET, inside, re.I) or re.search(BUDGET, prose + " " + head, re.I)
                     or section_budget.get(slide_section.get(head.strip(), ""), False))
            if not named:
                fails.append(f"no training budget named for: {head[:62]}")
        for bad in FORBIDDEN:
            if bad in prose:
                fails.append(f"code identifier {bad!r} under: {head[:62]}")

    print(f"{len(blocks)} slides, {len(body.split())} words in the talk body")
    return fails


if __name__ == "__main__":
    problems = check(sys.argv[1] if len(sys.argv) > 1 else DECK)
    for p in problems:
        print("  FAIL", p)
    print("OK" if not problems else f"{len(problems)} problems")
    sys.exit(1 if problems else 0)
