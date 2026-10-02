#!/usr/bin/env python3
"""
Build a single self-contained HTML copy of the talk, readable without GitHub or a Markdown viewer.

WHY THIS EXISTS. On 2026-10-02 github.com served its "Unicorn / bad day" page -- HTTP 503 -- for
blob views of this repository: `docs/presentation.md` on the talk branch, `README.md` on `main`, and
files nobody had touched, all of them, while the tree listing, the commit pages,
raw.githubusercontent.com and GitHub's own /markdown API kept working. Measured over a minute, 2 of
14 requests for the deck's page returned 200. Reloading eventually gets through; reading a 35-slide
deck that way does not work.

So the deck gets a copy that depends on nothing: one HTML file with every figure inlined as a
base64 data URI. It opens from disk, survives being moved or emailed, and needs no server, no
Markdown extension and no network.

THE MARKDOWN IS RENDERED BY GITHUB'S OWN /markdown API, through `gh`, so the result is the page you
would have read on github.com rather than a near-enough imitation -- the appendix's tables and the
centred `<img>` blocks come out exactly as GitHub lays them out. That endpoint stayed up throughout
the blob outage. If `gh` is unavailable the script says so and writes nothing, rather than silently
producing something that looks like GitHub and is not.

The output is NOT committed: it is a build product of docs/presentation.md and the PNGs, both of
which are in the repository, and it would otherwise be a third copy of the deck to keep in step.

Usage:  python trainRNNbrain/experiments_and_analysis/deck_html.py [-o docs/presentation.html]
        (run from the repository root)
"""

import argparse
import base64
import mimetypes
import os
import re
import shutil
import subprocess
import sys

DECK = "docs/presentation.md"
OUT = "docs/presentation.html"
REPO = "engellab/trainRNNbrain"

# Enough CSS to read 35 slides comfortably: GitHub's own measure and type scale, nothing more.
CSS = """
  :root { color-scheme: light dark; }
  body { max-width: 980px; margin: 0 auto; padding: 32px 16px 96px;
         font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
         color: #1f2328; background: #ffffff; }
  h1 { font-size: 1.9em; border-bottom: 1px solid #d1d9e0; padding-bottom: .3em; margin-top: 2.2em; }
  h3 { font-size: 1.15em; margin-top: 2em; }
  img { max-width: 100%; height: auto; }
  table { border-collapse: collapse; display: block; overflow: auto; }
  th, td { border: 1px solid #d1d9e0; padding: 6px 13px; }
  tr:nth-child(2n) { background: #f6f8fa; }
  code { background: #eff1f3; padding: .2em .4em; border-radius: 6px; font-size: 85%; }
  pre { background: #f6f8fa; padding: 16px; border-radius: 6px; overflow: auto; }
  pre code { background: none; padding: 0; }
  blockquote { border-left: .25em solid #d1d9e0; padding: 0 1em; color: #59636e; }
  @media (prefers-color-scheme: dark) {
    body { color: #f0f6fc; background: #0d1117; }
    h1 { border-bottom-color: #3d444d; }
    th, td { border-color: #3d444d; } tr:nth-child(2n) { background: #151b23; }
    code { background: #262c36; } pre { background: #151b23; }
    blockquote { border-left-color: #3d444d; color: #9198a1; }
  }
"""


def render_markdown(deck=DECK):
    """GitHub's own HTML for the deck's Markdown.

    Args:
        deck: path to the Markdown file.
    Returns:
        str of HTML, or None if `gh` is unavailable or the call fails.
    """
    if shutil.which("gh") is None:
        return None
    r = subprocess.run(["gh", "api", "--method", "POST", "/markdown", "-f", "mode=gfm",
                        "-f", f"context={REPO}", "-F", f"text=@{deck}"],
                       capture_output=True, text=True)
    return r.stdout if r.returncode == 0 and r.stdout.strip() else None


def inline_images(html, base="docs"):
    """Replace every relative <img src> with a base64 data URI of the file it names.

    Args:
        html: rendered HTML whose image sources are relative to `base`;
        base: the directory the Markdown file lives in, which its relative paths resolve against.
    Returns:
        (html with sources inlined, number inlined, list of paths that were missing).
    """
    missing, done = [], 0

    def sub(m):
        nonlocal done
        src = m.group(2)
        if src.startswith(("http://", "https://", "data:")):
            return m.group(0)
        path = os.path.normpath(os.path.join(base, src))
        if not os.path.exists(path):
            missing.append(src)
            return m.group(0)
        mime = mimetypes.guess_type(path)[0] or "image/png"
        b64 = base64.b64encode(open(path, "rb").read()).decode("ascii")
        done += 1
        return f'{m.group(1)}data:{mime};base64,{b64}"'

    return re.sub(r'(<img\b[^>]*?\bsrc=")([^"]+)"', sub, html), done, missing


def main(out=OUT):
    """Write the self-contained HTML. Returns 0 on success, 1 if it could not be built."""
    body = render_markdown()
    if body is None:
        print("could not render: `gh` is missing or the /markdown call failed. "
              "Run `gh auth status`.")
        return 1
    body, n, missing = inline_images(body)
    title = "Dormant units in trained ReLU RNNs — talk track"
    page = (f"<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            f"<title>{title}</title>\n<style>{CSS}</style>\n</head>\n<body>\n{body}\n"
            f"</body>\n</html>\n")
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {out}: {os.path.getsize(out) / 1e6:.1f} MB, {n} figures inlined")
    for s in missing:
        print(f"  MISSING, left as a link: {s}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default=OUT, help="output HTML path")
    sys.exit(main(ap.parse_args().out))
