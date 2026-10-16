"""Every analysis script that takes command-line arguments starts: `python <script> --help` exits 0.

Importing a module is not the same as running it: run as __main__, a script re-imports itself
through any import cycle, and a `from X import name` on the half-built module fails (this broke
fig_paper_F1.py and fig_paper_F2_fails.py on 2026-10-15). Scripts without argparse are skipped,
because `--help` would run them for real and overwrite figures.
Run: python tests/test_scripts_start.py
"""
import glob
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EA = os.path.join(REPO, "trainRNNbrain", "experiments_and_analysis")


def scripts_with_argparse():
    """Paths of the analysis scripts that parse command-line arguments.

    Returns:
        sorted list of str, absolute paths of tracked-or-not .py files in experiments_and_analysis
        whose source mentions argparse.
    """
    out = []
    for path in sorted(glob.glob(os.path.join(EA, "*.py"))):
        with open(path) as fh:
            if "argparse" in fh.read():
                out.append(path)
    return out


def main():
    """Run each script with --help from the repo root; exit 1 and print the error of any that fail."""
    env = dict(os.environ, PYTHONPATH=REPO, CUDA_VISIBLE_DEVICES="")
    bad = []
    paths = scripts_with_argparse()
    for path in paths:
        r = subprocess.run([sys.executable, path, "--help"], cwd=REPO, env=env,
                           capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            bad.append(path)
            tail = (r.stderr.strip().splitlines() or ["(no stderr)"])[-1]
            print(f"{os.path.relpath(path, REPO)}: {tail}")
    if bad:
        sys.exit(1)
    print(f"ok: {len(paths)} scripts start")


if __name__ == "__main__":
    main()
