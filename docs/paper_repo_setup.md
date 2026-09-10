# Handoff: create the paper repository for the silent-units manuscript

**Audience:** an agent starting fresh, with no memory of the work that produced these results.
**Goal:** a standalone, *private* GitHub repository containing a Nature Communications LaTeX
manuscript, synced to Overleaf, that draws its figures and numbers from this codebase.

Written 2026-09-10 against `trainRNNbrain` commit `2410914` (branch `dimensionality-analysis`).

---

## 0. Read this first — the two things most likely to go wrong

⚠️ **DO NOT put the paper on a branch of `trainRNNbrain`.** That repo's `origin` is
`github.com/engellab/trainRNNbrain`, a *shared lab* repo, and it sits beside ~139 GB of untracked
data. A separate repo was chosen deliberately: Overleaf's GitHub sync tracks a repo's default
branch, so making the paper its own repo removes any question about branch selection, keeps the
draft private, and means nobody can wipe a huge working tree with a stray `git checkout`.

⚠️ **VERIFY EVERY JOURNAL REQUIREMENT AGAINST THE PUBLISHER, NOT THIS DOCUMENT.** The formatting
notes in §3 are written from general knowledge and may be out of date or wrong in detail. Check
`https://www.nature.com/ncomms/submit` and the current Springer Nature LaTeX template before
relying on any word count, figure limit, or class-file name here.

---

## 1. Create the repository

Private, under Pavel's own account (**not** `engellab`). `gh` 2.89 is installed and authenticated.

```bash
cd ~/Documents/GitHub
mkdir dead-relu-paper && cd dead-relu-paper
git init -b main
gh repo create dead-relu-paper --private --source=. --remote=origin
```

Confirm it landed under the personal account and is private:

```bash
gh repo view --json nameWithOwner,visibility
```

Author identity is already configured globally (`Pavel`, `betadecay1993@gmail.com`).

---

## 2. Layout to create

```
dead-relu-paper/
  main.tex                 documentclass + \input of each section
  sections/                01_intro.tex, 02_results.tex, … (see §5)
  figures/                 copied image files — never symlinks
  figures/SOURCES.md       figure -> generating script -> source commit -> date
  caches/                  the four .npz analysis caches (~60 KB total)
  refs.bib
  sync_figures.sh
  Makefile                 `make pdf`
  README.md
  .gitignore
```

`.gitignore`:

```
*.aux
*.log
*.out
*.synctex.gz
*.fls
*.fdb_latexmk
*.bbl
*.blg
*.toc
```

⚠️ **Do not gitignore the built PDF if Overleaf sync is in use** — Overleaf will produce its own,
and an ignored PDF is fine, but never commit `.aux`/`.fls`, which cause sync conflicts.

---

## 3. LaTeX setup for Nature Communications

Springer Nature provides a LaTeX class, `sn-jnl.cls`, used across its journals; Overleaf's template
gallery carries it as the "Springer Nature LaTeX Template". For Nature Portfolio journals the
usual invocation is the `sn-nature` reference style:

```latex
\documentclass[sn-nature]{sn-jnl}
```

⚠️ **Verify the class name, options and current template location before committing to them.** If
the template is unavailable or fights you, Nature Communications accepts a plain PDF for *initial*
submission — a standard `\documentclass[11pt]{article}` with `natbib` is perfectly acceptable to
start, and converting to the house class later costs an hour. Do not lose a day to a class file.

What to check on the journal's author-instructions page and record here once confirmed:

- abstract word limit (believed ~150 words, unreferenced)
- main-text word limit (believed ~5,000 excluding Methods, references, captions)
- Methods word limit (believed ~3,000)
- display-item limit (believed 10 for the main text)
- reference limit, and whether Methods references count separately
- whether a Reporting Summary or Code Availability statement is required (Code Availability
  almost certainly is — see §7)

---

## 4. Where the science lives

Everything below is in `~/Documents/GitHub/trainRNNbrain/`.

| what | path |
|---|---|
| **Manuscript draft in prose** — the source to convert | `docs/paper.md` |
| Full experimental record, including every retraction | `docs/project_trajectory.md` |
| Analysis scripts | `trainRNNbrain/experiments_and_analysis/` |
| Generated figures | `img/internal_figures/` |
| Curated dataset (hard links, 63 GB, 5,237 files) | `dead_ReLU_data/` + its `README.md` |
| Analysis caches (60 KB — copy these into the paper repo) | `data/*.npz` |

**`docs/paper.md` is the primary source.** It is a full draft in markdown with sections already
numbered, results stated with numbers, and open items marked `⬜`. Converting it is the main job,
not writing from scratch.

**`docs/project_trajectory.md` is the lab notebook.** Consult it whenever a number needs its
provenance, and *especially* before reviving any claim — it contains a retraction record listing
several results that did not survive scrutiny. Do not resurrect them.

---

## 5. Suggested section files

Mirror `docs/paper.md`'s structure:

| file | source in `paper.md` |
|---|---|
| `01_intro.tex` | §1 The problem (§1.1–1.3) |
| `02_regularizers_fail.tex` | §2 Standard activity regularizers do not fix it |
| `03_what_works.tex` | §3 What we tried, and what worked |
| `04_tuned_units.tex` | §3.1 The rescued units are task-tuned |
| `05_stabilization.tex` | §3.2 What `rws` adds: it stabilizes the rescue |
| `06_prevent_or_resurrect.tex` | §4 |
| `07_discussion.tex` | §6 open questions, §7 framing |
| `08_methods.tex` | §8 Methods points that must be stated |
| `S1_constrained.tex` | §S1 supplementary |

⚠️ **Consider leading with §3.1/§3.2 rather than the silence result.** Noted at the end of the
project record: what penalised networks *buy you* (≈8× more task-tuned units) is more novel and
more useful than what unpenalised networks *lack*, and it makes the framing constructive from the
first paragraph. This is a judgement call for Pavel, not a settled decision.

⚠️ **The paper is currently one task.** The n-bit flip-flop carries §3.1, §3.2 and §6.4; CDDM
carries §1–2 and §4. A referee will ask whether the scaling law holds on both. CDDM data exists but
its analysis was withdrawn mid-project over a threshold error and a masking error — see §6.

---

## 6. Known-open items — do not present these as settled

Pulled from `docs/paper.md` (`⬜` markers) and the project record:

1. **CDDM re-analysis outstanding.** Earlier CDDM numbers were withdrawn: the flip-flop-calibrated
   silence threshold (`4e-2`) was wrongly applied to CDDM (whose calibrated value is `1e-6`), and
   metrics were computed over all timepoints instead of CDDM's masked windows `[(0,100),(200,300)]`.
   Both must be fixed before any CDDM number is quoted.
2. **`frm`'s untuned units** — the tuning regression uses *target bit states* only. A unit encoding
   transitions, timing, or a nonlinear conjunction scores low without being uninformative. The
   control (add input pulse trains and bit-product regressors) has **not** been run.
3. **N=4000 for `frm`/`both` does not exist.** It is not merely missing: at N=4000 the `frm`
   penalty OOMs on a 46 GB GPU, and a 400k run would need 136–251 h against a 96–144 h cluster cap.
   Any N-scaling claim for the penalised conditions stops at N=2000.
4. **A3 (the reverse arm of the intervention) had not converged at 50k** — its effect is a lower
   bound.

---

## 7. Figures

Copy, never symlink, and record provenance. Create `sync_figures.sh`:

```bash
#!/usr/bin/env bash
# Copy named figures from the analysis repo and record where each came from.
set -euo pipefail
SRC=~/Documents/GitHub/trainRNNbrain
COMMIT=$(git -C "$SRC" rev-parse --short HEAD)
DATE=$(date +%Y-%m-%d)
mkdir -p figures
for f in "$@"; do
  cp "$SRC/img/internal_figures/$f" figures/
  printf '| `%s` | %s | `%s` | %s |\n' "$f" "${f%%_*}" "$COMMIT" "$DATE" >> figures/SOURCES.md
done
```

Candidate main-text figures, with the script that produces each:

| figure | script | shows |
|---|---|---|
| `selectivity_matrix.png` | `flipflop_selectivity_matrix.py` | median R² and Hoyer sparsity over the (N,k) grid |
| `switch_summary.png` | `flipflop_switch_summary.py` | all four intervention arms on one page |
| `switch_trajectories_rep1.png` | `flipflop_switch_trajectories.py` | per-unit occupancy through the switch |
| `dimensionality_matrix.png` | `flipflop_dimensionality.py` | effective dimensionality over the grid |
| `unitcloud_sel_N2000_k3.gif` | `flipflop_unitcloud.py` | the selectivity star (**animated — needs a still frame for print**) |
| `temporal_pr_N2000_k3.png` | `flipflop_temporal_pr.py` | per-unit temporal PR distributions |
| `arms_matrix.png` | `flipflop_arms.py` | arm purity, evenness, 2k-cluster separability |

⚠️ `switch_trajectories.png` (no `_rep1`) plots the **one run of twelve** that hit a gradient-spike
instability. Use `_rep1`. The rep-0 version is retained only as a documented example.

⚠️ **Nature Communications will require a Code Availability statement**, and probably a data one.
The code is at `github.com/engellab/trainRNNbrain`; the dataset is 63 GB and will need a repository
(Zenodo/Figshare) — `dead_ReLU_data/README.md` documents its structure, manifest and checksums.

---

## 8. Overleaf sync via GitHub

Do this **after** the first commit is pushed, in the Overleaf web UI:

1. Confirm Overleaf Pro is active (Princeton has a site licence; GitHub sync is a paid feature).
2. New Project → **Import from GitHub** → authorise → pick `dead-relu-paper`.
3. Overleaf tracks the repo's **default branch** (`main`). Keep the paper on `main`; do not add
   branches unless you have confirmed Overleaf follows them.
4. Thereafter: *Menu → GitHub → Pull/Push*. Overleaf does not sync continuously — pushes and pulls
   are manual, so pull before editing locally and push after editing in Overleaf.

⚠️ **Conflicts are the main hazard.** Editing the same file in both places between syncs produces a
merge conflict Overleaf handles poorly. Convention: edit in one place at a time, and always sync
before switching.

If GitHub sync turns out to be unavailable, the alternative is Overleaf's **git bridge** — an
Overleaf project is itself a git repo:

```bash
git remote add overleaf https://git.overleaf.com/<project-id>
git push overleaf main:master
```

---

## 9. Definition of done

- [ ] Private repo `dead-relu-paper` under Pavel's account, pushed
- [ ] `main.tex` compiles to a PDF locally (`make pdf`)
- [ ] Sections seeded from `docs/paper.md`, `⬜` items carried through as visible TODOs
- [ ] Figures copied with `figures/SOURCES.md` populated
- [ ] The four `.npz` caches copied into `caches/`
- [ ] `README.md` states where the data lives and how sync works
- [ ] Overleaf project linked and one round-trip (push → edit → pull) tested
- [ ] Journal requirements in §3 replaced with values verified from the publisher
