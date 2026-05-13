# ADNI experiments — NOT part of the NeurIPS 2026 submission

This branch (`adni`) preserves ADNI experimental work for future
ADNI-focused publications. **The NeurIPS 2026 manuscript's Section 6 is
UKBB only** — nothing in this directory is referenced by `paper/paper.tex`
in its current form.

## What's here

- `ADNI_SexAD_Synthetic_Study.ipynb` — legacy 10-fold notebook; original
  source for the baseline checkpoints in `runs/2026-03-14_06-54-41/`.
- `ADNI_SexAD_Synthetic_Study.py` — Python export of the notebook,
  jupytext-formatted. The canonical version (the notebook's JSON cells
  still hold the old proj-orth paths and are not re-run).
- `run_adni_k3_crossfit.py` — current K=3 cross-fit runner with 5-fold
  outer CV (StratifiedGroupKFold, subject-grouped).

## Status

Imported from `mpff/proj-orthogonalisation/experiments/adni/` on 2026-05-12
(Phase 2 of the monorepo refactor — see
`~/.claude/plans/ok-we-have-a-cozy-lecun.md`). The `sys.path.insert` hacks
in the `.py` scripts have been retargeted to the new layout
(cocodeel = code root; nitorch = code/external/nitorch/; backbones =
code/experiments/common/). Phase 3 will replace these with
`pip install -e .` and proper packaging.

## Workflow

This branch sits on top of `main`. Phases 3–7 will be implemented on
`main`; Phase 8 includes a `git rebase adni main` to bring those
improvements into this branch.

Until then: switch to this branch when working on ADNI
(`git switch adni`), back to `main` for the NeurIPS submission line.

## Data access

ADNI data is access-controlled (separate from UK Biobank); see the ADNI
website for the application procedure.
