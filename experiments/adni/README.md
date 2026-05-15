# ADNI experiments — NOT part of the NeurIPS 2026 submission

ADNI experimental work, preserved for future ADNI-focused publications.
**The NeurIPS 2026 manuscript's Section 6 is UKBB only** — nothing in
this directory is referenced by `paper/paper.tex` in its current form.

## What's here

- `ADNI_SexAD_Synthetic_Study.ipynb` — legacy 10-fold notebook; original
  source for the baseline checkpoints in `runs/2026-03-14_06-54-41/`.
- `ADNI_SexAD_Synthetic_Study.py` — Python export of the notebook,
  jupytext-formatted. The canonical version (the notebook's JSON cells
  still hold the old proj-orth paths and are not re-run).
- `run_adni_k3_crossfit.py` — current K=3 cross-fit runner with 5-fold
  outer CV (StratifiedGroupKFold, subject-grouped).

## Status

The `.py` scripts use `sys.path.insert` hacks (cocodeel = code root;
nitorch = code/external/nitorch/; backbones = code/experiments/common/);
the package is not yet pip-installable.

## Data access

ADNI data is access-controlled (separate from UK Biobank); see the ADNI
website for the application procedure.

## TODO

- Split this directory to its own branch before NeurIPS submission to
  keep the submission line UKBB-only.
