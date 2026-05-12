# UKBB experiments — paper Section 6

Code and figure scripts for *"Controlling for Omitted Variable Bias in Deep
Neural Networks"*, Section 6: synthetic confounding on UK Biobank T1
structural MRI.

## What's here

- `ukbb_common.py` — shared loaders (`load_ukbb_data`), `resample_synthetic`,
  `NumpyCovarDataset`, `fast_loader`, model/trainer defaults,
  `setup_run_dir`, `write_manifest`.
- `run_ukbb_experiment.py` — source-of-truth training (K=2 sample-split,
  5-fold × {coef=0, 2.0}, produces `final_v2` checkpoints).
- `run_k_crossfit.py` — K-fold cross-fit eval-only on existing checkpoints
  (current default).
- `run_crossfit_prototype.py` — older K=2 A/B mirror; augments `final_v2`
  with role-flipped half.
- `refit_posthoc.py` — post-hoc refit using existing `base_half` checkpoints.
- `figures/UKBB_application_fig.R` — Panel A + B + appendix; reads CSVs
  from `runs/.../rexports/`.
- `lrp/` — Layer-wise relevance propagation (Panel C) attribution maps.

## Status

This directory was imported from
`mpff/proj-orthogonalisation/experiments/ukbb/` on 2026-05-12 (Phase 2 of
the monorepo refactor — see `~/.claude/plans/ok-we-have-a-cozy-lecun.md`).
Imports currently use `sys.path.insert` hacks (resolved relative to
`code_root`); Phase 3 will replace these with `pip install -e .`.

## Reproducing Section 6 figures

Full per-figure reproduction recipes (with compute budgets) come in
Phase 6 / 7. For now, scripts run from
`~/Research/ovb-ddns/code/` (`dl-mri` env required) — see each script's
`--help` for available flags.

Run artefacts (checkpoints + per-run CSVs) live in `runs/` (gitignored,
~96 GB). Paper-relevant runs are listed in
`~/Research/ovb-ddns/baselines/sha256.txt`.

## Data access

UK Biobank data is access-controlled and cannot be redistributed. See
forthcoming `code/docs/data_access.md` (Phase 7) for the application
procedure.
