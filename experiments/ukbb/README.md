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
- `refit_from_checkpoints.py` — refit using existing `base_half` checkpoints.
- `figures/UKBB_application_fig.R` — Panel A + B + appendix; reads CSVs
  from `runs/.../rexports/`.
- `lrp/` — Layer-wise relevance propagation (Panel C) attribution maps.

## Status

Imports use `sys.path.insert` hacks resolved relative to `code_root`;
the package is not yet pip-installable.

## Reproducing Section 6 figures

Scripts run from `~/Research/ovb-ddns/code/` (`dl-mri` env required) —
see each script's `--help` for available flags. Per-figure
reproduction recipes are not yet documented.

Run artefacts (checkpoints + per-run CSVs) live in `runs/` (gitignored,
~96 GB). Paper-relevant runs are listed in
`~/Research/ovb-ddns/baselines/sha256.txt`.

## Data access

UK Biobank data is access-controlled and cannot be redistributed.
