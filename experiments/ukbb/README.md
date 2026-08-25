# UKBB experiments — paper Section 6

Synthetic confounding on UK Biobank T1 structural MRI. A ResNet50 backbone is
trained under a known age/sex confounder; the linear covariate head is then
**refit from the frozen backbone** to control the omitted-variable bias. Every
paper output is regenerated from the released backbone checkpoints without
retraining the CNN.

## Stages → outputs → figure

| Script | Reads | Writes | Figure |
|---|---|---|---|
| `train_backbones.py` | UKBB HDF5 + pretrained ResNet | `runs/final_v2/coef=*/fold=*/base_{full,half,half_B}.pt` | — (source of truth; not in the reproducible loop) |
| `refit_from_checkpoints.py` | `base_{full,half,half_B}.pt` | per-fold `record.npz` + refit heads in `runs/final_v2_refit/` | Panel A/B |
| `run_crossfit.py` | `k{K}/backbone_k*.pt`, `full/base_full.pt` | per-fold `record.npz` in `runs/<run>_refit/` | K-fold robustness |
| `aggregate.py` | per-fold `record.npz` | `rexports/*.csv`, `raw_results.csv`, `crossfit_results.csv` | — |
| `figures/UKBB_application_fig.R` | the CSVs above | `runs/final_v2_refit/graphics/Fig_UKBB_application_{main,appendix}.pdf` | A/B |

Method keys in the CSVs: `base_full` (uncontrolled DNN), `base_half` (backbone on one
half), `refit_age` / `refit_age_sex` (single-split refit, RefitCovarNetwork on the h2
half), `crossfit_age` / `crossfit_age_sex` (2-fold cross-fit ensemble via
`CrossFitEnsemble`), and `dnn` / `refit_nosamp_*` / `refit_split_*` / `crossfit_k{K}_*`
from the general K-fold study.

## Layout

- `common/` — shared machinery: `data.py` (HDF5 loading, the synthetic-confounding
  DGP, `NumpyCovarDataset`, loaders) and `backbone.py` (the pretrained 3D ResNet50).
- `train_backbones.py` — trains the three sample-split backbones per (coef, fold).
  Kept for provenance only; the checkpoints are the irreplaceable artefact.
- `refit_from_checkpoints.py` — the reproducible core: refit the age and age+sex heads
  and the 2-fold cross-fit from the frozen backbones; one raw record per fold.
- `run_crossfit.py` — general K-fold cross-fit from released `backbone_k*.pt`, ensembled
  with `CrossFitEnsemble` (the K=2/K=3, n5k/n10k robustness runs).
- `aggregate.py` — reduces the per-fold records to the R-ready CSVs.
- `figures/` — one R script per figure, reading only CSVs.

Every stage uses fixed run directories with skip-if-exists resume: one output file per
(coef, fold); rerunning skips finished folds. Cross-fitting always goes through
`CrossFitEnsemble` from `src/cocodeel/crossfit.py` (`.recenter(pooled)` before effects
are reported) — no ad-hoc fold-averaging in experiment code.

## Reproducing Panel A/B

From `~/Research/ovb-ddns/code/` in the `dl-mri` env, given the released checkpoints in
`runs/2026-04-26_13-16-37_final_v2/`:

```
python experiments/ukbb/refit_from_checkpoints.py --coefs 0.0,2.0 --folds 0,1,2,3,4
python experiments/ukbb/aggregate.py --run experiments/ukbb/runs/final_v2_refit
Rscript experiments/ukbb/figures/UKBB_application_fig.R
```

`--gpu 0` by default (keep GPU 1 free). `refit_from_checkpoints.py` reads the base
backbones from `--src-run` and writes regenerated rexports + the 2-fold cross-fit to
`--out-run` (`runs/final_v2_refit/`); the figure's K=3 column and no-sample-split
baseline are carried from the released runs unchanged.

## Status

- Run artefacts (`runs/`, ~96 GB of checkpoints + per-run CSVs) are gitignored and
  access-controlled UKBB data cannot be redistributed. The paper-relevant runs are
  hashed in `~/Research/ovb-ddns/baselines/sha256.txt`.
- The attribution-map pipeline (`lrp/`) lives on the `dev` branch; it is not part of
  the paper's reproducible loop.
- The R figure packages (`ggplot2`, `dplyr`, `tidyr`, `mltools`, `grid`) must be
  installed in the `dl-mri` R separately.

## Data access

UK Biobank data is access-controlled and cannot be redistributed.
