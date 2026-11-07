# Application to neuroimaging data with synthetic confounding (Section 6)

Predict high alcohol consumption (`highalc`) from T1-weighted brain MRI of the UK Biobank, with `age` as the control variable. Each outer training fold is resampled to a logistic DGP in which `sex` carries the signal (`β_sex = 2`) and `age` the confounding (`β_age ∈ {0, -2}`, the `coef` of the scripts). The test set is resampled with `β_age = 0`, so a model that uses `age` as a shortcut loses AUC on it. The backbone is a 3D ResNet-50 pretrained on Kinetics (`common/backbone.py`).

| Script | Does | Writes to `runs/` |
|---|---|---|
| `train_backbones.py` | trains the DNNs without controls: on all 5,000 observations, on each half, and on each two thirds | `backbones/` |
| `refit_from_checkpoints.py` | DNN with Controls: Sample split and Cross-fit (K=2) | `refit/` |
| `run_crossfit.py` | DNN with Controls: Cross-fit (K=3) | `crossfit_k3/` |
| `competitors_from_checkpoints.py` | No sample split, orthogonalisation, PHO + Refit, LEACE, DFR | `refit/competitor_results.csv` |
| `train_joint.py` | Joint DNN `g(X, age)`, marginalised over age | `refit/joint_results.csv` |
| `aggregate.py` | reduces per-fold records to CSVs | `refit/`, `crossfit_k3/` |
| `figures/UKBB_application_fig.R` | Fig 5 and the appendix figure and table (`ggplot2`, `dplyr`, `tidyr`, `mltools`) | `refit/graphics/` |

## Running

From the repository root, with the data in `data/ukbb/` (or `UKBB_DATA_ROOT`) and the Kinetics ResNet-50 weights `r3d50_K_200ep.pth` in its `pretrained_models/`:

```
python data/ukbb/make_dummy_data.py              # without UK Biobank access
python experiments/ukbb/train_backbones.py --gpu 0
python experiments/ukbb/refit_from_checkpoints.py --gpu 0
python experiments/ukbb/run_crossfit.py --gpu 0
python experiments/ukbb/competitors_from_checkpoints.py --gpu 0
python experiments/ukbb/train_joint.py --gpu 0
python experiments/ukbb/aggregate.py
python experiments/ukbb/aggregate.py --run experiments/ukbb/runs/crossfit_k3
Rscript experiments/ukbb/figures/UKBB_application_fig.R
```

Every script skips the (coef, fold) pairs whose outputs exist, so an interrupted run resumes.
