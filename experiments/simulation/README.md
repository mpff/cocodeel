# Simulation study (Section 5)

Images from a synthetic "traffic-light" DGP with known `f_X`, `f_Z` and a control variable `Z` correlated with the image. Because the true effects are known, every estimator can be scored by the bias², variance and MSPE of `f̂_X`, `f̂_X^re` and `f̂_Z` over replications.

| Script | Sweeps | Figure scripts |
|---|---|---|
| `study_a_linear_consistency.py` | `increasing_bz`, `binary_increasing_bz`, `increasing_q`, `increasing_cv`, `increasing_p` | `figure_a1_linear_bz.R`, `figure_a2_binary_bz.R`, `figure_a3_adversarial_settings.R` |
| `study_b_misspecification.py` | `nonlinear_fz`, `nonlinear_fz_misspec` | `figure_b1_misspecification_fz.R`, `figure_b1_misspecification_fz_curve.R`, `figure_b2_misspecification_raw_z.R` |
| `study_c_concurvity_benchmark.py` | `concurvity`, `concurvity_q` | `figure_c1_concurvity.R`, `figure_c2_concurvity_backbone_width.R` |
| `study_d_noisy_z.py` | `increasing_a` | `figure_d1_noisy_z.R` |

Method keys in the CSVs: `refit` and `refit_orth` are the DNN with Controls without and with orthogonalisation (2-fold cross-fit), `base` the DNN without controls, `posthoc_web` its orthogonalised predictions, `posthoc_web_features` PHO + Refit (Weber et al., 2023), `leace` LEACE (Belrose et al., 2023), `nam` and `nam_mlp` neural additive models with linear and MLP `f_Z`, `nam_mlp_conc` the MLP NAM with the concurvity penalty of Siems et al. (2023), `ssn` the semi-structured network of Rügamer et al., and `joint` a joint DNN `g(X, Z)`.

## Layout

- `common/`: the DGP (`dgp.py`), the CNN backbone (`backbone.py`), the sample split into halves for cross-fitting (`loaders.py`), and a resumable multiprocessing runner (`grid_runner.py`).
- `study_*.py`: one script per study, hyperparameters as constants at the top.
- `hpsearch/`: the grid searches that chose those hyperparameters, with their results as JSON.
- `aggregate.py`: reduces per-seed predictions to `output/<sweep>.csv`.
- `figures/`: one R script per figure (`readr`, `dplyr`, `ggplot2`, `latex2exp`, `patchwork`, `tikzDevice`, `png`).

## Running

From the repository root:

```
python experiments/simulation/study_a_linear_consistency.py
python experiments/simulation/aggregate.py --out-dir experiments/simulation/output/new
Rscript experiments/simulation/figures/figure_a1_linear_bz.R
```

`NSIM` sets the number of replications and `COCODEEL_DEVICE` the device. A study writes to `output/runs/study_*/` and skips every (sweep, setting, seed) already there, so an interrupted run resumes. `aggregate.py` never overwrites an existing CSV.
