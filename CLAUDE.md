# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@~/.claude/rules/code-style.md

# Project: cocodeel
**Role:** Core Python package implementing the ovbdnn method. Related paper repo: `controls-paper` (ovbdnn).

## Method Background
- **Key insight (1):** A neural network component f_i acts as a feature extractor producing a design matrix Φ_i, with f_i = Φ_i β_i. Only f_i — not β_i — is the quantity of scientific interest.
- **Key insight (2):** When fitting additive models end-to-end with SGD, the full predictor η converges but individual components f_i may not — likely caused by concurvity.

## Scientific Goal
PyTorch implementation of post-hoc backfitting with ridge penalization for additive DNNs with control variables. Provides `PostHocCovarNetwork`: takes a pretrained backbone, refits the last layer to include covariate effects, with optional post-hoc orthogonalization.

## Commands

```bash
# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_posthoc_model.py

# Run a single test
pytest tests/test_model.py::TestBaseNetwork::test_center_features_updates_mean_and_intercept

# Activate the conda environment
conda activate dl-mri
```

No build or lint step — pure Python package, no pyproject.toml/setup.py.

## Architecture Overview

### Package (`cocodeel/`)
- `model.py` — `_BaseCovarNetwork` (base class with centering logic, GLM utilities), `BaseNetwork` (no covariates), `CovarNetwork` (end-to-end training with covariates)
- `posthoc_model.py` — `PostHocCovarNetwork`: the main contribution. Post-hoc IRLS backfitting with ridge penalty, X/Z internal standardization, glmnet-style λ path with adaptive expansion, coefficient-change convergence, and validation-based λ selection. Pure PyTorch.
- `benchmarking/posthoc_model.py` — `PostHocOrthNetwork`, `SemiStructuredNetwork`: older comparison baselines used in simulation experiments.
- `transform.py` — `Center` (centering module stored as buffer), `LinearRegressOut` (regresses out Z from fX)
- `dataset.py` — `CovarDataset`: returns batches as `{"X": ..., "Z": ..., "y": ...}`
- `trainer.py` — `covar_trainer`: training loop with Adam, configurable LR scheduler, early stopping, optional bf16 autocast

### Key Design Decisions
- Batches are always dicts with keys `"X"`, `"Z"`, `"y"` — every model and trainer assumes this
- Centering is handled by `Center` modules stored as buffers (travels with model state dict)
- `is_centered` is a buffer (not parameter) — guards against double-centering; `center_effects()` is idempotent
- `PostHocCovarNetwork.fit(train_loader, val_loader, lam=None, max_iters=50, tol=1e-2, penalty_z=None, n_lambdas=100)` — takes separate train/val loaders; calls `center_effects` internally
- X and Z are internally centered and standardized before the ridge solve; `fx.weight` and `fz.weight` are de-standardized at the end so coefficients are returned in the centered-raw scale
- λ path: log-spaced from glmnet-style `lambda_max = max|Xᵀy| / (N·α)` down to `1e-6 · lambda_max` (or `1e-3 · lambda_max` in the high-dimensional regime `N < d`), with adaptive expansion if the optimum lands at the boundary
- IRLS convergence is checked on coefficient change (`Δβ_fx`, `Δβ_fz`), not on prediction change. Unconverged solutions are rejected unless no λ converges — then the best-loss unconverged state is used with a warning.
- `covar_trainer(..., scheduler=None, scheduler_kwargs=None, use_amp=False)` — pass a scheduler *class* (e.g. `torch.optim.lr_scheduler.StepLR`) and its kwargs; defaults to `ReduceLROnPlateau(patience=max(1, patience // 3), factor=0.5)` if `None`. bf16 autocast is opt-in via `use_amp=True` (requires Ampere+); fp16 not supported (no GradScaler).
- Orthogonalization in `PostHocCovarNetwork`: stored in `self.orth` (Linear layer). `predict_fx(x, z)` subtracts `orth(z)`; `predict_fz(z)` adds it back — so total η is unchanged.

### Attributes available after `.fit()`
- `PostHocCovarNetwork`: `lam` (selected λ), `lambda_path_` (list of per-λ diagnostics: val_loss, converged, n_iters, Δβ, β-norms), `max_iters_`, `tol_`, `n_lambdas_`.
- `covar_trainer`-returned model: `val_losses_`, `lr_history_`, `best_epoch_`, `n_epochs_run_`.

### Supported Link Functions
- `"identity"` — Gaussian/linear
- `"logit"` — Bernoulli/binary classification
- `"log"` — Poisson

### Standard Workflows

**End-to-end:**
```python
model = covar_trainer(CovarNetwork, model_params, train_loader, val_loader, patience=12)
model = model.center_effects(train_loader)
```

**Post-hoc (sample-split, recommended):**
```python
# Two disjoint samples: A trains the backbone, B refits the posthoc.
base = covar_trainer(BaseNetwork, model_params, train_loader_A, val_loader_A)
posthoc = PostHocCovarNetwork(base, num_covariates=1, orthogonalize=False)
posthoc = posthoc.fit(train_loader_B, val_loader_B)  # centers on B internally
```

**Why sample-splitting.** The posthoc features `H = phi(X; theta*)` are a
generated regressor: `theta*` depends on `y`, so fitting on the backbone's
training sample violates exogeneity (`E[Hᵀε] ≠ 0`) and biases the FWL+ridge
refit (Pagan 1984). On a sample disjoint from the backbone's training set,
`H` is a deterministic function of `X` and FWL+ridge regain unbiasedness.
See `research/session-B-endogeneity-notes.md` and `research/session-D-synthesis.md`
for the full derivation and UKBB evidence. The
`experiments/simulation_images/utils.py:simulate_dataloaders_split` helper
builds three disjoint partitions (`full`, `half_A`, `half_B`) from one draw.

**Post-hoc (same-sample, legacy — biased):**
```python
# Only use for comparison against the split recipe. Biased under endogeneity.
base = covar_trainer(BaseNetwork, model_params, train_loader, val_loader)
posthoc = PostHocCovarNetwork(base, num_covariates=1, orthogonalize=False)
posthoc = posthoc.fit(train_loader, val_loader)
```

### Testing Conventions
- Tests use `DummyBackbone` — flattens input, `out_features` controls width
- Key invariants to test: centering does not change predictions; orthogonalization does not change η; state_dict round-trip preserves all behavior

## Experiments (`experiments/`)
- `simulation_images/` — simulation study with synthetic image data (notebooks + R scripts for figures)

### Key Files
- `experiments/simulation_images/1-simulation.ipynb` — exploratory notebook for the sample-split recipe (`simulate_and_fit`).
- `experiments/simulation_images/hp_search.py` — HP search over `lr × wd × early_patience × sched_patience` on two N anchors per outcome type. Writes `results/simulation_images/hp_search/chosen_hps.json`.
- `experiments/simulation_images/run_full_simulation.py` — resumable nsim=50 runner across six blocks (binary_increasing_bz, increasing_bz, increasing_cv, increasing_q, increasing_p, concurvity) on 4 workers. Saves per-sim NPZ predictions and a JSONL progress log.
- `experiments/simulation_images/aggregate_full_simulation.py` — NPZ → long-form CSV (`model, effect, metric, value, n, <sweep>`), one CSV per block; consumed by the R figure scripts.
- `experiments/simulation_images/4-Figure{1,2,3,4}_*.R` — paper figures. Fig 1 (continuous bz + appendix S1), Fig 2 (concurvity; deferred), Fig 3 (binary/IRLS), Fig 4 (adversarial q + p + cv1). `4-Rfunctions.R` holds shared theme/helper.
- `experiments/simulation_images/utils.py` — `simulate_dataloader` and `simulate_dataloaders_split` (three disjoint partitions from one draw).
- `experiments/simulation_images/2-evaluation.ipynb` — legacy evaluator (from the timestamped-CSV pipeline); superseded by `aggregate_full_simulation.py`. Kept for reference.
- `results/simulation_images/` — CSV results per simulation setting + `hp_search/chosen_hps.json` + `runs/<timestamp>/` per-sim NPZs (not tracked).
- `graphics/` — output figures (PDFs) produced by the R scripts (not tracked — regenerate by rerunning the R scripts).

## Experiment Management

### Tooling
- Config management: hardcoded in notebooks (no Hydra/YAML)
- Experiment tracking: none (results written to `results/` as CSVs)
- Notebook workflow: raw `.ipynb` (not version-controlled; only committed when final)

### Experiment Structure
Results in `results/simulation_images/<setting>.csv`. Notebooks in `experiments/simulation_images/` follow numbered sequence (1-simulation, 2-evaluation, 4-Figure*.R). No timestamp subfolders — each setting has a fixed output filename.

### What counts as a final result
A result is final when CSV is committed to `results/`, figures are committed to `graphics/`, and the notebook that produced them is committed. Config (simulation parameters) must be recoverable from the committed notebook.

## Environment
Python 3.13, conda env `dl-mri`. Key dependencies: PyTorch, numpy, pandas, scikit-learn, nibabel, nilearn, matplotlib, scipy, seaborn, pytest, opencv. The R figure scripts require `readr, dplyr, ggplot2, tikzDevice, latex2exp, png, grid, patchwork`.

## Coding Conventions
- Python, PyTorch; all models inherit from `nn.Module`
- No R/rpy2 dependency in current main branch code
- Tests in `tests/` using pytest (files use `unittest.TestCase`)
