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
- `posthoc_model.py` — `PostHocCovarNetwork`: the main contribution. Post-hoc IRLS backfitting with ridge penalty and validation-based λ selection. Pure PyTorch (no R dependency).
- `benchmarking/posthoc_model.py` — `PostHocOrthNetwork`, `SemiStructuredNetwork`: older comparison baselines used in simulation experiments.
- `transform.py` — `Center` (centering module stored as buffer), `LinearRegressOut` (regresses out Z from fX)
- `dataset.py` — `CovarDataset`: returns batches as `{"X": ..., "Z": ..., "y": ...}`
- `trainer.py` — `covar_trainer`: training loop with Adam, ReduceLROnPlateau, early stopping

### Key Design Decisions
- Batches are always dicts with keys `"X"`, `"Z"`, `"y"` — every model and trainer assumes this
- Centering is handled by `Center` modules stored as buffers (travels with model state dict)
- `is_centered` is a buffer (not parameter) — guards against double-centering; `center_effects()` is idempotent
- `PostHocCovarNetwork.fit(train_loader, val_loader)` — takes separate train/val loaders; calls `center_effects` internally, so do not call it separately beforehand
- λ path: log-spaced from `lambda_max` to `lambda_min` (following glmnet convention), with adaptive path expansion if optimum is at boundary
- IRLS loop exits after one iteration for identity link (Gaussian case)
- Orthogonalization in `PostHocCovarNetwork`: stored in `self.orth` (Linear layer). `predict_fx(x, z)` subtracts `orth(z)`; `predict_fz(z)` adds it back — so total η is unchanged.

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
- `experiments/simulation_images/1-simulation.ipynb` — runs simulation using the sample-split recipe (see `simulate_and_fit`)
- `experiments/simulation_images/2-evaluation.ipynb` — evaluates results
- `experiments/simulation_images/3-smoke_test_sample_splitting.py` — contrasts same-sample vs. split recipes on paper settings
- `experiments/simulation_images/4-Figure*.R` — final paper figures (R/ggplot2)
- `experiments/simulation_images/utils.py` — `simulate_dataloader` and `simulate_dataloaders_split` helpers
- `results/simulation_images/` — CSV results per simulation setting
- `graphics/` — output figures (PDFs) produced by the R scripts

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
Python 3.13, conda env `dl-mri`. Key dependencies: PyTorch, numpy, pandas, scikit-learn, nibabel, nilearn, matplotlib, scipy, seaborn, pytest, opencv.
Note: `environment.yml` still lists `rpy2/r-base/r-glmnet` but these are no longer used in the main branch — environment.yml needs updating.

## Coding Conventions
- Python, PyTorch; all models inherit from `nn.Module`
- No R/rpy2 dependency in current main branch code
- Tests in `tests/` using pytest (files use `unittest.TestCase`)
