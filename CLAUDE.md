@~/.claude/rules/code-style.md

# Project: cocodeel
**Role:** Core Python package implementing the ovbdnn method. Related paper repo: `controls-paper` (ovbdnn).

## Method Background
- **Key insight (1):** A neural network component f_i acts as a feature extractor producing a design matrix Φ_i, with f_i = Φ_i β_i. Only f_i — not β_i — is the quantity of scientific interest.
- **Key insight (2):** When fitting additive models end-to-end with SGD, the full predictor η converges but individual components f_i may not — likely caused by concurvity.

## Scientific Goal
PyTorch implementation of post-hoc backfitting with ridge penalization for additive DNNs with control variables. Provides `PostHocCovarNetwork`: takes a pretrained backbone, refits the last layer to include covariate effects, with optional post-hoc orthogonalization.

## Architecture Overview

### Package (`cocodeel/`)
- `model.py` — `_BaseCovarNetwork` (base class), `BaseNetwork` (no covariates), `CovarNetwork` (end-to-end training with covariates)
- `posthoc_model.py` — `PostHocCovarNetwork`: the main contribution. Post-hoc IRLS backfitting with ridge penalty and validation-based λ selection. Pure PyTorch (no R dependency as of dev branch).
- `transform.py` — `Center` (centering module), `LinearRegressOut` (regresses out Z from fX for orthogonalization)
- `dataset.py` — `CovarDataset`: returns batches as `{"X": ..., "Z": ..., "y": ...}`
- `trainer.py` — `covar_trainer`: simple training loop with Adam, ReduceLROnPlateau, early stopping

### Key Design Decisions
- Batches are dicts with keys `"X"`, `"Z"`, `"y"` — always use this convention
- Centering is handled by `Center` modules stored as buffers (travels with model state)
- `PostHocCovarNetwork.fit(train_loader, val_loader)` — takes separate train/val loaders
- λ path: log-spaced from `lambda_max` to `lambda_min` (following glmnet convention), with adaptive path expansion if optimum is at boundary
- IRLS loop exits after one iteration for identity link (Gaussian case)

### Supported Link Functions
- `"identity"` — Gaussian/linear
- `"logit"` — Bernoulli/binary classification
- `"log"` — Poisson

## Experiments (`experiments/`)
- `simulation_images/` — simulation study with synthetic image data (notebooks + R scripts for figures)
- `simulation_linear/` — linear simulation baseline

## Key Files
- `experiments/simulation_images/1-simulation.ipynb` — runs simulation
- `experiments/simulation_images/2-evaluation.ipynb` — evaluates results
- `experiments/simulation_images/3-figures.ipynb` — generates figures
- `experiments/simulation_images/4-Figure*.R` — final paper figures (R/ggplot2)
- `results/simulation_images/` — CSV results per simulation setting
- `graphics/` — output figures (PDFs) for the paper

## Environment
Python 3.13, conda. Key dependencies: PyTorch, numpy, pandas, scikit-learn, nibabel, nilearn, matplotlib, scipy, seaborn, pytest, opencv.
Note: `environment.yml` still lists `rpy2/r-base/r-glmnet` but these are no longer used in the dev branch code — environment.yml needs updating.

## Coding Conventions
- Python, PyTorch
- All models inherit from `nn.Module`
- No R/rpy2 dependency (removed in dev branch)
- Tests in `tests/` using pytest

## Branch Status
- `main`: stable, uses rpy2/glmnet for λ selection
- `dev`: active development — pure PyTorch λ selection, separate train/val loaders, `LinearRegressOut` in transform.py. To be merged to main soon.
