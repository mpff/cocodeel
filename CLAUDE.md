# Project: Code for the Paper "Controlling for Omitted Variable Bias in Deep Neural Networks"

## Scientific Goal
PyTorch implementation of post-hoc crossfitting with ridge penalization for additive DNNs with control variables. 
Provides `PostHocCovarNetwork`: takes a pretrained backbone, refits the last layer to include covariate effects, with optional post-hoc orthogonalization.

## Commands

```bash
# Run all tests
pytest tests/

# Activate the conda environment
conda activate dl-mri
```

## Architecture Overview

### Package (`src/`)
- `model.py` — `_BaseCovarNetwork` (base class with centering logic, GLM utilities), `BaseNetwork` (no covariates)
- `posthoc_model.py` — `PostHocCovarNetwork`: the main contribution. Post-hoc IRLS backfitting with ridge penalty, X/Z internal standardization, glmnet-style λ path with adaptive expansion, coefficient-change convergence, and validation-based λ selection. Pure PyTorch.
- `benchmarking/` — competitor methods, not part of the method itself: `model.py` (`CovarNetwork`: end-to-end NAM-style training with covariates), `posthoc_model.py` (`PostHocOrthNetwork`, `SemiStructuredNetwork`).
- `transform.py` — `Center` (centering module stored as buffer), `LinearRegressOut` (regresses out Z from fX)
- `dataset.py` — `CovarDataset`: returns batches as `{"X": ..., "Z": ..., "y": ...}`
- `trainer.py` — `covar_trainer`: training loop with Adam, configurable LR scheduler, early stopping, optional bf16 autocast

### Key Design Decisions
- Batches are always dicts with keys `"X"`, `"Z"`, `"y"` — every model and trainer assumes this
- Centering is handled by `Center` modules stored as buffers (travels with model state dict)
- `PostHocCovarNetwork.fit(train_loader, val_loader, lam=None, max_iters=50, tol=1e-2, penalty_z=None, n_lambdas=100)` — takes separate train/val loaders; calls `center_effects` internally.
- X and Z are internally centered and standardized before the ridge solve; `fx.weight` and `fz.weight` are de-standardized at the end so coefficients are returned in the centered-raw scale
- Orthogonalization in `PostHocCovarNetwork`: stored in `self.orth` (Linear layer). `predict_fx(x, z)` subtracts `orth(z)`; `predict_fz(z)` adds it back — so total η is unchanged.

### Attributes available after `.fit()`
- `PostHocCovarNetwork`: `lam` (selected λ), `lambda_path_` (list of per-λ diagnostics: val_loss, converged, n_iters, Δβ, β-norms), `max_iters_`, `tol_`, `n_lambdas_`.
- `covar_trainer`-returned model: `val_losses_`, `lr_history_`, `best_epoch_`, `n_epochs_run_`.

### Supported Link Functions
- `"identity"` — Gaussian/linear
- `"logit"` — Bernoulli/binary classification

### Standard Workflows

**End-to-end (benchmark):**
```python
from cocodeel.benchmarking.model import CovarNetwork
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
