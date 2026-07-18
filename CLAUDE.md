# Project: Code for the Paper "Controlling for Omitted Variable Bias in Deep Neural Networks"

## Scientific Goal
PyTorch implementation of cross-fitted last-layer refitting with ridge penalization for additive DNNs with control variables and the simulation study, benchmark and application from the paper.


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
- `refit_model.py` — `RefitCovarNetwork`: the main contribution. IRLS backfitting with ridge penalty, X/Z internal standardization, glmnet-style λ path with adaptive expansion, coefficient-change convergence, and validation-based λ selection. Pure PyTorch.
- `benchmarking/` — competitor methods, not part of the method itself: `model.py` (`CovarNetwork`: end-to-end NAM-style training with covariates), `posthoc_model.py` (`PostHocOrthNetwork`, `SemiStructuredNetwork`), `adversarial_trainer.py` (`adversarial_trainer`: br-net/CF-Net style adversarial correlation penalty, Zhao et al. 2020, trained on top of `BaseNetwork`).
- `transform.py` — `Center` (centering module stored as buffer), `LinearRegressOut` (regresses out Z from fX)
- `dataset.py` — `CovarDataset`: returns batches as `{"X": ..., "Z": ..., "y": ...}`
- `trainer.py` — `covar_trainer`: training loop with Adam, configurable LR scheduler, early stopping, optional bf16 autocast

### Key Design Decisions
- Batches are always dicts with keys `"X"`, `"Z"`, `"y"` — every model and trainer assumes this
- Centering is handled by `Center` modules stored as buffers (travels with model state dict)
- `RefitCovarNetwork.fit(train_loader, val_loader, lam=None, max_iters=50, tol=1e-2, penalty_z=None, n_lambdas=100)` — takes separate train/val loaders; calls `center_effects` internally.
- X and Z are internally centered and standardized before the ridge solve; `fx.weight` and `fz.weight` are de-standardized at the end so coefficients are returned in the centered-raw scale
- Orthogonalization in `RefitCovarNetwork`: stored in `self.orth` (Linear layer). `predict_fx(x, z)` subtracts `orth(z)`; `predict_fz(z)` adds it back — so total η is unchanged.

### Attributes available after `.fit()`
- `RefitCovarNetwork`: `lam` (selected λ), `lambda_path_` (list of per-λ diagnostics: val_loss, converged, n_iters, Δβ, β-norms), `max_iters_`, `tol_`, `n_lambdas_`.
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

**Adversarial (benchmark):**
```python
from cocodeel.benchmarking.adversarial_trainer import adversarial_trainer
model = adversarial_trainer(BaseNetwork, model_params, num_covariates=1, train_loader=train_loader, val_loader=val_loader)
model = model.center_effects(train_loader)  # same two-step contract as covar_trainer — required for comparability across benchmarks
```

**Refit (sample-split, recommended):**
```python
# Two disjoint samples: A trains the backbone, B hosts the refit.
base = covar_trainer(BaseNetwork, model_params, train_loader_A, val_loader_A)
refit = RefitCovarNetwork(base, num_covariates=1, orthogonalize=False)
refit = refit.fit(train_loader_B, val_loader_B)  # centers on B internally
```
