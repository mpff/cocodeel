# Copilot Instructions for cocodeel

## Project Overview
`cocodeel` is a PyTorch library for deep learning models with control variables. The core concept: train neural networks that separate feature effects f(X) from covariate effects f(Z), enabling control variable approaches in deep learning. We can additionally fit an orthogonalization of f(X) onto Z so that the effects are uncorrelated and mediated effects of Z are removed from f(X). The library implements both end-to-end training and post-hoc methods.

## Core Architecture Patterns

### Model Hierarchy
- **BaseNetwork**: Basic neural network that ignores covariates (Z)
- **CovarNetwork**: Full covariate network training f(X) + f(Z) end-to-end using SGD
- **PostHocCovarNetwork**: Freezes backbone, fits linear effects using R's glmnet with ridge regression, includes optional orthogonalization of f(X) with respect to Z.

All models inherit from `_BaseCovarNetwork` with centering logic via `Center` transforms. The centering logic is important to make estimated effects comparable across the different model types. In particular the intercept can then be interpreted as the mean of y (linear model) in the training data and the effects f(X) and f(Z) are mean-centered over the training data.

There are two main categories of models:
1. **End-to-End Models** (`CovarNetwork`): Train f(X) and f(Z) jointly using backpropagation.
2. **Post-Hoc Models** (`PostHocCovarNetwork`): Train f(X) first, then fit f(Z) using a separate regression step. May include optional orthogonalization.

### Required Model Components
Every model must implement:
- `forward(x, z)`: Main prediction combining intercept + f(X) + f(Z)
- `predict_fx(x, z=None)`: Feature contribution only (requires z when orthogonalizing)
- `predict_fz(z)`: Covariate contribution only (may or may not include orthogonal component of fx)
- `center_effects(dataloader)`: Fit centering transforms and update intercept. Important to call after training.

Every posthoc model must also implement:
- `fit(dataloader, fit_kwargs)`: Refit the last layer f(X) + f(Z) to include the covariate effects f(Z) after freezing the backbone.

### Data Flow Pattern
```python
# Standard workflow:
1. Initialize model: BaseNetwork(backbone, backbone_params)
2. Train with covar_trainer() 
3. Center effects: model.center_effects(train_loader)
4. Evaluate predictions: model(x, z), model.predict_fx(x, z), model.predict_fz(z)
# Post-hoc workflow:
1. Initialize base model: BaseNetwork(backbone, backbone_params)
2. Train base model with covar_trainer() 
3. Initialize posthoc model: PostHocCovarNetwork(base_model, num_covariates)
4. Fit posthoc model: posthoc_model.fit(train_loader, fit_kwargs)
5. Center effects: posthoc_model.center_effects(train_loader)
6. Evaluate predictions: posthoc_model(x, z), posthoc_model.predict_fx(x, z), posthoc_model.predict_fz(z)
```

## Key Implementation Details

### Centering is Critical
All models use `Center` transforms to mean-center features, covariates, and targets. The intercept automatically adjusts during centering to be the mean of the dataset and make predictions comparable across models. Always call `center_effects()` after training.

### Dataset Structure
Use `CovarDataset` with batch format: `{"X": images, "Z": covariates, "y": targets}`. Models detect covariate usage via `num_covariates` attribute.

### R Integration (PostHoc Models)
PostHocCovarNetwork uses rpy2 to call R's glmnet for ridge regression. Environment requires both Python packages and R dependencies (see environment.yml).

### Orthogonalization Option
PostHocCovarNetwork supports orthogonalization of f(X) with respect to Z. When enabled, `predict_fx(x, z)` requires z input to compute orthogonalized feature effects. Orthogonalization can be seen as a linear projection of f(X) onto the space orthogonal to Z, ensuring uncorrelated effects. It is a linear regression / residualization step applied to the feature effects and we save the regression coefficients as part of the model state for prediction on new data.


## Development Patterns

### Testing Convention
- Tests use `DummyBackbone` for deterministic model testing
- Test centering invariance: predictions unchanged after `center_effects()`
- Test orthogonalisation: fx uncorrelated with Z after orthogonalization (on training data), predictions unchanged after orthogonalization
- Verify shape compatibility: models handle batch dimensions correctly, model handle x,z inputs correctly, multiple covariates handled correctly, etc.
- Verify saving and loading model state_dicts: predictions consistent after reload, centering preserved, orthogonalization preserved, R model parameters preserved for PostHoc models, etc.

### Experiment Structure
- Notebooks in `experiments/` follow numbered sequence (1-simulation.ipynb, 2-evaluation.ipynb)
- Simulation utilities in utils.py: `simulate_dataloader()`, `evaluate_model()`, `mspe_decomposition()`
- Results stored in structured `outputs/` directory with timestamp folders

### Training Pattern
Use `covar_trainer()` with early stopping:
```python
model = covar_trainer(
    model=CovarNetwork, 
    model_params={"backbone": backbone, "num_covariates": 1},
    train_loader=train_loader,
    val_loader=val_loader,
    patience=12
)
model = model.center_effects(train_loader)
```

Then fit posthoc models if needed (with or without orthogonalization)
```python
posthoc_model = PostHocCovarNetwork(base_model=model, num_covariates=1, orthogonalize=False)
posthoc_model = posthoc_model.fit(train_loader, fit_kwargs={"lam": 0.0})  # no regularization
posthoc_model = posthoc_model.center_effects(train_loader)
```

Models can then be evaluated using the standard prediction methods.
```python
preds = model(x, z)
fx = model.predict_fx(x)  # or model.predict_fx(x, z) if orthogonalized
fz = model.predict_fz(z)
```

## File Organization
- `cocodeel/`: Core library (model.py, dataset.py, trainer.py, transform.py, posthoc_model.py)
- `experiments/`: Research notebooks and simulation code
- `tests/`: Unit tests following unittest framework
- `outputs/`: Experiment results with parametric naming scheme
- `environment.yml`: Conda environment with Python and R dependencies

When adding new models, follow the base class pattern with proper centering and component separation.
If it is a posthoc model, ensure refitting routine and R integration is correctly implemented.


## Environment Setup
The project requires both Python and R dependencies. Use the provided `environment.yml` to create a conda environment with all necessary packages, including rpy2 for R integration. The standard name used for the environment is `dl-mri`. You can activate the environment using `conda activate dl-mri`.