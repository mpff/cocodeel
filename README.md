# cocodeel

Control variables for deep neural networks, by refitting the last layer.

Code for *"Controlling for Omitted Variable Bias in Deep Neural Networks"*.
Shortcut learning in a DNN is framed as omitted variable bias: a covariate
`Z` (a confounder, an acquisition artefact, a demographic attribute) that
correlates with both the network input `X` and the outcome `y` gets its
effect wrongly attributed to `X`. The fix is the classical one from
regression analysis — include `Z` as a control variable — applied to the
last layer of a pretrained DNN, since concurvity between `X` and `Z` makes
joint end-to-end training of both effects unreliable.

`RefitCovarNetwork` refits a pretrained backbone's last layer as an
additive model `η = β₀ + f_X(X) + f_Z(Z)`, with `f_X` linear in the
backbone features and `f_Z` linear in `Z`. Fitting is a ridge-penalised
backfit (glmnet-style λ path, IRLS for non-identity links), with an
optional secondary regression that orthogonalises `f_X` against `Z`.

## Install

```
conda env create -f environment.yml
conda activate dl-mri
pip install -e .
```

`environment.yml` also pulls in R (`r-base`, `rpy2`, `r-glmnet`) — not used
by the package itself. The R figure scripts under `experiments/` need
additional R packages (`ggplot2`, `tikzDevice`, `patchwork`, …) that are
**not** in `environment.yml`; see the per-experiment `README.md` for what
each figure script needs.

## Quickstart

The estimator is only consistent if the backbone is pretrained on a sample
disjoint from the one used for the last-layer refit — otherwise the
backbone features are not exogenous (Pagan, 1984, "generated regressors";
see the sample-splitting lemma in the paper). The recipe below is the
paper's method: split the data in half, pretrain on `A`, refit on `B`.

```python
import torch
from torch.utils.data import DataLoader

from cocodeel.dataset import CovarDataset
from cocodeel.model import BaseNetwork
from cocodeel.refit_model import RefitCovarNetwork
from cocodeel.trainer import covar_trainer


class Backbone(torch.nn.Module):
    """Stand-in for a pretrained DNN backbone phi: X -> R^q."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.out_features = out_features

    def forward(self, x):
        return torch.relu(self.linear(x))


def toy_data(n, p_x=10, p_z=1):
    X = torch.randn(n, p_x)
    Z = torch.randn(n, p_z)
    y = (2.0 * X[:, 0] + 3.0 * Z[:, 0] + 1.5).unsqueeze(1)
    return X, Z, y


def loaders(X, Z, y, batch_size=32):
    ds = CovarDataset(X, Z, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True), DataLoader(ds, batch_size=batch_size)


# Two disjoint samples: A pretrains the backbone, B refits the last layer.
X_A, Z_A, y_A = toy_data(400)
X_B, Z_B, y_B = toy_data(400)
train_A, val_A = loaders(X_A, Z_A, y_A)
train_B, val_B = loaders(X_B, Z_B, y_B)

# 1. Pretrain the backbone on A. BaseNetwork ignores Z.
base = covar_trainer(
    BaseNetwork,
    dict(backbone=Backbone, backbone_params=dict(in_features=10, out_features=8)),
    train_A, val_A, patience=12,
)

# 2. Refit the last layer on B: Z enters additively, with a ridge-penalised
#    backfit and an optional orthogonalisation of f_X against Z.
model = RefitCovarNetwork(base, num_covariates=1, orthogonalize=True)
model = model.fit(train_B, val_B)

fx = model.predict_fx(X_B, Z_B)   # DNN-input effect, orthogonal to Z
fz = model.predict_fz(Z_B)        # control-variable effect
```

`model.lam` holds the selected ridge penalty and `model.lambda_path_` the
full per-λ diagnostics (validation loss, convergence, coefficient norms) —
useful for checking the λ selection.

## Key concepts

- **Link functions.** `link="identity"` (Gaussian) and `link="logit"`
  (binary) are supported; the backfit reduces to a closed-form ridge solve
  for `"identity"` and runs IRLS otherwise. See `cocodeel/links.py`.
- **Centering.** `X`, `Z`, and `y` are centered before fitting so that
  `β₀` carries the population mean and the additive terms are
  identifiable; `RefitCovarNetwork.center_effects` fits these means on
  the refit sample. `predict_fx`/`predict_fz` stay zero-mean on that
  sample after fitting.
- **Orthogonalization.** With `orthogonalize=True`, a secondary linear
  regression removes the part of `f_X` that is linearly explained by `Z`,
  isolating the direct effect of `X` from the mediated effect that runs
  through `Z`. Without it, `f_X` and `f_Z` are still debiased for omitted
  variable bias, but a mediated effect of `Z` through `X` remains in `f_X`.
- **Cross-fitting.** The two-sample split above uses half the data for
  the backbone. `cocodeel.crossfit.CrossFitEnsemble` combines K fold-wise
  refits into the cross-fit ensemble (Chernozhukov et al., 2018) that
  recovers the rest.
- **Benchmarks, not the method.** `cocodeel.benchmarking` holds
  competitor baselines — `CovarNetwork` (end-to-end NAM-style joint
  training of `f_X` and `f_Z`), `PostHocOrthNetwork`, and the CF-Net
  adversarial trainer — used to show where joint training fails under
  concurvity. None of
  them is exported from `cocodeel` directly.

## Reproducing the paper's experiments

- `experiments/simulation/` — the simulation study (paper Section
  5): a synthetic image DGP, one self-contained script per study, and the
  R scripts that produce every simulation figure. See its `README.md`.
- `experiments/ukbb/` — the UK Biobank application (paper Section 6):
  synthetic confounding on real T1 structural MRI. See its `README.md`.
  UK Biobank data is access-controlled and not included here.

## Testing

```
pytest tests/
```

Tests pin the ridge solve against closed-form OLS/ridge solutions and
check the sample-split recovery of known effects — see `tests/`.
