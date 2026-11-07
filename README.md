# cocodeel

Control variables for deep neural networks: refit the last layer of a trained DNN as an additive model in the network's features and a control variable.

Code for *Controlling for Omitted Variable Bias in Deep Neural Networks*.

A DNN trained to predict `Y` from an input `X` (an image) also learns from any variable `Z` that is correlated with both: age, sex, the scanner site. It attributes the effect of `Z` to `X`. This is shortcut learning, and in regression terms it is omitted variable bias. The remedy from regression analysis is to include `Z` as a control variable. `cocodeel` does this for a trained DNN with backbone `φ`. It keeps `φ` fixed and refits the last layer as

```
g(E[Y | X, Z]) = η(X, Z) = β₀ + f_X(X) + f_Z(Z),    f_X(X) = φ(X)ᵀ β_X,    f_Z(Z) = Zᵀ β_Z,
```

by ridge-penalised backfitting, with IRLS for a non-identity link `g` and the ridge penalty `λ` chosen on a validation split. `f_X` is the direct effect of `X` with `Z` held fixed. The backbone and the refit must use disjoint samples, or the refit inherits the backbone's bias. Cross-fitting over `K` folds (Chernozhukov et al., 2018) recovers the full sample size. With `orthogonalize=True`, a post-hoc regression additionally removes from `f_X` the part that is linear in `Z`, giving `f_X^re`.

## Install

```
pip install -e .
```

The package needs `torch`, `numpy`, `scipy` and `scikit-learn`. `environment.yml` is the conda environment of the experiments, including R for the figures.

## Usage

`RefitCovarNetwork` is a `torch.nn.Module`. It takes any trained `BaseNetwork`, whose backbone is a module with an `out_features` attribute.

```python
import torch
from torch.utils.data import DataLoader

from cocodeel.dataset import CovarDataset
from cocodeel.model import BaseNetwork
from cocodeel.refit_model import RefitCovarNetwork
from cocodeel.crossfit import CrossFitEnsemble
from cocodeel.trainer import covar_trainer


class Backbone(torch.nn.Module):          # phi: X -> R^q
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(in_features, 32), torch.nn.ReLU(),
                                       torch.nn.Linear(32, out_features), torch.nn.ReLU())
        self.out_features = out_features

    def forward(self, x):
        return self.net(x)


def sample(n):                            # Z drives both X and y: y = 2 X_1 + 3 Z + noise
    Z = torch.randn(n, 1)
    X = torch.randn(n, 10) + Z
    y = 2.0 * X[:, :1] + 3.0 * Z + 0.5 * torch.randn(n, 1)
    return X, Z, y


def loaders(n):
    ds_tr, ds_va = CovarDataset(*sample(n)), CovarDataset(*sample(n // 4))
    return DataLoader(ds_tr, batch_size=64, shuffle=True), DataLoader(ds_va, batch_size=256)


# 1. Train a DNN without controls on each of two disjoint samples A and B.
(train_A, val_A), (train_B, val_B) = loaders(2000), loaders(2000)
backbone = dict(backbone=Backbone, backbone_params=dict(in_features=10, out_features=8))
dnn_A = covar_trainer(BaseNetwork, backbone, train_A, val_A).center_effects(train_A)
dnn_B = covar_trainer(BaseNetwork, backbone, train_B, val_B).center_effects(train_B)

# 2. Refit each last layer with Z as a control variable on the other sample.
refit_A = RefitCovarNetwork(dnn_A, num_covariates=1).fit(train_B, val_B)
refit_B = RefitCovarNetwork(dnn_B, num_covariates=1).fit(train_A, val_A)

# 3. Cross-fit (K=2): average the two refits.
model = CrossFitEnsemble([refit_A, refit_B]).recenter(train_A)

X, Z, _ = sample(500)
with torch.no_grad():
    f_x = model.predict_fx(X, Z)          # direct effect of X
    f_z = model.predict_fz(Z)             # effect of the control variable (true: 3 Z)
    y_hat = model(X, Z)                   # g^{-1}(η)
```

For a binary outcome, pass `link="logit"` in the model dictionary and `loss_fn=torch.nn.BCELoss()` to `covar_trainer`. A single sample split is one `RefitCovarNetwork`, without the ensemble.

## Experiments

The experiments are scripts that run the method end to end, one per question. Each script opens with what it shows and which figures it produces; the R scripts in `figures/` draw them.

| Script | Shows | Paper figures |
|---|---|---|
| `experiments/simulation/study_a_linear_consistency.py` | The DNN with Controls recovers `f_X` as the confounding strength, backbone width, number of controls and sample size grow, while the DNN without controls absorbs `f_Z` | Fig 1, Fig 4, appendix |
| `experiments/simulation/study_b_misspecification.py` | A nonlinear `f_Z` refit as linear in `Z` | appendix |
| `experiments/simulation/study_c_concurvity_benchmark.py` | The DNN with Controls against neural additive models and post-hoc methods under concurvity | Fig 2, appendix |
| `experiments/simulation/study_d_noisy_z.py` | A control variable measured with noise | appendix |
| `experiments/ukbb/` | Brain MRI from the UK Biobank with synthetic confounding by age | Fig 5, appendix |

The simulations run from the repository root, for example `NSIM=5 python experiments/simulation/study_a_linear_consistency.py`, and `python experiments/simulation/aggregate.py` reduces their predictions to the CSVs in `experiments/simulation/output/` that the figure scripts read. The UKBB data cannot be redistributed. `data/ukbb/make_dummy_data.py` writes random files with the same layout, so the UKBB pipeline runs end to end without access; `UKBB_DATA_ROOT` points it at the real data.

## Citation

```bibtex
@article{pfeuffer2026controlling,
  title  = {Controlling for Omitted Variable Bias in Deep Neural Networks},
  author = {Pfeuffer, Manuel},
  year   = {2026}
}
```
