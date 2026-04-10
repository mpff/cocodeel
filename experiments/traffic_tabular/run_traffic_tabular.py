#!/usr/bin/env python
"""Tabular mirror of the cocodeel traffic_light DGP: do baseline NN and PostHoc
converge in MSPE and Corr(fx, Z) as N grows?

The DGP mirrors simulate_traffic_light_data (cocodeel/experiments/simulation_images)
but replaces the 20×60 image construction with a 3-dimensional tabular X = (v1, v2, v3),
and uses an MLP backbone instead of a CNN.

  v1 = (1 - cv1) · U[0,1] + cv1 · Z         (decoy: confounded, no signal)
  v2 = (1 - cv2) · U[0,1] + cv2 · Z         (signal + partial confound)
  v3 ~ U[0,1]                                (pure signal, Z-independent)
  fx = b2 · (v2 - 0.5) + b3 · (v3 - 0.5)    (population-centred)
  fz = bz · (Z - 0.5)                        (population-centred)
  y  = fx + fz + N(0, sdy²)

Scientific question:
  For fixed cv1=0.8, cv2=0.5, how do MSPE(y), MSPE(fx), MSPE(fz), and
  Corr(fx_hat, Z) converge with N for
    (a) the baseline DNN without covariates (BaseNetwork), and
    (b) the PostHoc refit (PostHocCovarNetwork)?

Analytic reference:
  True Corr(fx, Z) = b2·cv2 / sqrt(b2²·[(1-cv2)² + cv2²] + b3²)
                   = 0.4082 at b2 = b3 = 1, cv2 = 0.5.

Centring protocol (important):
  - DGP produces POPULATION-centred fx_true, fz_true (we subtract the
    population mean 0.5 from v2, v3, Z by construction). fx_true and fz_true
    are never re-centred — they are the evaluation target as-is.
  - Models are TRAIN-centred via `center_effects(train_loader)` called after
    training. `predict_fx`/`predict_fz` then return train-data-centred values,
    which is the best estimate of the population-centred quantity using only
    training information.
  - MSPE is `((pred - true)² ).mean()` with NO further centring. Any residual
    offset between pred and true on the test set is legitimate bias to be
    measured (e.g. OVB contamination in the baseline DNN), not noise to be
    removed.

Usage:
    python -m experiments.traffic_tabular.run_traffic_tabular \
        --n-seeds 5 --N-values 100 200 400 800 1600
"""
import argparse
import csv
import datetime
import json
import multiprocessing as mp
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cocodeel.dataset import CovarDataset
from cocodeel.model import BaseNetwork, CovarNetwork
from cocodeel.posthoc_model import PostHocCovarNetwork
from cocodeel.trainer import covar_trainer


# ═══════════════════════════════════════════════════════════════════════════════
# DGP
# ═══════════════════════════════════════════════════════════════════════════════
def make_traffic_tabular(N, cv1=0.8, cv2=0.5, b2=1.0, b3=1.0, bz=1.0,
                         sdy=1.0, seed=0):
    """Tabular mirror of simulate_traffic_light_data (p=1 covariate case).

    Returns:
        X:  (N, 3) stack of v1, v2, v3.
        Z:  (N, 1) covariate (Uniform[0,1]).
        y:  (N, 1) outcome = fx + fz + Gaussian noise.
        fx: (N, 1) population-centred image effect.
        fz: (N, 1) population-centred covariate effect.
    """
    rng = np.random.default_rng(seed)
    Z = torch.tensor(rng.uniform(0, 1, (N, 1)), dtype=torch.float32)
    v1_raw = torch.tensor(rng.uniform(0, 1, (N, 1)), dtype=torch.float32)
    v2_raw = torch.tensor(rng.uniform(0, 1, (N, 1)), dtype=torch.float32)
    v3 = torch.tensor(rng.uniform(0, 1, (N, 1)), dtype=torch.float32)

    v1 = (1 - cv1) * v1_raw + cv1 * Z
    v2 = (1 - cv2) * v2_raw + cv2 * Z

    X = torch.cat([v1, v2, v3], dim=1)          # (N, 3)
    fx = b2 * (v2 - 0.5) + b3 * (v3 - 0.5)      # E[v2] = E[v3] = 0.5 → E[fx] = 0
    fz = bz * (Z - 0.5)                          # E[Z] = 0.5 → E[fz] = 0
    y = fx + fz + sdy * torch.tensor(rng.standard_normal((N, 1)),
                                     dtype=torch.float32)
    return X, Z, y, fx, fz


def true_corr_fx_z(b2, b3, cv2):
    """Closed-form Corr(fx, Z) for the traffic_light DGP.

    Derivation: Cov(fx, Z) = b2·cv2·σ_Z²; Var(fx) = σ_Z²·(b2²·[(1-cv2)² + cv2²] + b3²);
    σ_Z cancels.
    """
    return b2 * cv2 / np.sqrt(b2**2 * ((1-cv2)**2 + cv2**2) + b3**2)


def _ols_limit_beta(b2, b3, bz, cv1, cv2):
    """Population β̂_OLS for y ~ v1 + v2 + v3 under the traffic_tabular DGP.

    Returns (beta, Sigma_X_scaled) where quantities are in units of σ_Z² = 1/12
    (the common variance of U[0,1]). The 1/12 factor cancels in correlations but
    must be reinstated for MSPEs.
    """
    A = (1 - cv1)**2 + cv1**2
    B = (1 - cv2)**2 + cv2**2
    C = cv1 * cv2
    Sigma_X = np.array([[A, C, 0.0],
                        [C, B, 0.0],
                        [0.0, 0.0, 1.0]])
    Sigma_Xy = np.array([
        cv1 * (b2 * cv2 + bz),   # v1 has no signal loading, only OVB
        b2 * B + bz * cv2,
        b3,
    ])
    beta = np.linalg.solve(Sigma_X, Sigma_Xy)
    return beta, Sigma_X


def baseline_corr_ols_limit(b2, b3, bz, cv1, cv2):
    """Asymptotic Corr(f_hat_base, Z) for an OLS baseline y ~ v1 + v2 + v3.

    The MLP baseline (trained without Z) converges in the large-sample limit to
    the Bayes-optimal predictor E[y | v1, v2, v3]. For this DGP that conditional
    expectation is essentially linear in (v1, v2, v3), so the MLP converges to
    the OLS solution on X = (v1, v2, v3).

    Formula (OVB in an OLS linear model):
        β̂_OLS = Σ_X⁻¹ Σ_Xy  (population)
        Corr(β̂'X, Z) = β̂'Σ_XZ / sqrt(β̂'Σ_X β̂ · Var(Z))

    The σ_Z² = 1/12 factor cancels in the correlation.
    """
    beta, Sigma_X = _ols_limit_beta(b2, b3, bz, cv1, cv2)
    Sigma_XZ = np.array([cv1, cv2, 0.0])
    cov_pred_Z = beta @ Sigma_XZ
    var_pred   = beta @ Sigma_X @ beta
    return float(cov_pred_Z / np.sqrt(var_pred * 1.0))


def baseline_mspe_fx_ols_limit(b2, b3, bz, cv1, cv2):
    """Asymptotic MSPE(f_hat_base, f_x) for an OLS baseline y ~ v1 + v2 + v3.

    Derivation: in the limit, f_hat_base(X) = β̂_OLS' (X - E[X]). The true
    population-centred f_x is β_true' (X - E[X]) with
        β_true = (0, b2, b3)
    because v1 has no signal loading, only v2 and v3 do.

    Then
        MSPE(f_hat_base) = E[(β̂ - β_true)' (X - E[X]))²]
                         = (β̂ - β_true)' Σ_X (β̂ - β_true)

    i.e. a quadratic form in the bias vector.  The overall 1/12 factor (from
    Var(U[0,1])) is reinstated here.
    """
    beta_ols, Sigma_X_scaled = _ols_limit_beta(b2, b3, bz, cv1, cv2)
    beta_true = np.array([0.0, b2, b3])
    bias = beta_ols - beta_true
    return float((bias @ Sigma_X_scaled @ bias) / 12.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Backbones: MLP (the method under test) and Linear (the oracle)
# ═══════════════════════════════════════════════════════════════════════════════
class TabularMLP(nn.Module):
    """Single-hidden-layer MLP. Over-parameterised relative to the
    three-dimensional input so the backbone still has enough capacity to mix
    (v1, v2, v3) non-linearly, but the simpler architecture is a sanity check:
    if the earlier two-hidden-layer failure mode persists here, it is not an
    artefact of depth.
    """
    def __init__(self, in_features=3, hidden=64, out_features=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(),
            nn.Linear(hidden, out_features),
        )
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        return self.net(x)


class LinearBackbone(nn.Module):
    """Identity pass-through: H = X. No parameters.

    Use with PostHocCovarNetwork to get the linear-model oracle — ridge
    regression of y on (v1, v2, v3, Z). Because there is no non-linear mixing,
    H is exogenous wrt (y − fz) and FWL recovers fx consistently: this is the
    best a linear estimator can do and serves as the "correct model class"
    reference for the traffic_tabular DGP.
    """
    def __init__(self, in_features, out_features=None):
        super().__init__()
        out_features = in_features if out_features is None else out_features
        assert out_features == in_features, \
            f"LinearBackbone requires in_features==out_features, got {in_features}, {out_features}"
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# Custom trainer for NAM with preconditioned lr_fz
# ═══════════════════════════════════════════════════════════════════════════════
def train_nam_precond(mlp_params, train_loader, val_loader, Z_train,
                      lr_fx=3e-3, device='cpu', epochs=500, patience=20):
    """NAM (CovarNetwork + MLP) with data-driven lr for the fz parameter.

    Sets lr_fz to the theoretically optimal Newton step for the linear fz
    subproblem:

        lr_fz = 1 / λ_max(Z_c^T Z_c / N)

    where Z_c is the centred Z. For a 1-D U[0, 1] covariate this collapses
    to 1 / Var(Z) ≈ 12. The fx learning rate is the usual HP-search value.
    Uses a CosineAnnealingLR schedule with early stopping.

    covar_trainer doesn't support per-parameter-group learning rates, so the
    training loop is open-coded here.
    """
    import copy

    model = CovarNetwork(
        backbone=mlp_params['backbone'],
        backbone_params=mlp_params['backbone_params'],
        num_covariates=1, link='identity',
    ).to(device)

    # ---- data-driven lr_fz from the curvature of the linear subproblem ----
    Z = Z_train.to(device)
    Zc = Z - Z.mean(dim=0, keepdim=True)
    A = (Zc.T @ Zc) / max(1, Zc.shape[0])
    lam_max = float(torch.linalg.eigvalsh(A).max())
    lr_fz = 1.0 / max(lam_max, 1e-12)

    opt = torch.optim.Adam([
        {'params': [*model.backbone.parameters(), *model.fx.parameters()],
         'lr': lr_fx, 'weight_decay': 1e-4},
        {'params': [*model.fz.parameters(), model.intercept],
         'lr': lr_fz, 'weight_decay': 0.0},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    best_val = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    val_losses = []
    pctr = 0

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x = batch['X'].to(device)
            z = batch['Z'].to(device)
            y = batch['y'].to(device)
            opt.zero_grad()
            loss = loss_fn(model(x, z), y)
            loss.backward()
            opt.step()
        scheduler.step()

        model.eval()
        v_sum, v_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['X'].to(device)
                z = batch['Z'].to(device)
                y = batch['y'].to(device)
                v_sum += loss_fn(model(x, z), y).item() * y.size(0)
                v_n += y.size(0)
        v_loss = v_sum / max(1, v_n)
        val_losses.append(v_loss)

        if v_loss < best_val:
            best_val = v_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            pctr = 0
        else:
            pctr += 1
            if pctr >= patience:
                break

    model.load_state_dict(best_state)
    model.val_losses_ = val_losses
    model.best_epoch_ = best_epoch
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Closed-form OLS oracles (no ridge, no IRLS, no iteration)
# ═══════════════════════════════════════════════════════════════════════════════
def train_oracle_ols(X_tr, Z_tr, y_tr, device='cpu'):
    """Oracle = OLS of y on [v1, v2, v3, Z, 1].

    Closed-form: beta = (A'A)^{-1} A'y with A = [X, Z, 1]. Wraps the result
    in a CovarNetwork(LinearBackbone) so it plugs into evaluate().
    """
    N = X_tr.shape[0]
    A = torch.cat([X_tr, Z_tr, torch.ones(N, 1)], dim=1)  # (N, 5)
    beta = torch.linalg.lstsq(A, y_tr).solution          # (5, 1)
    b_X = beta[:3]                                        # (3, 1)
    b_Z = beta[3:4]                                       # (1, 1)
    # beta[4] (intercept) is implicit; OLS passes through (mean X, mean Z, mean y).

    model = CovarNetwork(
        backbone=LinearBackbone,
        backbone_params={'in_features': 3},
        num_covariates=1, link='identity',
    ).to(device)
    model.fx.weight.data.copy_(b_X.T)        # Linear weight shape (out, in) = (1, 3)
    model.fz.weight.data.copy_(b_Z.T)        # (1, 1)
    model.center_x.mean.copy_(X_tr.mean(dim=0).to(device))
    model.center_z.mean.copy_(Z_tr.mean(dim=0).to(device))
    model.intercept.data.fill_(float(y_tr.mean()))  # OLS passes through the mean
    model.is_centered.data = torch.tensor(True)
    return model


def train_oracle_ovb_ols(X_tr, y_tr, device='cpu'):
    """Oracle OVB = OLS of y on [v1, v2, v3, 1] (no Z).

    Closed-form. Wraps in BaseNetwork(LinearBackbone).
    """
    N = X_tr.shape[0]
    A = torch.cat([X_tr, torch.ones(N, 1)], dim=1)  # (N, 4)
    beta = torch.linalg.lstsq(A, y_tr).solution    # (4, 1)
    b_X = beta[:3]

    model = BaseNetwork(
        backbone=LinearBackbone,
        backbone_params={'in_features': 3},
        num_covariates=0, link='identity',
    ).to(device)
    model.fx.weight.data.copy_(b_X.T)
    model.center_x.mean.copy_(X_tr.mean(dim=0).to(device))
    model.intercept.data.fill_(float(y_tr.mean()))
    model.is_centered.data = torch.tensor(True)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-fitted PostHoc (2-fold)
# ═══════════════════════════════════════════════════════════════════════════════
class EnsemblePostHoc(nn.Module):
    """Averages two PostHocCovarNetwork instances from 2-fold cross-fitting.

    Each instance has its own backbone trained on one fold and PostHoc fit
    performed on the OTHER fold. Predictions are the mean of the two.
    """
    def __init__(self, phm_a, phm_b):
        super().__init__()
        self.phm_a = phm_a
        self.phm_b = phm_b
        self.num_covariates = phm_a.num_covariates
        self.orthogonalize = getattr(phm_a, 'orthogonalize', False)
        # Average fz.weight scalar across folds for b_z_hat tracking
        self.b_z_hat_ = 0.5 * (
            float(phm_a.fz.weight.data.flatten()[0])
            + float(phm_b.fz.weight.data.flatten()[0]))

    def predict_fx(self, x, z=None):
        if self.orthogonalize:
            fx_a = self.phm_a.predict_fx(x, z)
            fx_b = self.phm_b.predict_fx(x, z)
        else:
            fx_a = self.phm_a.predict_fx(x)
            fx_b = self.phm_b.predict_fx(x)
        return 0.5 * (fx_a + fx_b)

    def predict_fz(self, z):
        return 0.5 * (self.phm_a.predict_fz(z) + self.phm_b.predict_fz(z))

    def forward(self, x, z):
        return 0.5 * (self.phm_a(x, z) + self.phm_b(x, z))


def train_posthoc_cf(mlp_params, X_tr, Z_tr, y_tr, lr, scheduler,
                     scheduler_kwargs, device, orthogonalize=False,
                     epochs=500, patience=20):
    """2-fold cross-fitted PostHoc.

    Fold A: train backbone on X_tr[:half], PostHoc refit on X_tr[half:].
    Fold B: train backbone on X_tr[half:], PostHoc refit on X_tr[:half].
    Ensemble both via EnsemblePostHoc.
    """
    N = X_tr.shape[0]
    half = N // 2
    bs = min(64, max(8, N // 8))

    def _loaders(X, Z, y):
        n = X.shape[0]; h = n // 2
        tr_ds = CovarDataset(X[:h], Z[:h], y[:h])
        va_ds = CovarDataset(X[h:], Z[h:], y[h:])
        return (DataLoader(tr_ds, batch_size=bs, shuffle=True),
                DataLoader(va_ds, batch_size=bs, shuffle=False))

    def _fit_fold(tr_X, tr_Z, tr_y, re_X, re_Z, re_y):
        tr_ld, va_ld = _loaders(tr_X, tr_Z, tr_y)
        base = covar_trainer(
            model=BaseNetwork, model_params=mlp_params,
            train_loader=tr_ld, val_loader=va_ld,
            epochs=epochs, lr=lr, patience=patience,
            scheduler=scheduler, scheduler_kwargs=scheduler_kwargs,
            device=device,
        ).center_effects(tr_ld)
        re_tr_ld, re_va_ld = _loaders(re_X, re_Z, re_y)
        phm = PostHocCovarNetwork(base, num_covariates=1,
                                  orthogonalize=orthogonalize).to(device)
        phm.fit(re_tr_ld, re_va_ld, n_lambdas=20)
        return phm

    phm_ab = _fit_fold(X_tr[:half], Z_tr[:half], y_tr[:half],
                       X_tr[half:], Z_tr[half:], y_tr[half:])
    phm_ba = _fit_fold(X_tr[half:], Z_tr[half:], y_tr[half:],
                       X_tr[:half], Z_tr[:half], y_tr[:half])
    return EnsemblePostHoc(phm_ab, phm_ba)


# ═══════════════════════════════════════════════════════════════════════════════
# Double Machine Learning (Robinson partialling-out)
# ═══════════════════════════════════════════════════════════════════════════════
def train_dml(mlp_params, X_tr, Z_tr, y_tr, train_loader, val_loader,
              lr, scheduler, scheduler_kwargs, device='cpu',
              epochs=500, patience=20):
    """Robinson/DML for the partially linear model y = f(X) + Z*b_z + eps.

    2-fold cross-fitting for bias-free b_z_hat:
      - Fold A: train m_hat, l_hat on first half, residualise second half
      - Fold B: train m_hat, l_hat on second half, residualise first half
      - Pool out-of-fold residuals and run scalar OLS for b_z_hat
      - Stage 3: train f_hat(X) on y - Z*b_z_hat using all data

    Returns a CovarNetwork wrapping the stage-3 backbone+fx and b_z_hat,
    with the intercept adjusted so forward(X, Z) predicts y (not y').
    """
    N = X_tr.shape[0]
    half = N // 2
    bs = min(64, max(8, N // 8))

    def _fold_loaders(X, Z, target):
        """Build 50/50 train/val loaders within a fold. target is the y-slot."""
        n = X.shape[0]
        h = n // 2
        tr_ds = CovarDataset(X[:h], Z[:h], target[:h])
        va_ds = CovarDataset(X[h:], Z[h:], target[h:])
        return (DataLoader(tr_ds, batch_size=bs, shuffle=True),
                DataLoader(va_ds, batch_size=bs, shuffle=False))

    def _train_nuisance(train_ld, val_ld):
        return covar_trainer(
            model=BaseNetwork, model_params=mlp_params,
            train_loader=train_ld, val_loader=val_ld,
            epochs=epochs, lr=lr, patience=patience,
            scheduler=scheduler, scheduler_kwargs=scheduler_kwargs,
            device=device,
        )

    # 2-fold cross-fitting: each sample is in nuisance-training for one fold
    # and in residual-eval for the other. Pool out-of-fold residuals.
    folds = [(slice(0, half),  slice(half, N)),
             (slice(half, N),  slice(0, half))]
    tilde_y_list, tilde_z_list = [], []
    for train_idx, eval_idx in folds:
        # Nuisance training on this fold's data (internal 50/50 split for ES).
        X_f, Z_f, y_f = X_tr[train_idx], Z_tr[train_idx], y_tr[train_idx]
        m_tr_ld, m_va_ld = _fold_loaders(X_f, Z_f, y_f)
        l_tr_ld, l_va_ld = _fold_loaders(X_f, Z_f, Z_f)
        m_model_f = _train_nuisance(m_tr_ld, m_va_ld)
        l_model_f = _train_nuisance(l_tr_ld, l_va_ld)

        # Out-of-fold residuals on the OTHER fold.
        with torch.no_grad():
            X_e = X_tr[eval_idx].to(device)
            Z_e = Z_tr[eval_idx].to(device)
            y_e = y_tr[eval_idx].to(device)
            m_hat = m_model_f(X_e)
            l_hat = l_model_f(X_e)
            tilde_y_list.append((y_e - m_hat).squeeze(-1))
            tilde_z_list.append((Z_e - l_hat).squeeze(-1))

    # Stage 2: scalar OLS on pooled out-of-fold residuals
    tilde_y = torch.cat(tilde_y_list)
    tilde_z = torch.cat(tilde_z_list)
    b_z_hat = float((tilde_z * tilde_y).sum() / ((tilde_z ** 2).sum() + 1e-12))

    # Stage 3: f_hat(X) on y' = y - Z*b_z_hat, trained on all data (50/50 ES split)
    y_prime = y_tr - Z_tr * b_z_hat
    f_train_ds = CovarDataset(X_tr[:half], Z_tr[:half], y_prime[:half])
    f_val_ds   = CovarDataset(X_tr[half:], Z_tr[half:], y_prime[half:])
    f_train_loader = DataLoader(f_train_ds, batch_size=bs, shuffle=True)
    f_val_loader   = DataLoader(f_val_ds,   batch_size=bs, shuffle=False)
    f_model = covar_trainer(
        model=BaseNetwork, model_params=mlp_params,
        train_loader=f_train_loader, val_loader=f_val_loader,
        epochs=epochs, lr=lr, patience=patience,
        scheduler=scheduler, scheduler_kwargs=scheduler_kwargs,
        device=device,
    ).center_effects(f_train_loader)

    # Stage 4: assemble a CovarNetwork reusing stage-3 weights + manual fz
    dml = CovarNetwork(
        backbone=mlp_params['backbone'],
        backbone_params=mlp_params['backbone_params'],
        num_covariates=1, link='identity',
    ).to(device)
    # Copy stage-3 backbone + fx + center_x + center_y (features and y' mean).
    dml.backbone.load_state_dict(f_model.backbone.state_dict())
    dml.fx.weight.data.copy_(f_model.fx.weight.data)
    dml.center_x.mean.copy_(f_model.center_x.mean)
    dml.center_y.mean.copy_(f_model.center_y.mean)
    # Stage-3 intercept absorbs E[y'] = E[y] - E[Z]*b_z_hat. The CovarNetwork's
    # predict_fz already subtracts E[Z] from Z, so we need the intercept to
    # equal E[y], not E[y']. Add E[Z_train]*b_z_hat back.
    z_train_mean = Z_tr.mean(dim=0).to(device)  # shape (num_covariates,)
    dml.intercept.data.copy_(f_model.intercept.data + (z_train_mean * b_z_hat).sum())
    # Set fz weights and its centering mean.
    dml.fz.weight.data.fill_(float(b_z_hat))
    dml.center_z.mean.copy_(z_train_mean)
    dml.is_centered.data = torch.tensor(True)

    # Expose the stage-2 coefficient for the b_z_hat tracking.
    dml.b_z_hat_ = b_z_hat
    # For val_loss logging, reuse stage 3's.
    dml.val_losses_ = f_model.val_losses_
    dml.best_epoch_ = f_model.best_epoch_
    return dml


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ═══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def mspe(pred, true):
    """MSPE of model predictions against population-centred targets.

    NEVER re-centre on the test set. DGP fx_true/fz_true are population-centred
    by construction; model predictions are train-centred via
    `center_effects(train_loader)`. Any residual offset on the test set is
    legitimate bias (e.g. OVB contamination) and must appear in the metric.
    """
    return float(((pred - true) ** 2).mean())


@torch.no_grad()
def evaluate(model, X_te, Z_te, y_te, fx_te, fz_te):
    """Return {mspe_y, mspe_fx, mspe_fz, corr_fx_z, b_z_hat} on the test set.

    Moves test tensors to the model's device, moves predictions back to CPU
    before MSPE/correlation computation so the metric code stays device-agnostic.
    """
    model.eval()
    device = next(model.parameters()).device
    X_te_d = X_te.to(device)
    Z_te_d = Z_te.to(device)

    if getattr(model, 'num_covariates', 0) > 0:
        y_hat = model(X_te_d, Z_te_d).cpu()
    else:
        y_hat = model(X_te_d).cpu()
    # PostHocCovarNetwork with orthogonalize=True requires z in predict_fx.
    if getattr(model, 'orthogonalize', False):
        fx_hat = model.predict_fx(X_te_d, Z_te_d).squeeze().cpu()
    else:
        fx_hat = model.predict_fx(X_te_d).squeeze().cpu()
    fz_hat = (model.predict_fz(Z_te_d).squeeze().cpu()
              if hasattr(model, 'predict_fz') else torch.zeros_like(fx_hat))

    # Extract b_z_hat: scalar coefficient on Z. Methods without Z report NaN.
    if hasattr(model, 'b_z_hat_'):                       # DML stashed it
        b_z_hat = float(model.b_z_hat_)
    elif getattr(model, 'num_covariates', 0) > 0 and hasattr(model, 'fz'):
        b_z_hat = float(model.fz.weight.data.flatten()[0])
    else:
        b_z_hat = float('nan')

    return {
        'mspe_y':    mspe(y_hat.squeeze(), y_te.squeeze()),
        'mspe_fx':   mspe(fx_hat, fx_te.squeeze()),
        'mspe_fz':   mspe(fz_hat, fz_te.squeeze()),
        'corr_fx_z': float(np.corrcoef(fx_hat.numpy(),
                                       Z_te.squeeze().numpy())[0, 1]),
        'b_z_hat':   b_z_hat,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# One training run (baseline + posthoc) for a given (N, seed)
# ═══════════════════════════════════════════════════════════════════════════════
def _nan_metrics():
    """Placeholder metrics for methods that are skipped in a partial sweep."""
    return {'mspe_y': float('nan'), 'mspe_fx': float('nan'),
            'mspe_fz': float('nan'), 'corr_fx_z': float('nan'),
            'b_z_hat': float('nan'), 'val_loss': float('nan')}


def run_one(N_train, seed, cv1=0.8, cv2=0.5, N_test=2000,
            hidden=64, out_features=32,
            lr=1e-3, scheduler=None, scheduler_kwargs=None,
            device='cpu', models_dir=None, methods=None):
    """Train selected methods on one (N, seed).

    Args:
        lr: learning rate for the backbone training step.
        scheduler: scheduler class (not instance) forwarded to covar_trainer.
            None triggers covar_trainer's default ReduceLROnPlateau.
        scheduler_kwargs: kwargs dict for the scheduler class.
        device: 'cpu' or 'cuda'. Passed through to covar_trainer; evaluate()
            moves test tensors to the model's device automatically.
        models_dir: if not None, save each fitted model to
            ``{models_dir}/{method}_N{N}_seed{seed}.pt``.
        methods: iterable of method names to run. If None, run all METHODS.
            Skipped methods return NaN placeholder metrics.

    Returns:
        dict method → metrics dict (with 'val_loss' field)
    """
    torch.manual_seed(seed)
    methods = set(methods) if methods is not None else set(METHODS)

    # Generate train and a fixed test set (different seed).
    X_tr, Z_tr, y_tr, _, _ = make_traffic_tabular(N_train, cv1=cv1, cv2=cv2, seed=seed)
    X_te, Z_te, y_te, fx_te, fz_te = make_traffic_tabular(
        N_test, cv1=cv1, cv2=cv2, seed=9999)

    # 50/50 train/val split inside the training data — covar_trainer's early
    # stopping uses the val loader.
    half = N_train // 2
    train_ds = CovarDataset(X_tr[:half], Z_tr[:half], y_tr[:half])
    val_ds   = CovarDataset(X_tr[half:], Z_tr[half:], y_tr[half:])
    bs = min(64, max(8, N_train // 8))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False)

    mlp_params = {
        'backbone': TabularMLP,
        'backbone_params': {'in_features': 3, 'hidden': hidden,
                            'out_features': out_features},
    }

    # ── Baseline: BaseNetwork with MLP trained without covariates ──
    needs_base = bool(methods & {'baseline', 'posthoc', 'posthoc_orth'})
    if needs_base:
        base_model = covar_trainer(
            model=BaseNetwork, model_params=mlp_params,
            train_loader=train_loader, val_loader=val_loader,
            epochs=500, lr=lr, patience=20,
            scheduler=scheduler, scheduler_kwargs=scheduler_kwargs,
            device=device,
        ).center_effects(train_loader)
        base_val_loss = float(base_model.val_losses_[base_model.best_epoch_])
        base_metrics = (evaluate(base_model, X_te, Z_te, y_te, fx_te, fz_te)
                        if 'baseline' in methods else _nan_metrics())
    else:
        base_model = None
        base_val_loss = float('nan')
        base_metrics = _nan_metrics()

    # ── NAM: CovarNetwork (MLP + Z), trained end-to-end, single Adam, same lr ──
    if 'nam' in methods:
        nam_model = covar_trainer(
            model=CovarNetwork, model_params=mlp_params,
            train_loader=train_loader, val_loader=val_loader,
            epochs=500, lr=lr, patience=20,
            scheduler=scheduler, scheduler_kwargs=scheduler_kwargs,
            device=device,
        ).center_effects(train_loader)
        nam_metrics = evaluate(nam_model, X_te, Z_te, y_te, fx_te, fz_te)
        nam_val_loss = float(nam_model.val_losses_[nam_model.best_epoch_])
    else:
        nam_model = None
        nam_metrics = _nan_metrics()
        nam_val_loss = float('nan')

    # ── NAM precond + optional refit ──
    needs_nam_precond = bool(methods & {'nam_precond', 'nam_precond_refit'})
    if needs_nam_precond:
        nam_precond_model = train_nam_precond(
            mlp_params=mlp_params,
            train_loader=train_loader, val_loader=val_loader,
            Z_train=Z_tr[:half],
            lr_fx=lr, device=device,
            epochs=500, patience=20,
        )
        nam_precond_model.center_effects(train_loader)
        nam_precond_val_loss = float(
            nam_precond_model.val_losses_[nam_precond_model.best_epoch_])
        nam_precond_metrics = (
            evaluate(nam_precond_model, X_te, Z_te, y_te, fx_te, fz_te)
            if 'nam_precond' in methods else _nan_metrics())
    else:
        nam_precond_model = None
        nam_precond_metrics = _nan_metrics()
        nam_precond_val_loss = float('nan')

    if 'nam_precond_refit' in methods:
        refit_base = BaseNetwork(
            backbone=TabularMLP,
            backbone_params=mlp_params['backbone_params'],
            num_covariates=0, link='identity',
        ).to(device)
        refit_base.backbone.load_state_dict(nam_precond_model.backbone.state_dict())
        nam_precond_refit_model = PostHocCovarNetwork(
            refit_base, num_covariates=1).to(device)
        nam_precond_refit_model.fit(train_loader, val_loader, n_lambdas=20)
        nam_precond_refit_metrics = evaluate(
            nam_precond_refit_model, X_te, Z_te, y_te, fx_te, fz_te)
    else:
        nam_precond_refit_model = None
        nam_precond_refit_metrics = _nan_metrics()

    # ── PostHoc (same-sample) ──
    if 'posthoc' in methods:
        phm = PostHocCovarNetwork(base_model, num_covariates=1).to(device)
        phm.fit(train_loader, val_loader, n_lambdas=20)
        ph_metrics = evaluate(phm, X_te, Z_te, y_te, fx_te, fz_te)
    else:
        phm = None
        ph_metrics = _nan_metrics()

    # ── PostHoc + Orth (same-sample) ──
    if 'posthoc_orth' in methods:
        phm_orth = PostHocCovarNetwork(base_model, num_covariates=1,
                                       orthogonalize=True).to(device)
        phm_orth.fit(train_loader, val_loader, n_lambdas=20)
        ph_orth_metrics = evaluate(phm_orth, X_te, Z_te, y_te, fx_te, fz_te)
    else:
        phm_orth = None
        ph_orth_metrics = _nan_metrics()

    # ── PostHoc CF: 2-fold cross-fitted ──
    if 'posthoc_cf' in methods:
        phm_cf = train_posthoc_cf(
            mlp_params=mlp_params, X_tr=X_tr, Z_tr=Z_tr, y_tr=y_tr,
            lr=lr, scheduler=scheduler, scheduler_kwargs=scheduler_kwargs,
            device=device, orthogonalize=False, epochs=500, patience=20,
        )
        phm_cf_metrics = evaluate(phm_cf, X_te, Z_te, y_te, fx_te, fz_te)
    else:
        phm_cf = None
        phm_cf_metrics = _nan_metrics()

    # ── PostHoc + Orth CF ──
    if 'posthoc_orth_cf' in methods:
        phm_orth_cf = train_posthoc_cf(
            mlp_params=mlp_params, X_tr=X_tr, Z_tr=Z_tr, y_tr=y_tr,
            lr=lr, scheduler=scheduler, scheduler_kwargs=scheduler_kwargs,
            device=device, orthogonalize=True, epochs=500, patience=20,
        )
        phm_orth_cf_metrics = evaluate(phm_orth_cf, X_te, Z_te, y_te, fx_te, fz_te)
    else:
        phm_orth_cf = None
        phm_orth_cf_metrics = _nan_metrics()

    # ── Oracle (closed-form OLS on [v1, v2, v3, Z]) ──
    if 'oracle' in methods:
        oracle = train_oracle_ols(X_tr, Z_tr, y_tr, device=device)
        oracle_metrics = evaluate(oracle, X_te, Z_te, y_te, fx_te, fz_te)
    else:
        oracle = None
        oracle_metrics = _nan_metrics()

    # ── Oracle OVB (closed-form OLS on [v1, v2, v3], no Z) ──
    if 'oracle_ovb' in methods:
        oracle_ovb = train_oracle_ovb_ols(X_tr, y_tr, device=device)
        oracle_ovb_metrics = evaluate(oracle_ovb, X_te, Z_te, y_te, fx_te, fz_te)
        oracle_ovb_val_loss = float('nan')  # no iterative training
    else:
        oracle_ovb = None
        oracle_ovb_metrics = _nan_metrics()
        oracle_ovb_val_loss = float('nan')

    # ── DML (Robinson partialling-out, 2-fold cross-fit) ──
    if 'dml' in methods:
        dml_model = train_dml(
            mlp_params=mlp_params,
            X_tr=X_tr, Z_tr=Z_tr, y_tr=y_tr,
            train_loader=train_loader, val_loader=val_loader,
            lr=lr, scheduler=scheduler, scheduler_kwargs=scheduler_kwargs,
            device=device, epochs=500, patience=20,
        )
        dml_metrics = evaluate(dml_model, X_te, Z_te, y_te, fx_te, fz_te)
        dml_val_loss = float(dml_model.val_losses_[dml_model.best_epoch_])
    else:
        dml_model = None
        dml_metrics = _nan_metrics()
        dml_val_loss = float('nan')

    base_metrics['val_loss']              = base_val_loss
    nam_metrics['val_loss']               = nam_val_loss
    nam_precond_metrics['val_loss']       = nam_precond_val_loss
    nam_precond_refit_metrics['val_loss'] = nam_precond_val_loss  # reuses nam_precond training
    ph_metrics['val_loss']                = base_val_loss  # posthoc reuses base's training
    ph_orth_metrics['val_loss']           = base_val_loss  # same base_model
    phm_cf_metrics['val_loss']            = float('nan')   # ensembles two backbones
    phm_orth_cf_metrics['val_loss']       = float('nan')
    dml_metrics['val_loss']               = dml_val_loss
    oracle_metrics['val_loss']            = float('nan')   # no MLP training
    oracle_ovb_metrics['val_loss']        = oracle_ovb_val_loss

    # ── Persist fitted models for post-hoc re-plotting and diagnosis ──
    if models_dir is not None:
        fitted = {
            'baseline':          base_model,
            'nam':               nam_model,
            'nam_precond':       nam_precond_model,
            'nam_precond_refit': nam_precond_refit_model,
            'posthoc':           phm,
            'posthoc_cf':        phm_cf,
            'posthoc_orth':      phm_orth,
            'posthoc_orth_cf':   phm_orth_cf,
            'dml':               dml_model,
            'oracle':            oracle,
            'oracle_ovb':        oracle_ovb,
        }
        for name, m in fitted.items():
            if m is None:  # skipped in partial sweep
                continue
            fname = f'{name}_N{N_train}_seed{seed}.pt'
            torch.save(m.state_dict(), os.path.join(models_dir, fname))

    return {'baseline':          base_metrics,
            'nam':               nam_metrics,
            'nam_precond':       nam_precond_metrics,
            'nam_precond_refit': nam_precond_refit_metrics,
            'posthoc':           ph_metrics,
            'posthoc_cf':        phm_cf_metrics,
            'posthoc_orth':      ph_orth_metrics,
            'posthoc_orth_cf':   phm_orth_cf_metrics,
            'dml':               dml_metrics,
            'oracle':            oracle_metrics,
            'oracle_ovb':        oracle_ovb_metrics}


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregation + plotting
# ═══════════════════════════════════════════════════════════════════════════════
def aggregate(results, methods, metrics, N_values):
    """Stack per-(N, seed) dicts into mean±std arrays per (method, metric, N)."""
    agg = {m: {k: [] for k in metrics} for m in methods}
    for method in methods:
        for N in N_values:
            subset = [r for r in results[method] if r['N'] == N]
            for k in metrics:
                vals = [r[k] for r in subset]
                agg[method][k].append((float(np.mean(vals)), float(np.std(vals))))
    return agg


METHODS = ('baseline', 'nam', 'nam_precond', 'nam_precond_refit',
           'posthoc', 'posthoc_cf', 'posthoc_orth', 'posthoc_orth_cf',
           'dml',
           'oracle_ovb', 'oracle')
COLORS = {'baseline':           '#888888',
          'nam':                '#9467bd',
          'nam_precond':        '#17becf',
          'nam_precond_refit':  '#8c564b',
          'posthoc':            '#1f77b4',
          'posthoc_cf':         '#08306b',
          'posthoc_orth':       '#e377c2',
          'posthoc_orth_cf':    '#7a0177',
          'dml':                '#bcbd22',
          'oracle_ovb':         '#ff7f0e',
          'oracle':             '#2ca02c'}
LABELS = {'baseline':           'Baseline (MLP, no Z)',
          'nam':                'NAM (MLP + Z, joint)',
          'nam_precond':        r'NAM precond ($\eta_g = 1/\lambda_\mathrm{max}$)',
          'nam_precond_refit':  'NAM precond + PostHoc refit',
          'posthoc':            'PostHoc (MLP + FWL)',
          'posthoc_cf':         'PostHoc CF (2-fold)',
          'posthoc_orth':       r'PostHoc + Orth ($f_x^\mathrm{re}$)',
          'posthoc_orth_cf':    'PostHoc + Orth CF (2-fold)',
          'dml':                'DML (Robinson)',
          'oracle_ovb':         'Oracle OVB (Linear, no Z)',
          'oracle':             'Oracle (Linear + FWL)'}


def plot_convergence(agg, N_values, true_corr, baseline_ovb_corr,
                     baseline_ovb_mspe_fx, cv1, cv2, sdy, out_dir):
    """1×4 panel figure: MSPE(y), MSPE(fx), MSPE(fz), Corr(fx, Z) vs N.

    Reference lines (each entry is (value, label, color)):
      MSPE(y)    → sdy² (population noise variance, Bayes floor)
      MSPE(fx)   → 0 (target for a consistent estimator)
                   + baseline_ovb_mspe_fx (OVB asymptote for baseline MLP)
      MSPE(fz)   → 0
      Corr(fx,Z) → true_corr (Oracle target) + baseline_ovb_corr (OVB asymptote)

    X-axis ticks are set explicitly to the swept N values (no auto log-ticks).
    """
    metrics_cfg = [
        ('mspe_y',    r'MSPE($y$)',        True,  [(sdy**2,                fr'$\sigma^2 = {sdy**2:.2f}$', 'red')]),
        ('mspe_fx',   r'MSPE($f_x$)',      True,  [(0.0,                    None,            'red'),
                                                   (baseline_ovb_mspe_fx,   'Oracle OVB',    'orange')]),
        ('mspe_fz',   r'MSPE($f_z$)',      True,  [(0.0,                    None,            'red')]),
        ('corr_fx_z', r'Corr($f_x$, $Z$)', False, [(true_corr,              'Oracle Corr',   'red'),
                                                   (baseline_ovb_corr,      'Oracle OVB',    'orange')]),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    for ax, (metric, ylabel, log_y, hlines) in zip(axes, metrics_cfg):
        for method in METHODS:
            means = np.array([m for m, _ in agg[method][metric]])
            stds  = np.array([s for _, s in agg[method][metric]])
            # Clip the lower fill_between bound on log axes to avoid
            # log(negative) when std > mean at small N with high variance.
            lower = np.maximum(means - stds, means * 0.1) if log_y else means - stds
            upper = means + stds
            ax.plot(N_values, means, '-o', color=COLORS[method],
                    label=LABELS[method], linewidth=2, markersize=5)
            ax.fill_between(N_values, lower, upper,
                            color=COLORS[method], alpha=0.2)
        for hval, hlabel, hcolor in hlines:
            lbl = None if hlabel is None else f'{hlabel} ({hval:.3f})'
            ax.axhline(hval, color=hcolor, linestyle=':', linewidth=1.2, label=lbl)
        ax.set_xlabel(r'$N_\mathrm{train}$')
        ax.set_ylabel(ylabel)
        ax.set_xscale('log')
        ax.set_xticks(N_values)
        ax.set_xticklabels([str(n) for n in N_values], rotation=30, fontsize=8)
        ax.minorticks_off()  # suppress default minor log ticks between our explicit ones
        if log_y:
            ax.set_yscale('log')
            # Prevent matplotlib from auto-scaling the y-axis to absurd ranges
            # (e.g. 10^-15) when fill_between regions are narrow.
            if metric in ('mspe_fx', 'mspe_fz'):
                ax.set_ylim(bottom=1e-4)
            elif metric == 'mspe_y':
                ax.set_ylim(0.95, 1.30)
    axes[0].legend(fontsize=8, loc='upper right')
    fig.suptitle(f'Traffic tabular DGP (cv1={cv1}, cv2={cv2}): '
                 f'method comparison across N', fontsize=11)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'n_sweep.{ext}'),
                    bbox_inches='tight', dpi=150)
    print(f"Plot saved to {out_dir}/n_sweep.{{pdf,png}}")


# ═══════════════════════════════════════════════════════════════════════════════
# Run directory infrastructure (immutable, timestamped, self-contained)
# ═══════════════════════════════════════════════════════════════════════════════
class TeeStream:
    """Duplicate writes to two streams (e.g. stdout + a log file)."""
    def __init__(self, primary, secondary):
        self.primary = primary
        self.secondary = secondary

    def write(self, data):
        self.primary.write(data)
        self.secondary.write(data)
        self.secondary.flush()

    def flush(self):
        self.primary.flush()
        self.secondary.flush()


def _git_commit():
    """Return current git HEAD hash, or 'unknown' if not in a repo."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        out = subprocess.check_output(
            ['git', '-C', here, 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return 'unknown'


def setup_run_dir(out_root, run_name=None):
    """Create a timestamped run directory under ``out_root/runs/``.

    Returns the run_dir path. Never overwrites: if ``run_name`` collides it
    appends a suffix. Creates ``models/`` subdirectory.
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    base = timestamp if run_name is None else f'{timestamp}_{run_name}'
    runs_root = os.path.join(out_root, 'runs')
    os.makedirs(runs_root, exist_ok=True)
    run_dir = os.path.join(runs_root, base)
    suffix = 0
    while os.path.exists(run_dir):
        suffix += 1
        run_dir = os.path.join(runs_root, f'{base}_{suffix}')
    os.makedirs(run_dir)
    os.makedirs(os.path.join(run_dir, 'models'))
    return run_dir


def write_config_and_manifest(run_dir, config, start_time):
    """Write config.json (hyperparameters, DGP params, CLI args) and
    manifest.json (PID, host, git commit, start time)."""
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    manifest = {
        'pid':        os.getpid(),
        'host':       socket.gethostname(),
        'git_commit': _git_commit(),
        'start_time': start_time,
        'run_dir':    run_dir,
        'argv':       sys.argv,
    }
    with open(os.path.join(run_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    return manifest


def finalize_manifest(run_dir, status, total_seconds):
    """Update manifest.json with end_time, status, and total runtime."""
    manifest_path = os.path.join(run_dir, 'manifest.json')
    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest['end_time'] = datetime.datetime.now().isoformat()
    manifest['status'] = status
    manifest['total_seconds'] = round(total_seconds, 1)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# Scheduler registry — names the CLI and HP search grid refer to
# ═══════════════════════════════════════════════════════════════════════════════
def resolve_scheduler(name, epochs=500):
    """Return (scheduler_cls, scheduler_kwargs) for a given scheduler name.

    None → covar_trainer default (ReduceLROnPlateau).
    """
    if name in (None, 'plateau'):
        return None, None  # covar_trainer will install its default ReduceLROnPlateau
    if name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR, {'T_max': epochs}
    if name == 'step':
        return torch.optim.lr_scheduler.StepLR, {'step_size': 20, 'gamma': 0.5}
    raise ValueError(f"Unknown scheduler: {name}")


# ═══════════════════════════════════════════════════════════════════════════════
# HP search (quick grid over lr × scheduler at fixed N)
# ═══════════════════════════════════════════════════════════════════════════════
def hp_search(device, N_hp=800, n_seeds=2, cv1=0.8, cv2=0.5):
    """Quick grid over (lr, scheduler) at a single fixed N.

    Selects the best combo by mean posthoc MSPE(fx) across seeds. Posthoc is
    the scientific target, so it's the right selection criterion — if a given
    (lr, sched) combo produces a backbone whose features give better posthoc
    estimates, that's what we want.
    """
    lr_grid = [3e-4, 1e-3, 3e-3, 1e-2]
    sched_grid = ['plateau', 'cosine']

    print(f"HP search: N={N_hp}, n_seeds={n_seeds}, device={device}", flush=True)
    print(f"  lr grid:        {lr_grid}")
    print(f"  scheduler grid: {sched_grid}")
    print()

    rows = []
    t0 = time.time()
    for sched_name in sched_grid:
        sched_cls, sched_kw = resolve_scheduler(sched_name)
        for lr in lr_grid:
            per_seed = []
            t_combo = time.time()
            for seed in range(n_seeds):
                r = run_one(N_hp, seed, cv1=cv1, cv2=cv2,
                            lr=lr, scheduler=sched_cls, scheduler_kwargs=sched_kw,
                            device=device)
                per_seed.append(r)
            base_fx  = np.mean([r['baseline']['mspe_fx'] for r in per_seed])
            ph_fx    = np.mean([r['posthoc']['mspe_fx']  for r in per_seed])
            base_y   = np.mean([r['baseline']['mspe_y']  for r in per_seed])
            ph_y     = np.mean([r['posthoc']['mspe_y']   for r in per_seed])
            base_cz  = np.mean([r['baseline']['corr_fx_z'] for r in per_seed])
            ph_cz    = np.mean([r['posthoc']['corr_fx_z']  for r in per_seed])
            vloss    = np.mean([r['baseline']['val_loss'] for r in per_seed])
            rows.append({'lr': lr, 'scheduler': sched_name,
                         'base_mspe_fx': base_fx, 'ph_mspe_fx': ph_fx,
                         'base_mspe_y':  base_y,  'ph_mspe_y':  ph_y,
                         'base_corr_z':  base_cz, 'ph_corr_z':  ph_cz,
                         'val_loss':     vloss})
            print(f"  lr={lr:.0e}  sched={sched_name:>8s}  "
                  f"val={vloss:.4f}  base_fx={base_fx:.4f} ph_fx={ph_fx:.4f}  "
                  f"base_cz={base_cz:+.3f} ph_cz={ph_cz:+.3f}  "
                  f"({time.time()-t_combo:4.1f}s)", flush=True)

    best = min(rows, key=lambda r: r['ph_mspe_fx'])
    print(f"\nTotal HP search time: {time.time()-t0:.1f}s")
    print(f"\nBest combo (by posthoc MSPE(fx)):")
    print(f"  lr        = {best['lr']:.0e}")
    print(f"  scheduler = {best['scheduler']}")
    print(f"  posthoc  MSPE(fx) = {best['ph_mspe_fx']:.4f}  "
          f"Corr(fx,Z) = {best['ph_corr_z']:+.3f}")
    print(f"  baseline MSPE(fx) = {best['base_mspe_fx']:.4f}  "
          f"Corr(fx,Z) = {best['base_corr_z']:+.3f}")
    return best


# ═══════════════════════════════════════════════════════════════════════════════
# N sweep (main experiment)
# ═══════════════════════════════════════════════════════════════════════════════
def n_sweep(args, device, lr, scheduler_name, run_dir):
    sdy = 1.0  # DGP noise std; Bayes-optimal MSPE(y) is sdy²
    true_corr            = true_corr_fx_z(b2=1.0, b3=1.0, cv2=args.cv2)
    baseline_ovb_corr    = baseline_corr_ols_limit(b2=1.0, b3=1.0, bz=1.0,
                                                   cv1=args.cv1, cv2=args.cv2)
    baseline_ovb_mspe_fx = baseline_mspe_fx_ols_limit(b2=1.0, b3=1.0, bz=1.0,
                                                      cv1=args.cv1, cv2=args.cv2)
    print(f"Run dir: {run_dir}")
    print(f"Analytic references (b2=b3=bz=1, cv1={args.cv1}, cv2={args.cv2}):")
    print(f"  Oracle Corr (true)                = {true_corr:.4f}")
    print(f"  Oracle OVB  Corr (OLS limit)      = {baseline_ovb_corr:.4f}")
    print(f"  Oracle OVB  MSPE(fx) (OLS limit)  = {baseline_ovb_mspe_fx:.4f}")
    print(f"  Population σ² = sdy²              = {sdy**2:.4f}")
    print(f"N values: {args.N_values}   seeds: {args.n_seeds}   "
          f"device={device}  lr={lr:.0e}  sched={scheduler_name}\n", flush=True)

    sched_cls, sched_kw = resolve_scheduler(scheduler_name)

    # Models directory (created already by setup_run_dir, but be defensive).
    models_dir = os.path.join(run_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    t0 = time.time()
    results = {m: [] for m in METHODS}

    # Parallelise over (N, seed) pairs. Each worker is a fresh subprocess so
    # we need spawn (not fork) for CUDA safety. Per-N timing is preserved by
    # counting completions and printing when all seeds for an N are done.
    mp_ctx = mp.get_context('spawn')
    n_workers = max(1, int(args.n_workers))

    futures_meta = {}  # future → (N, seed)
    n_done_per_N = {N: 0 for N in args.N_values}
    t_first_done_per_N = {}

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx) as pool:
        for N in args.N_values:
            for seed in range(args.n_seeds):
                f = pool.submit(
                    run_one, N, seed,
                    cv1=args.cv1, cv2=args.cv2,
                    lr=lr, scheduler=sched_cls, scheduler_kwargs=sched_kw,
                    device=device, models_dir=models_dir,
                    methods=args.methods,
                )
                futures_meta[f] = (N, seed)

        print(f"  submitted {len(futures_meta)} tasks to {n_workers} workers",
              flush=True)

        for f in as_completed(futures_meta):
            N, seed = futures_meta[f]
            r = f.result()
            for method in METHODS:
                results[method].append({'N': N, 'seed': seed, **r[method]})
            n_done_per_N[N] += 1
            if n_done_per_N[N] == args.n_seeds:
                t_first_done_per_N[N] = time.time()
                print(f"  N={N:5d} done ({args.n_seeds} seeds, "
                      f"total {time.time()-t0:5.1f}s)", flush=True)

    # Aggregate and print.
    metrics = ('mspe_y', 'mspe_fx', 'mspe_fz', 'corr_fx_z')
    agg = aggregate(results, METHODS, metrics, args.N_values)

    header = (f"\n{'method':>10s} {'N':>6s}  "
              f"{'MSPE(y)':>14s} {'MSPE(fx)':>14s} {'MSPE(fz)':>14s} "
              f"{'Corr(fx,Z)':>16s}")
    subhdr = (f"{'':>10s} {'':>6s}  "
              f"{'(true=' + f'{sdy**2:.1f})':>14s} "
              f"{'(true=0)':>14s} {'(true=0)':>14s} "
              f"{'(true=' + f'{true_corr:.3f})':>16s}")
    print(header); print(subhdr); print("-" * len(header))
    for method in METHODS:
        for i, N in enumerate(args.N_values):
            row = [agg[method][k][i] for k in metrics]
            cells = "  ".join(f"{m:.4f}±{s:.3f}" for m, s in row)
            print(f"{method:>10s} {N:6d}  {cells}")
        print()

    # ── Save raw per-(method, N, seed) results as CSV for later re-plotting ──
    csv_path = os.path.join(run_dir, 'raw_results.csv')
    fieldnames = ['method', 'N', 'seed', 'mspe_y', 'mspe_fx', 'mspe_fz',
                  'corr_fx_z', 'b_z_hat', 'val_loss']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method in METHODS:
            for r in results[method]:
                writer.writerow({
                    'method': method,
                    'N': r['N'], 'seed': r['seed'],
                    'mspe_y': r['mspe_y'], 'mspe_fx': r['mspe_fx'],
                    'mspe_fz': r['mspe_fz'], 'corr_fx_z': r['corr_fx_z'],
                    'b_z_hat': r.get('b_z_hat', float('nan')),
                    'val_loss': r.get('val_loss', float('nan')),
                })
    print(f"Raw results saved to {csv_path} "
          f"({len(args.N_values) * args.n_seeds * len(METHODS)} rows)\n")

    plot_convergence(agg, args.N_values, true_corr=true_corr,
                     baseline_ovb_corr=baseline_ovb_corr,
                     baseline_ovb_mspe_fx=baseline_ovb_mspe_fx,
                     cv1=args.cv1, cv2=args.cv2, sdy=sdy, out_dir=run_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['sweep', 'hp_search'], default='sweep')
    parser.add_argument('--n-seeds', type=int, default=10)
    parser.add_argument('--n-workers', type=int, default=10,
                        help='number of parallel subprocesses for the sweep '
                             '(set to 1 to run sequentially)')
    parser.add_argument('--N-values', type=int, nargs='+',
                        default=[100, 200, 400, 800, 1600, 3200, 6400, 12800,
                                 25600, 51200])
    parser.add_argument('--cv1', type=float, default=0.8)
    parser.add_argument('--cv2', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--scheduler', choices=['plateau', 'cosine', 'step'],
                        default='plateau')
    parser.add_argument('--device', default='auto',
                        help='"auto" picks cuda if available, else cpu')
    parser.add_argument('--run-name', default=None,
                        help='optional suffix appended to the timestamped run dir')
    parser.add_argument('--methods', nargs='+', default=None,
                        help='subset of methods to run; defaults to all of METHODS')
    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    out_root = os.path.dirname(os.path.abspath(__file__))

    if args.mode == 'hp_search':
        # HP search is quick and doesn't need a run dir.
        hp_search(device=device, cv1=args.cv1, cv2=args.cv2)
        return

    # ── Create a fresh run directory for this sweep ──
    run_dir = setup_run_dir(out_root, run_name=args.run_name)
    start_time = datetime.datetime.now().isoformat()

    # Redirect stdout/stderr into progress.log in addition to the terminal so
    # background runs can be monitored via `tail -f progress.log`.
    log_path = os.path.join(run_dir, 'progress.log')
    log_file = open(log_path, 'w', buffering=1)  # line-buffered
    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)

    config = {
        'mode':        args.mode,
        'cv1':         args.cv1,
        'cv2':         args.cv2,
        'N_values':    list(args.N_values),
        'n_seeds':     args.n_seeds,
        'n_workers':   args.n_workers,
        'lr':          args.lr,
        'scheduler':   args.scheduler,
        'device':      device,
        'dgp': {
            'b2': 1.0, 'b3': 1.0, 'bz': 1.0, 'sdy': 1.0,
            'N_test': 2000, 'seed_test': 9999,
        },
        'mlp': {'in_features': 3, 'hidden': 64, 'out_features': 32},
        'training': {'epochs': 500, 'patience': 20, 'batch_size_cap': 64},
        'methods': list(args.methods) if args.methods else list(METHODS),
    }
    write_config_and_manifest(run_dir, config, start_time)

    t_start = time.time()
    status = 'unknown'
    try:
        n_sweep(args, device=device, lr=args.lr, scheduler_name=args.scheduler,
                run_dir=run_dir)
        status = 'completed'
    except KeyboardInterrupt:
        status = 'interrupted'
        raise
    except Exception as e:
        status = f'failed: {type(e).__name__}: {e}'
        raise
    finally:
        finalize_manifest(run_dir, status, time.time() - t_start)
        # Restore stdout/stderr BEFORE closing the log file, otherwise
        # interpreter shutdown tries to flush via TeeStream into a closed file.
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.flush()
        log_file.close()


if __name__ == '__main__':
    main()
