#!/usr/bin/env python
"""Endogeneity test: compare training approaches for joint fx/fz estimation.

DGP with controlled confound strength (cv). Evaluates whether each approach
recovers the true direct effects fx and fz when the backbone is trained on
confounded data.

Approaches:
  5.  Pretrained baseline (identity backbone, no fine-tuning)
  2a. NAM-like (CovarNetwork, single optimizer, same lr)
  2b. Fast fz (CovarNetwork, larger lr for fz+intercept)
  3b. SGD backfitting (alternating fz/fx steps per batch)
"""
import copy
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cocodeel.dataset import CovarDataset
from cocodeel.model import CovarNetwork
from cocodeel.posthoc_model import PostHocCovarNetwork


# ─── DGP ──────────────────────────────────────────────────────────────────────
def make_dgp(N=500, d_in=20, cv=0.8, seed=42, num_covariates=2):
    """Generate confounded data with known true effects.

    Args:
        cv: confound strength. X features correlated with Z via cv.
        num_covariates: 1 = scalar Z (age-like), 2 = [age, sex] with different scales.

    Returns:
        X, Z, y, fx_true, fz_true (all torch tensors).
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    if num_covariates == 1:
        Z = torch.tensor(rng.standard_normal((N, 1)), dtype=torch.float32)
        X = torch.tensor(rng.standard_normal((N, d_in)), dtype=torch.float32)
        X[:, 0] = cv * Z.squeeze() + np.sqrt(max(1e-8, 1 - cv**2)) * X[:, 0]
        fx_true = 1.0 * X[:, [5]] + 0.5 * X[:, [6]]
        fz_true = 3.0 * Z
    else:
        # Z1 = "age" (continuous, large variance, std≈8)
        # Z2 = "sex" (binary, small variance, std≈0.5)
        Z1 = torch.tensor(rng.normal(60, 8, size=(N, 1)), dtype=torch.float32)
        Z2 = torch.tensor(rng.binomial(1, 0.5, size=(N, 1)), dtype=torch.float32)
        Z = torch.cat([Z1, Z2], dim=1)  # (N, 2)

        X = torch.tensor(rng.standard_normal((N, d_in)), dtype=torch.float32)
        # Both covariates confound X features (different features for each)
        X[:, 0] = cv * (Z1.squeeze() - 60) / 8 + np.sqrt(max(1e-8, 1 - cv**2)) * X[:, 0]
        X[:, 1] = cv * (Z2.squeeze() - 0.5) / 0.5 + np.sqrt(max(1e-8, 1 - cv**2)) * X[:, 1]

        # True effects: signal in Z-uncorrelated features
        fx_true = 1.0 * X[:, [5]] + 0.5 * X[:, [6]]
        # fz: age effect (standardized) + sex effect
        # Use standardized age so both coefficients are on similar scale in the DGP
        # but Z itself has very different scales → (Z'Z)⁻¹ matters
        fz_true = 0.3 * Z1 + 2.0 * Z2  # age coef on RAW scale, sex coef on raw scale

    y = fx_true + fz_true + 0.5 * torch.randn(N, 1)
    return X, Z, y, fx_true, fz_true


# ─── Backbone ─────────────────────────────────────────────────────────────────
class MLPBackbone(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32), nn.ReLU(), nn.Linear(32, out_features))
        self.out_features = out_features
        self.in_features = in_features

    def forward(self, x):
        return self.net(x)


class IdentityBackbone(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.out_features = out_features
        self.in_features = in_features

    def forward(self, x):
        return self.linear(x)


# ─── Evaluation ───────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, X_test, Z_test, fx_true_test, fz_true_test):
    """Evaluate centered predictions against centered true effects."""
    model.eval()
    fx_hat = model.predict_fx(X_test).squeeze()
    fz_hat = model.predict_fz(Z_test).squeeze()
    eta_hat = model.intercept + fx_hat + fz_hat
    y_hat = model.output_func(eta_hat.unsqueeze(1)).squeeze()

    fx_t = (fx_true_test - fx_true_test.mean()).squeeze()
    fz_t = (fz_true_test - fz_true_test.mean()).squeeze()
    fx_h = (fx_hat - fx_hat.mean())
    fz_h = (fz_hat - fz_hat.mean())

    z = Z_test[:, 0]

    return {
        'mspe_fx': float(((fx_h - fx_t) ** 2).mean()),
        'mspe_fz': float(((fz_h - fz_t) ** 2).mean()),
        'corr_z_fx': float(np.corrcoef(z.numpy(), fx_h.numpy())[0, 1]),
        'std_fx': float(fx_h.std()),
        'std_fz': float(fz_h.std()),
    }


# ─── Training helpers ─────────────────────────────────────────────────────────
def make_loaders(X, Z, y, ntrain=400, batch_size=64):
    tl = DataLoader(CovarDataset(X[:ntrain], Z[:ntrain], y[:ntrain]),
                    batch_size=batch_size, shuffle=True)
    vl = DataLoader(CovarDataset(X[ntrain:], Z[ntrain:], y[ntrain:]),
                    batch_size=batch_size, shuffle=False)
    tl_eval = DataLoader(CovarDataset(X[:ntrain], Z[:ntrain], y[:ntrain]),
                         batch_size=batch_size, shuffle=False)
    return tl, vl, tl_eval


def train_loop(model, tl, vl, optimizer, scheduler, epochs=100, patience=20):
    """Generic training loop with early stopping. Returns best model."""
    loss_fn = nn.MSELoss()
    best_val = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    patience_ctr = 0

    for epoch in range(epochs):
        model.train()
        for batch in tl:
            optimizer.zero_grad()
            eta = model(batch['X'], batch['Z'])
            loss = loss_fn(eta, batch['y'])
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0; n = 0
        with torch.no_grad():
            for batch in vl:
                eta = model(batch['X'], batch['Z'])
                val_loss += loss_fn(eta, batch['y']).item() * batch['X'].size(0)
                n += batch['X'].size(0)
        val_loss /= max(1, n)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    model.load_state_dict(best_state)
    model.best_epoch_ = best_epoch
    return model


# ─── Approach 5: Pretrained baseline ──────────────────────────────────────────
def train_approach_5(X, Z, y, d_in, d_out, ntrain, num_covariates=1):
    """Identity backbone + post-hoc FWL refit. Exogenous reference."""
    from cocodeel.model import BaseNetwork
    base = BaseNetwork(backbone=IdentityBackbone,
                       backbone_params={'in_features': d_in, 'out_features': d_out},
                       num_covariates=0, link='identity')
    base.backbone.linear.weight.data = torch.eye(d_out, d_in)
    base.backbone.linear.bias.data.zero_()
    # OLS for fx without Z control.
    w = torch.linalg.lstsq(X[:ntrain], y[:ntrain]).solution
    base.fx.weight.data = w[:d_out].T

    tl_eval = DataLoader(CovarDataset(X[:ntrain], Z[:ntrain], y[:ntrain]),
                         batch_size=64, shuffle=False)
    vl = DataLoader(CovarDataset(X[ntrain:], Z[ntrain:], y[ntrain:]),
                    batch_size=64, shuffle=False)

    phm = PostHocCovarNetwork(base, num_covariates=num_covariates)
    phm.fit(tl_eval, vl, n_lambdas=10)

    # Wrap into a CovarNetwork-like interface for uniform evaluation.
    # Re-create as CovarNetwork with same backbone.
    model = CovarNetwork(backbone=IdentityBackbone,
                         backbone_params={'in_features': d_in, 'out_features': d_out},
                         num_covariates=num_covariates, link='identity')
    model.backbone.linear.weight.data = torch.eye(d_out, d_in)
    model.backbone.linear.bias.data.zero_()
    model.fx.weight.data = phm.fx.weight.data.clone()
    model.fz.weight.data = phm.fz.weight.data.clone()
    model.intercept.data = phm.intercept.data.clone()
    model.center_x.mean = phm.center_x.mean.clone()
    model.center_z.mean = phm.center_z.mean.clone()
    model.center_y.mean = phm.center_y.mean.clone()
    model.is_centered.data = torch.tensor(True)
    return model


# ─── Approach 2a: NAM-like (same lr) ─────────────────────────────────────────
def train_approach_2a(X, Z, y, d_in, d_out, ntrain, num_covariates=1, epochs=100, lr=0.01):
    """CovarNetwork, single optimizer, same lr for all parameters."""
    model = CovarNetwork(backbone=MLPBackbone,
                         backbone_params={'in_features': d_in, 'out_features': d_out},
                         num_covariates=num_covariates, link='identity')
    tl, vl, tl_eval = make_loaders(X, Z, y, ntrain)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)

    model = train_loop(model, tl, vl, optimizer, scheduler, epochs=epochs)
    model.is_centered.data = torch.tensor(False)
    model.center_effects(tl_eval)
    return model


# ─── Approach 2b: Fast fz ────────────────────────────────────────────────────
def train_approach_2b(X, Z, y, d_in, d_out, ntrain, num_covariates=1, epochs=100,
                      lr_backbone=0.01, lr_fz=0.1):
    """CovarNetwork, separate optimizer groups: larger lr for fz+intercept."""
    model = CovarNetwork(backbone=MLPBackbone,
                         backbone_params={'in_features': d_in, 'out_features': d_out},
                         num_covariates=num_covariates, link='identity')
    tl, vl, tl_eval = make_loaders(X, Z, y, ntrain)

    optimizer = torch.optim.Adam([
        {'params': [*model.backbone.parameters(), *model.fx.parameters()],
         'lr': lr_backbone, 'weight_decay': 1e-4},
        {'params': [*model.fz.parameters(), model.intercept],
         'lr': lr_fz, 'weight_decay': 0},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)

    model = train_loop(model, tl, vl, optimizer, scheduler, epochs=epochs)
    model.is_centered.data = torch.tensor(False)
    model.center_effects(tl_eval)
    return model


# ─── Approach 2c: Preconditioned fz ((Z'Z)⁻¹ scaling) ─────────────────────────
def train_approach_2c(X, Z, y, d_in, d_out, ntrain, num_covariates=1, epochs=100, lr=0.01, patience=20):
    """CovarNetwork with (Z'Z)⁻¹-preconditioned manual updates for fz.

    Backbone+fx: Adam with lr. fz: manual update Δγ = α·(Z'Z)⁻¹·∇γ per batch,
    where α is scaled so the Newton step magnitude matches the backbone's step.
    """
    model = CovarNetwork(backbone=MLPBackbone,
                         backbone_params={'in_features': d_in, 'out_features': d_out},
                         num_covariates=num_covariates, link='identity')
    tl, vl, tl_eval = make_loaders(X, Z, y, ntrain)
    loss_fn = nn.MSELoss()

    # Precompute (Z'Z)⁻¹ on full training data (fixed).
    Z_tr = Z[:ntrain]
    ZtZ_inv = torch.linalg.inv(Z_tr.T @ Z_tr)  # (1, 1) for scalar Z

    # Adam for backbone+fx only; fz is updated manually.
    opt_fx = torch.optim.Adam(
        [*model.backbone.parameters(), *model.fx.parameters()],
        lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_fx, patience=5, factor=0.5)

    best_val = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    patience_ctr = 0

    # Step size for the preconditioned fz update. Full Newton = 1.0.
    # Scale down slightly for stability with mini-batches.
    alpha_fz = 0.5

    for epoch in range(epochs):
        model.train()
        for batch in tl:
            x, z, yb = batch['X'], batch['Z'], batch['y']

            opt_fx.zero_grad()
            # Need grad for fz too (to compute its gradient).
            model.fz.weight.requires_grad_(True)
            model.intercept.requires_grad_(True)

            eta = model(x, z)
            loss = loss_fn(eta, yb)
            loss.backward()

            # Step backbone+fx via Adam.
            opt_fx.step()

            # Step fz via preconditioned update.
            with torch.no_grad():
                if model.fz.weight.grad is not None:
                    # Preconditioned: Δγ = α · (Z'Z)⁻¹ · ∇γ  (Newton direction)
                    grad_fz = model.fz.weight.grad  # (1, num_covariates)
                    delta = alpha_fz * (ZtZ_inv @ grad_fz.T).T  # (1, num_covariates)
                    model.fz.weight.data -= delta
                if model.intercept.grad is not None:
                    model.intercept.data -= alpha_fz * model.intercept.grad

            model.fz.weight.requires_grad_(True)
            model.intercept.requires_grad_(True)

        model.eval()
        val_loss = 0; n = 0
        with torch.no_grad():
            for batch in vl:
                val_loss += loss_fn(model(batch['X'], batch['Z']),
                                    batch['y']).item() * batch['X'].size(0)
                n += batch['X'].size(0)
        val_loss /= max(1, n)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    model.load_state_dict(best_state)
    model.best_epoch_ = best_epoch
    model.is_centered.data = torch.tensor(False)
    model.center_effects(tl_eval)
    return model


# ─── Approach 3b: SGD backfitting ─────────────────────────────────────────────
def train_approach_3b(X, Z, y, d_in, d_out, ntrain, num_covariates=1, epochs=100, lr=0.01, patience=20):
    """Alternating fz/fx SGD steps per batch. fz steps first (gets first pick)."""
    model = CovarNetwork(backbone=MLPBackbone,
                         backbone_params={'in_features': d_in, 'out_features': d_out},
                         num_covariates=num_covariates, link='identity')
    tl, vl, tl_eval = make_loaders(X, Z, y, ntrain)
    loss_fn = nn.MSELoss()

    opt_fx = torch.optim.Adam(
        [*model.backbone.parameters(), *model.fx.parameters()],
        lr=lr, weight_decay=1e-4)
    opt_fz = torch.optim.Adam(
        [*model.fz.parameters(), model.intercept],
        lr=lr, weight_decay=0)
    sched_fx = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_fx, patience=5, factor=0.5)
    sched_fz = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_fz, patience=5, factor=0.5)

    best_val = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    patience_ctr = 0

    for epoch in range(epochs):
        model.train()
        for batch in tl:
            x, z, yb = batch['X'], batch['Z'], batch['y']

            # Step 1: update fz on current residual.
            opt_fz.zero_grad()
            loss_fz = loss_fn(model(x, z), yb)
            loss_fz.backward()
            opt_fz.step()

            # Step 2: update backbone+fx on current residual.
            opt_fx.zero_grad()
            loss_fx = loss_fn(model(x, z), yb)
            loss_fx.backward()
            opt_fx.step()

        model.eval()
        val_loss = 0; n = 0
        with torch.no_grad():
            for batch in vl:
                val_loss += loss_fn(model(batch['X'], batch['Z']),
                                    batch['y']).item() * batch['X'].size(0)
                n += batch['X'].size(0)
        val_loss /= max(1, n)

        sched_fx.step(val_loss)
        sched_fz.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    model.load_state_dict(best_state)
    model.best_epoch_ = best_epoch
    model.is_centered.data = torch.tensor(False)
    model.center_effects(tl_eval)
    return model


# ─── Approach 6: Post-hoc refit (current method) ─────────────────────────────
def train_approach_posthoc(X, Z, y, d_in, d_out, ntrain, num_covariates=1, epochs=100, lr=0.01, patience=20):
    """Train backbone WITHOUT Z, then post-hoc FWL refit. The current method."""
    from cocodeel.model import BaseNetwork
    # Step 1: train backbone end-to-end without Z.
    base = BaseNetwork(backbone=MLPBackbone,
                       backbone_params={'in_features': d_in, 'out_features': d_out},
                       num_covariates=0, link='identity')
    tl_noZ = DataLoader(CovarDataset(X[:ntrain], Z[:ntrain], y[:ntrain]),
                        batch_size=64, shuffle=True)
    vl_noZ = DataLoader(CovarDataset(X[ntrain:], Z[ntrain:], y[ntrain:]),
                        batch_size=64, shuffle=False)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(base.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)

    best_val = float('inf'); best_state = copy.deepcopy(base.state_dict())
    best_epoch = 0; patience_ctr = 0
    for epoch in range(epochs):
        base.train()
        for batch in tl_noZ:
            optimizer.zero_grad()
            loss = loss_fn(base(batch['X']), batch['y'])
            loss.backward()
            optimizer.step()
        base.eval()
        vl_loss = 0; n = 0
        with torch.no_grad():
            for batch in vl_noZ:
                vl_loss += loss_fn(base(batch['X']), batch['y']).item() * batch['X'].size(0)
                n += batch['X'].size(0)
        vl_loss /= max(1, n)
        scheduler.step(vl_loss)
        if vl_loss < best_val:
            best_val = vl_loss; best_state = copy.deepcopy(base.state_dict())
            best_epoch = epoch; patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience: break
    base.load_state_dict(best_state)

    # Step 2: post-hoc FWL refit.
    tl_eval = DataLoader(CovarDataset(X[:ntrain], Z[:ntrain], y[:ntrain]),
                         batch_size=64, shuffle=False)
    vl_eval = DataLoader(CovarDataset(X[ntrain:], Z[ntrain:], y[ntrain:]),
                         batch_size=64, shuffle=False)
    phm = PostHocCovarNetwork(base, num_covariates=num_covariates)
    phm.fit(tl_eval, vl_eval, n_lambdas=10)

    # Wrap into CovarNetwork for uniform evaluation.
    model = CovarNetwork(backbone=MLPBackbone,
                         backbone_params={'in_features': d_in, 'out_features': d_out},
                         num_covariates=num_covariates, link='identity')
    model.load_state_dict(phm.state_dict(), strict=False)
    model.center_x.mean = phm.center_x.mean.clone()
    model.center_z.mean = phm.center_z.mean.clone()
    model.center_y.mean = phm.center_y.mean.clone()
    model.is_centered.data = torch.tensor(True)
    return model


# ─── Approach 7: CovarNetwork + post-hoc refit ────────────────────────────────
def train_approach_7_covar_posthoc(X, Z, y, d_in, d_out, ntrain, num_covariates=1,
                                   epochs=100, lr=0.01, patience=20):
    """Train CovarNetwork end-to-end (with fz), then post-hoc refit.

    Tests whether training with fz first (reducing endogeneity) followed by
    post-hoc refit (exact FWL) improves over pure post-hoc (approach 6).
    """
    # Step 1: train CovarNetwork end-to-end (same as 2a).
    model = CovarNetwork(backbone=MLPBackbone,
                         backbone_params={'in_features': d_in, 'out_features': d_out},
                         num_covariates=num_covariates, link='identity')
    tl, vl, tl_eval = make_loaders(X, Z, y, ntrain)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)

    model = train_loop(model, tl, vl, optimizer, scheduler, epochs=epochs)

    # Step 2: freeze backbone, post-hoc refit fx+fz via FWL.
    # Create a BaseNetwork from the trained backbone for PostHocCovarNetwork.
    from cocodeel.model import BaseNetwork
    base = BaseNetwork(backbone=MLPBackbone,
                       backbone_params={'in_features': d_in, 'out_features': d_out},
                       num_covariates=0, link='identity')
    # Copy trained backbone weights.
    base.backbone.load_state_dict(model.backbone.state_dict())
    base.fx.weight.data = model.fx.weight.data.clone()
    base.intercept.data = model.intercept.data.clone()

    tl_eval2 = DataLoader(CovarDataset(X[:ntrain], Z[:ntrain], y[:ntrain]),
                          batch_size=64, shuffle=False)
    vl2 = DataLoader(CovarDataset(X[ntrain:], Z[ntrain:], y[ntrain:]),
                     batch_size=64, shuffle=False)
    phm = PostHocCovarNetwork(base, num_covariates=num_covariates)
    phm.fit(tl_eval2, vl2, n_lambdas=10)

    # Wrap into CovarNetwork.
    out = CovarNetwork(backbone=MLPBackbone,
                       backbone_params={'in_features': d_in, 'out_features': d_out},
                       num_covariates=num_covariates, link='identity')
    out.load_state_dict(phm.state_dict(), strict=False)
    out.center_x.mean = phm.center_x.mean.clone()
    out.center_z.mean = phm.center_z.mean.clone()
    out.center_y.mean = phm.center_y.mean.clone()
    out.is_centered.data = torch.tensor(True)
    return out


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    d_in, d_out, N, ntrain = 20, 16, 500, 400

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-covariates', type=int, default=1, choices=[1, 2])
    args = parser.parse_args()
    nc = args.num_covariates

    approaches = {
        '5_pretrained':  lambda X, Z, y: train_approach_5(X, Z, y, d_in, d_out, ntrain, nc),
        '6_posthoc':     lambda X, Z, y: train_approach_posthoc(X, Z, y, d_in, d_out, ntrain, nc),
        '7_covar+ph':    lambda X, Z, y: train_approach_7_covar_posthoc(X, Z, y, d_in, d_out, ntrain, nc),
        '2a_nam':        lambda X, Z, y: train_approach_2a(X, Z, y, d_in, d_out, ntrain, nc),
        '2b_fast_fz':    lambda X, Z, y: train_approach_2b(X, Z, y, d_in, d_out, ntrain, nc),
        '2c_precond':    lambda X, Z, y: train_approach_2c(X, Z, y, d_in, d_out, ntrain, nc),
        '3b_backfit':    lambda X, Z, y: train_approach_3b(X, Z, y, d_in, d_out, ntrain, nc),
    }

    cv_values = [0.0, 0.2, 0.4, 0.6, 0.8]

    colors = {
        '5_pretrained': '#000000',
        '6_posthoc':    '#888888',
        '7_covar+ph':   '#bbbbbb',
        '2a_nam':       '#aec7e8',
        '2b_fast_fz':   '#1f77b4',
        '2c_precond':   '#08306b',
        '3b_backfit':   '#d62728',
    }
    labels = {
        '5_pretrained': '5: pretrained (baseline)',
        '6_posthoc':    '6: base+post-hoc (current)',
        '7_covar+ph':   '7: covar+post-hoc',
        '2a_nam':       '2a: NAM (same lr)',
        '2b_fast_fz':   '2b: fast fz (lr=0.1)',
        '2c_precond':   '2c: preconditioned fz',
        '3b_backfit':   '3b: SGD backfitting',
    }

    results = {name: {m: [] for m in ['mspe_fx', 'mspe_fz', 'corr_z_fx']}
               for name in approaches}

    print(f"num_covariates={nc}")
    print(f"{'approach':>15s} {'cv':>4s}  {'mspe_fx':>8s} {'mspe_fz':>8s} "
          f"{'corr_Z':>7s} {'std_fx':>7s} {'std_fz':>7s}")
    print("-" * 65)

    for cv in cv_values:
        X, Z, y, fx_true, fz_true = make_dgp(N=N, d_in=d_in, cv=cv, num_covariates=nc)
        X_te, Z_te = X[ntrain:], Z[ntrain:]
        fx_te, fz_te = fx_true[ntrain:], fz_true[ntrain:]

        for name, train_fn in approaches.items():
            t0 = time.time()
            model = train_fn(X, Z, y)
            elapsed = time.time() - t0

            res = evaluate(model, X_te, Z_te, fx_te, fz_te)
            for m in ['mspe_fx', 'mspe_fz', 'corr_z_fx']:
                results[name][m].append(res[m])

            print(f"{name:>15s} {cv:4.1f}  {res['mspe_fx']:8.4f} {res['mspe_fz']:8.4f} "
                  f"{res['corr_z_fx']:7.4f} {res['std_fx']:7.4f} {res['std_fz']:7.4f}"
                  f"  ({elapsed:.1f}s)", flush=True)

        print()

    # ── Plot ──
    metrics = [('mspe_fx', 'MSPE($f_x$)'), ('mspe_fz', 'MSPE($f_z$)'),
               ('corr_z_fx', 'Corr($Z$, $\\hat{f}_x$)')]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, (metric, ylabel) in zip(axes, metrics):
        for name in approaches:
            vals = results[name][metric]
            style = '--' if name in ('5_pretrained', '6_posthoc') else '-'
            marker = 's' if name in ('5_pretrained', '6_posthoc') else 'o'
            ax.plot(cv_values, vals, style, marker=marker, color=colors[name],
                    label=labels[name], linewidth=2, markersize=5)
        ax.set_xlabel('cv (confound strength)')
        ax.set_ylabel(ylabel)
        ax.set_xticks(cv_values)
        if metric == 'corr_z_fx':
            ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')

    axes[0].legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    fig.savefig('experiments/endogeneity_test_results.pdf', bbox_inches='tight', dpi=150)
    fig.savefig('experiments/endogeneity_test_results.png', bbox_inches='tight', dpi=150)
    print(f"\nPlot saved to experiments/endogeneity_test_results.{{pdf,png}}")
