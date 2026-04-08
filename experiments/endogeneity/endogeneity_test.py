#!/usr/bin/env python
"""Endogeneity test: compare training approaches for joint fx/fz estimation.

DGP with controlled confound strength (cv). Evaluates whether each approach
recovers the true direct effects fx and fz when the backbone is trained on
confounded data.

Multi-seed: fixed test set per cv, variable training sets. Computes bias²/var
decomposition of MSPE following cocodeel/experiments/simulation_images/utils.py.

Usage:
    python -m experiments.endogeneity_test --num-covariates 1 --n-seeds 20 --n-workers 20
"""
import argparse
import copy
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cocodeel.dataset import CovarDataset
from cocodeel.model import BaseNetwork, CovarNetwork
from cocodeel.posthoc_model import PostHocCovarNetwork


# ═══════════════════════════════════════════════════════════════════════════════
# DGP
# ═══════════════════════════════════════════════════════════════════════════════
def _generate(N, d_in, cv, rng, num_covariates):
    """Core DGP: generate X, Z, y, fx_true, fz_true from a given rng."""
    if num_covariates == 1:
        Z = torch.tensor(rng.standard_normal((N, 1)), dtype=torch.float32)
        X = torch.tensor(rng.standard_normal((N, d_in)), dtype=torch.float32)
        X[:, 0] = cv * Z.squeeze() + np.sqrt(max(1e-8, 1 - cv**2)) * X[:, 0]
        fx_true = 1.0 * X[:, [5]] + 0.5 * X[:, [6]]
        fz_true = 3.0 * Z
    else:
        Z1_raw = torch.tensor(rng.normal(60, 8, size=(N, 1)), dtype=torch.float32)
        Z2_raw = torch.tensor(rng.binomial(1, 0.5, size=(N, 1)), dtype=torch.float32)
        Z1 = (Z1_raw - Z1_raw.mean()) / (Z1_raw.std() + 1e-8)
        Z2 = (Z2_raw - Z2_raw.mean()) / (Z2_raw.std() + 1e-8)
        Z = torch.cat([Z1, Z2], dim=1)
        X = torch.tensor(rng.standard_normal((N, d_in)), dtype=torch.float32)
        X[:, 0] = cv * Z1.squeeze() + np.sqrt(max(1e-8, 1 - cv**2)) * X[:, 0]
        X[:, 1] = cv * Z2.squeeze() + np.sqrt(max(1e-8, 1 - cv**2)) * X[:, 1]
        fx_true = 1.0 * X[:, [5]] + 0.5 * X[:, [6]]
        fz_true = 2.0 * Z1 + 3.0 * Z2

    y = fx_true + fz_true + 0.5 * torch.tensor(
        rng.standard_normal((N, 1)), dtype=torch.float32)
    return X, Z, y, fx_true, fz_true


def make_dgp(N_train, N_test, d_in, cv, seed_train, seed_test=9999, num_covariates=1):
    """Generate separate train/test sets. Test set is fixed across seeds."""
    rng_test = np.random.default_rng(seed_test)
    X_te, Z_te, y_te, fx_te, fz_te = _generate(N_test, d_in, cv, rng_test, num_covariates)
    rng_train = np.random.default_rng(seed_train)
    X_tr, Z_tr, y_tr, fx_tr, fz_tr = _generate(N_train, d_in, cv, rng_train, num_covariates)
    return (X_tr, Z_tr, y_tr), (X_te, Z_te, y_te, fx_te, fz_te)


# ═══════════════════════════════════════════════════════════════════════════════
# Backbone
# ═══════════════════════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation: return raw centered predictions for bias/var decomposition
# ═══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def predict_centered(model, X_test, Z_test):
    """Return centered fx_hat and fz_hat as numpy arrays (n_test,)."""
    model.eval()
    fx_hat = model.predict_fx(X_test).squeeze()
    fz_hat = model.predict_fz(Z_test).squeeze()
    fx_hat = fx_hat - fx_hat.mean()
    fz_hat = fz_hat - fz_hat.mean()
    return fx_hat.numpy(), fz_hat.numpy()


def bias_var_decomposition(preds, targets):
    """Bias²/Var decomposition following cocodeel simulation_images/utils.py.

    Args:
        preds:   (n_test, n_seeds) — predictions from different training runs.
        targets: (n_test,) — true values (centered), same across seeds.

    Returns:
        dict with bias2, var, mspe.
    """
    mean_pred = preds.mean(axis=1)  # (n_test,)
    bias2 = ((mean_pred - targets) ** 2).mean()
    var = ((preds - mean_pred[:, None]) ** 2).mean()
    mspe = ((preds - targets[:, None]) ** 2).mean()
    return {'bias2': float(bias2), 'var': float(var), 'mspe': float(mspe)}


# ═══════════════════════════════════════════════════════════════════════════════
# Training helpers
# ═══════════════════════════════════════════════════════════════════════════════
def make_loaders(X, Z, y, batch_size=64):
    tl = DataLoader(CovarDataset(X, Z, y), batch_size=batch_size, shuffle=True)
    tl_eval = DataLoader(CovarDataset(X, Z, y), batch_size=batch_size, shuffle=False)
    return tl, tl_eval


def train_loop(model, tl, optimizer, scheduler, epochs=100, patience=20):
    """Train with early stopping. No val set — use train loss for stopping."""
    loss_fn = nn.MSELoss()
    best_loss = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    patience_ctr = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0; nb = 0
        for batch in tl:
            optimizer.zero_grad()
            eta = model(batch['X'], batch['Z'])
            loss = loss_fn(eta, batch['y'])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item(); nb += 1
        epoch_loss /= nb

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_loss)
        else:
            scheduler.step()

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    model.load_state_dict(best_state)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Approaches
# ═══════════════════════════════════════════════════════════════════════════════
D_IN, D_OUT = 20, 16  # module-level constants for all approaches


def train_approach_5(X_tr, Z_tr, y_tr, nc):
    """Pretrained baseline: identity backbone + post-hoc FWL."""
    base = BaseNetwork(backbone=IdentityBackbone,
                       backbone_params={'in_features': D_IN, 'out_features': D_OUT},
                       num_covariates=0, link='identity')
    base.backbone.linear.weight.data = torch.eye(D_OUT, D_IN)
    base.backbone.linear.bias.data.zero_()
    w = torch.linalg.lstsq(X_tr, y_tr).solution
    base.fx.weight.data = w[:D_OUT].T

    tl_eval = DataLoader(CovarDataset(X_tr, Z_tr, y_tr), batch_size=64, shuffle=False)
    phm = PostHocCovarNetwork(base, num_covariates=nc)
    phm.fit(tl_eval, tl_eval, n_lambdas=10)

    model = CovarNetwork(backbone=IdentityBackbone,
                         backbone_params={'in_features': D_IN, 'out_features': D_OUT},
                         num_covariates=nc, link='identity')
    model.backbone.linear.weight.data = torch.eye(D_OUT, D_IN)
    model.backbone.linear.bias.data.zero_()
    model.fx.weight.data = phm.fx.weight.data.clone()
    model.fz.weight.data = phm.fz.weight.data.clone()
    model.intercept.data = phm.intercept.data.clone()
    model.center_x.mean = phm.center_x.mean.clone()
    model.center_z.mean = phm.center_z.mean.clone()
    model.center_y.mean = phm.center_y.mean.clone()
    model.is_centered.data = torch.tensor(True)
    return model


def train_approach_posthoc(X_tr, Z_tr, y_tr, nc, epochs=100, lr=0.01):
    """Current method: train backbone WITHOUT Z, then post-hoc FWL refit."""
    base = BaseNetwork(backbone=MLPBackbone,
                       backbone_params={'in_features': D_IN, 'out_features': D_OUT},
                       num_covariates=0, link='identity')
    tl = DataLoader(CovarDataset(X_tr, Z_tr, y_tr), batch_size=64, shuffle=True)
    tl_eval = DataLoader(CovarDataset(X_tr, Z_tr, y_tr), batch_size=64, shuffle=False)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(base.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    best_loss = float('inf'); best_state = copy.deepcopy(base.state_dict()); pctr = 0
    for epoch in range(epochs):
        base.train()
        eloss = 0; nb = 0
        for batch in tl:
            opt.zero_grad()
            loss = loss_fn(base(batch['X']), batch['y'])
            loss.backward(); opt.step()
            eloss += loss.item(); nb += 1
        eloss /= nb; sched.step(eloss)
        if eloss < best_loss:
            best_loss = eloss; best_state = copy.deepcopy(base.state_dict()); pctr = 0
        else:
            pctr += 1
            if pctr >= 20: break
    base.load_state_dict(best_state)

    phm = PostHocCovarNetwork(base, num_covariates=nc)
    phm.fit(tl_eval, tl_eval, n_lambdas=10)

    model = CovarNetwork(backbone=MLPBackbone,
                         backbone_params={'in_features': D_IN, 'out_features': D_OUT},
                         num_covariates=nc, link='identity')
    model.load_state_dict(phm.state_dict(), strict=False)
    model.center_x.mean = phm.center_x.mean.clone()
    model.center_z.mean = phm.center_z.mean.clone()
    model.center_y.mean = phm.center_y.mean.clone()
    model.is_centered.data = torch.tensor(True)
    return model


def train_approach_2a(X_tr, Z_tr, y_tr, nc, epochs=100, lr=0.01):
    """NAM-like: CovarNetwork, single optimizer, same lr."""
    model = CovarNetwork(backbone=MLPBackbone,
                         backbone_params={'in_features': D_IN, 'out_features': D_OUT},
                         num_covariates=nc, link='identity')
    tl, tl_eval = make_loaders(X_tr, Z_tr, y_tr)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    model = train_loop(model, tl, opt, sched, epochs=epochs)
    model.is_centered.data = torch.tensor(False)
    model.center_effects(tl_eval)
    return model


def train_approach_2b(X_tr, Z_tr, y_tr, nc, epochs=100, lr_backbone=0.01, lr_fz=0.1):
    """Fast fz: separate optimizer groups, larger lr for fz+intercept."""
    model = CovarNetwork(backbone=MLPBackbone,
                         backbone_params={'in_features': D_IN, 'out_features': D_OUT},
                         num_covariates=nc, link='identity')
    tl, tl_eval = make_loaders(X_tr, Z_tr, y_tr)
    opt = torch.optim.Adam([
        {'params': [*model.backbone.parameters(), *model.fx.parameters()],
         'lr': lr_backbone, 'weight_decay': 1e-4},
        {'params': [*model.fz.parameters(), model.intercept],
         'lr': lr_fz, 'weight_decay': 0},
    ])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    model = train_loop(model, tl, opt, sched, epochs=epochs)
    model.is_centered.data = torch.tensor(False)
    model.center_effects(tl_eval)
    return model


def train_approach_2c(X_tr, Z_tr, y_tr, nc, epochs=100, lr=0.01, patience=20):
    """Preconditioned fz: (Z'Z)⁻¹-scaled manual Newton step. Z assumed standardized."""
    model = CovarNetwork(backbone=MLPBackbone,
                         backbone_params={'in_features': D_IN, 'out_features': D_OUT},
                         num_covariates=nc, link='identity')
    tl, tl_eval = make_loaders(X_tr, Z_tr, y_tr)
    loss_fn = nn.MSELoss()

    ZtZ_inv = torch.linalg.inv(Z_tr.T @ Z_tr)
    opt_fx = torch.optim.Adam(
        [*model.backbone.parameters(), *model.fx.parameters()],
        lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_fx, patience=5, factor=0.5)

    best_loss = float('inf'); best_state = copy.deepcopy(model.state_dict()); pctr = 0
    alpha_fz = 0.5

    for epoch in range(epochs):
        model.train()
        eloss = 0; nb = 0
        for batch in tl:
            x, z, yb = batch['X'], batch['Z'], batch['y']
            opt_fx.zero_grad()
            model.fz.weight.requires_grad_(True)
            model.intercept.requires_grad_(True)

            loss = loss_fn(model(x, z), yb)
            loss.backward()
            opt_fx.step()

            with torch.no_grad():
                if model.fz.weight.grad is not None:
                    delta = alpha_fz * (ZtZ_inv @ model.fz.weight.grad.T).T
                    model.fz.weight.data -= delta
                if model.intercept.grad is not None:
                    model.intercept.data -= alpha_fz * model.intercept.grad
            model.fz.weight.requires_grad_(True)
            model.intercept.requires_grad_(True)
            eloss += loss.item(); nb += 1
        eloss /= nb; sched.step(eloss)

        if eloss < best_loss:
            best_loss = eloss; best_state = copy.deepcopy(model.state_dict()); pctr = 0
        else:
            pctr += 1
            if pctr >= patience: break

    model.load_state_dict(best_state)
    model.is_centered.data = torch.tensor(False)
    model.center_effects(tl_eval)
    return model


def train_approach_3b(X_tr, Z_tr, y_tr, nc, epochs=100, lr=0.01, patience=20):
    """SGD backfitting: alternating fz/fx steps per batch."""
    model = CovarNetwork(backbone=MLPBackbone,
                         backbone_params={'in_features': D_IN, 'out_features': D_OUT},
                         num_covariates=nc, link='identity')
    tl, tl_eval = make_loaders(X_tr, Z_tr, y_tr)
    loss_fn = nn.MSELoss()

    opt_fx = torch.optim.Adam(
        [*model.backbone.parameters(), *model.fx.parameters()],
        lr=lr, weight_decay=1e-4)
    opt_fz = torch.optim.Adam(
        [*model.fz.parameters(), model.intercept], lr=lr, weight_decay=0)
    sched_fx = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_fx, patience=5, factor=0.5)
    sched_fz = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_fz, patience=5, factor=0.5)

    best_loss = float('inf'); best_state = copy.deepcopy(model.state_dict()); pctr = 0
    for epoch in range(epochs):
        model.train()
        eloss = 0; nb = 0
        for batch in tl:
            x, z, yb = batch['X'], batch['Z'], batch['y']
            # Step fz first.
            opt_fz.zero_grad()
            loss = loss_fn(model(x, z), yb)
            loss.backward(); opt_fz.step()
            # Step backbone+fx.
            opt_fx.zero_grad()
            loss = loss_fn(model(x, z), yb)
            loss.backward(); opt_fx.step()
            eloss += loss.item(); nb += 1
        eloss /= nb; sched_fx.step(eloss); sched_fz.step(eloss)

        if eloss < best_loss:
            best_loss = eloss; best_state = copy.deepcopy(model.state_dict()); pctr = 0
        else:
            pctr += 1
            if pctr >= patience: break

    model.load_state_dict(best_state)
    model.is_centered.data = torch.tensor(False)
    model.center_effects(tl_eval)
    return model


# ─── Approach 2d: Formula-derived lr for fz ───────────────────────────────────
def train_approach_2d(X_tr, Z_tr, y_tr, nc, epochs=100, lr_backbone=0.01, batch_size=64):
    """CovarNetwork with lr_fz derived from (Z'Z)⁻¹ diagonal.

    lr_fz = batch_size / diag(Z'Z).mean(). Adapts to N and covariate scale
    automatically — no HP tuning needed for fz.
    """
    model = CovarNetwork(backbone=MLPBackbone,
                         backbone_params={'in_features': D_IN, 'out_features': D_OUT},
                         num_covariates=nc, link='identity')
    tl, tl_eval = make_loaders(X_tr, Z_tr, y_tr, batch_size=batch_size)

    # Formula: lr_fz = batch_size / mean(diag(Z'Z))
    ZtZ_diag = (Z_tr.T @ Z_tr).diag()
    lr_fz = float(batch_size / ZtZ_diag.mean())

    opt = torch.optim.Adam([
        {'params': [*model.backbone.parameters(), *model.fx.parameters()],
         'lr': lr_backbone, 'weight_decay': 1e-4},
        {'params': [*model.fz.parameters(), model.intercept],
         'lr': lr_fz, 'weight_decay': 0},
    ])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    model = train_loop(model, tl, opt, sched, epochs=epochs)
    model.is_centered.data = torch.tensor(False)
    model.center_effects(tl_eval)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Runner: one seed, all approaches, all cv values
# ═══════════════════════════════════════════════════════════════════════════════
APPROACHES = {
    '5_pretrained': train_approach_5,
    '6_posthoc':    train_approach_posthoc,
    '2a_nam':       train_approach_2a,
    '2b_fast_fz':   train_approach_2b,
    '2c_precond':   train_approach_2c,
    '2d_formula':   train_approach_2d,
    '3b_backfit':   train_approach_3b,
}


def run_seed(seed, cv_values, nc, N_train=400, N_test=200):
    """Run all approaches for one training seed. Returns raw predictions."""
    torch.manual_seed(seed)
    rows = []
    for cv in cv_values:
        (X_tr, Z_tr, y_tr), (X_te, Z_te, y_te, fx_te, fz_te) = make_dgp(
            N_train, N_test, D_IN, cv, seed_train=seed, num_covariates=nc)

        # Centered targets (fixed across seeds for this cv via seed_test=9999).
        fx_target = (fx_te - fx_te.mean()).squeeze().numpy()
        fz_target = (fz_te - fz_te.mean()).squeeze().numpy()
        z_test = Z_te[:, 0].numpy()

        for name, train_fn in APPROACHES.items():
            model = train_fn(X_tr, Z_tr, y_tr, nc)
            fx_hat, fz_hat = predict_centered(model, X_te, Z_te)
            corr_z = float(np.corrcoef(z_test, fx_hat)[0, 1])
            rows.append({
                'seed': seed, 'cv': cv, 'approach': name,
                'fx_hat': fx_hat, 'fz_hat': fz_hat,
                'fx_target': fx_target, 'fz_target': fz_target,
                'corr_z_fx': corr_z,
            })
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-covariates', type=int, default=1, choices=[1, 2])
    parser.add_argument('--n-seeds', type=int, default=1)
    parser.add_argument('--n-workers', type=int, default=1)
    parser.add_argument('--mode', choices=['cv_sweep', 'n_sweep'], default='cv_sweep')
    args = parser.parse_args()
    nc = args.num_covariates
    cv_values = [0.0, 0.2, 0.4, 0.6, 0.8]
    approach_names = list(APPROACHES.keys())

    import json, os, datetime
    OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(OUT_DIR, exist_ok=True)

    config = {
        'date': datetime.datetime.now().isoformat(),
        'num_covariates': nc, 'n_seeds': args.n_seeds, 'n_workers': args.n_workers,
        'cv_values': cv_values, 'N_train': 400, 'N_test': 200,
        'd_in': D_IN, 'd_out': D_OUT, 'approaches': approach_names,
    }
    with open(os.path.join(OUT_DIR, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"num_covariates={nc}, n_seeds={args.n_seeds}, n_workers={args.n_workers}",
          flush=True)
    t0_all = time.time()

    # ── Collect results ──
    all_results = []
    if args.n_workers <= 1 or args.n_seeds == 1:
        for seed in range(args.n_seeds):
            all_results.extend(run_seed(seed, cv_values, nc))
            print(f"  seed {seed} done", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            futures = {pool.submit(run_seed, s, cv_values, nc): s
                       for s in range(args.n_seeds)}
            for future in as_completed(futures):
                all_results.extend(future.result())
                print(f"  seed {futures[future]} done", flush=True)

    print(f"Total: {time.time()-t0_all:.1f}s\n", flush=True)

    # ── Bias²/Var decomposition ──
    # For each (approach, cv): stack predictions across seeds → (n_test, n_seeds).
    results = {}
    for name in approach_names:
        results[name] = {m: [] for m in ['bias2_fx', 'var_fx', 'bias2_fz', 'var_fz',
                                          'corr_z_fx']}

    print(f"{'approach':>15s} {'cv':>4s}  {'bias2_fx':>10s} {'var_fx':>10s}  "
          f"{'bias2_fz':>10s} {'var_fz':>10s}  {'corr_Z':>10s}")
    print("-" * 80)

    for cv in cv_values:
        for name in approach_names:
            subset = [r for r in all_results if r['cv'] == cv and r['approach'] == name]
            n_seeds = len(subset)

            # Stack predictions: (n_test, n_seeds).
            fx_preds = np.stack([r['fx_hat'] for r in subset], axis=1)
            fz_preds = np.stack([r['fz_hat'] for r in subset], axis=1)
            fx_target = subset[0]['fx_target']  # same across seeds (fixed test set)
            fz_target = subset[0]['fz_target']

            dec_fx = bias_var_decomposition(fx_preds, fx_target)
            dec_fz = bias_var_decomposition(fz_preds, fz_target)
            corr_z_vals = [r['corr_z_fx'] for r in subset]
            mean_corr = np.mean(corr_z_vals)

            results[name]['bias2_fx'].append(dec_fx['bias2'])
            results[name]['var_fx'].append(dec_fx['var'])
            results[name]['bias2_fz'].append(dec_fz['bias2'])
            results[name]['var_fz'].append(dec_fz['var'])
            results[name]['corr_z_fx'].append(mean_corr)

            if n_seeds > 1:
                print(f"{name:>15s} {cv:4.1f}  {dec_fx['bias2']:10.4f} {dec_fx['var']:10.4f}  "
                      f"{dec_fz['bias2']:10.4f} {dec_fz['var']:10.4f}  "
                      f"{mean_corr:+7.4f}±{np.std(corr_z_vals):.4f}", flush=True)
            else:
                print(f"{name:>15s} {cv:4.1f}  {dec_fx['bias2']:10.4f} {dec_fx['var']:10.4f}  "
                      f"{dec_fz['bias2']:10.4f} {dec_fz['var']:10.4f}  "
                      f"{mean_corr:+10.4f}", flush=True)
        print()

    # ── Plotting helpers ──
    colors = {
        '5_pretrained': '#000000',
        '6_posthoc':    '#888888',
        '2a_nam':       '#aec7e8',
        '2b_fast_fz':   '#1f77b4',
        '2c_precond':   '#6baed6',
        '2d_formula':   '#08306b',
        '3b_backfit':   '#d62728',
    }
    labels = {
        '5_pretrained': '5: pretrained (baseline)',
        '6_posthoc':    '6: post-hoc (current)',
        '2a_nam':       '2a: NAM (same lr)',
        '2b_fast_fz':   '2b: fast fz (oracle lr)',
        '2c_precond':   '2c: precond fz',
        '2d_formula':   '2d: formula lr (ours)',
        '3b_backfit':   '3b: SGD backfitting',
    }
    baselines = ('5_pretrained', '6_posthoc')

    def plot_sweep(results, x_values, x_label, title, filename):
        metrics = [
            ('mspe_fx', r'MSPE($\hat{f}_x$)'),
            ('mspe_fz', r'MSPE($\hat{f}_z$)'),
            ('corr_z_fx', r'Corr($Z$, $\hat{f}_x$)'),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, (metric, ylabel) in zip(axes, metrics):
            for name in approach_names:
                if name not in results:
                    continue
                vals = results[name][metric]
                style = '--' if name in baselines else '-'
                marker = 's' if name in baselines else 'o'
                ax.plot(x_values, vals, style, marker=marker, color=colors[name],
                        label=labels[name], linewidth=2, markersize=5)
            ax.set_xlabel(x_label)
            ax.set_ylabel(ylabel)
            if metric == 'corr_z_fx':
                ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
        axes[0].legend(fontsize=6, loc='upper left')
        fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, filename + '.pdf'), bbox_inches='tight', dpi=150)
        fig.savefig(os.path.join(OUT_DIR, filename + '.png'), bbox_inches='tight', dpi=150)
        print(f"Plot saved to {OUT_DIR}/{filename}.{{pdf,png}}")

    def aggregate_results(all_results, sweep_key, sweep_values, approach_names):
        """Aggregate raw results into bias²/var/mspe/corr per (approach, sweep_value)."""
        agg = {name: {m: [] for m in ['bias2_fx', 'var_fx', 'mspe_fx',
                                       'bias2_fz', 'var_fz', 'mspe_fz', 'corr_z_fx']}
               for name in approach_names}
        for sv in sweep_values:
            for name in approach_names:
                subset = [r for r in all_results if r[sweep_key] == sv and r['approach'] == name]
                if not subset:
                    for m in agg[name]:
                        agg[name][m].append(float('nan'))
                    continue
                fx_preds = np.stack([r['fx_hat'] for r in subset], axis=1)
                fz_preds = np.stack([r['fz_hat'] for r in subset], axis=1)
                fx_t = subset[0]['fx_target']
                fz_t = subset[0]['fz_target']
                d_fx = bias_var_decomposition(fx_preds, fx_t)
                d_fz = bias_var_decomposition(fz_preds, fz_t)
                corrs = [r['corr_z_fx'] for r in subset]
                agg[name]['bias2_fx'].append(d_fx['bias2'])
                agg[name]['var_fx'].append(d_fx['var'])
                agg[name]['mspe_fx'].append(d_fx['mspe'])
                agg[name]['bias2_fz'].append(d_fz['bias2'])
                agg[name]['var_fz'].append(d_fz['var'])
                agg[name]['mspe_fz'].append(d_fz['mspe'])
                agg[name]['corr_z_fx'].append(np.mean(corrs))
        return agg

    if args.mode == 'cv_sweep':
        # ── cv sweep (existing) ──
        results = aggregate_results(all_results, 'cv', cv_values, approach_names)

        print(f"\n{'approach':>15s} {'cv':>4s}  {'mspe_fx':>10s} {'mspe_fz':>10s} {'corr_Z':>10s}")
        print("-" * 55)
        for i, cv in enumerate(cv_values):
            for name in approach_names:
                print(f"{name:>15s} {cv:4.1f}  {results[name]['mspe_fx'][i]:10.4f} "
                      f"{results[name]['mspe_fz'][i]:10.4f} "
                      f"{results[name]['corr_z_fx'][i]:+10.4f}", flush=True)
            print()

        plot_sweep(results, cv_values, 'cv (confound strength)',
                   f'cv sweep (p={nc}, n_seeds={args.n_seeds})', 'cv_sweep')

    elif args.mode == 'n_sweep':
        # ── N sweep: fixed cv=0.8, vary N ──
        N_values = [100, 200, 400, 800, 1600]
        cv_fixed = 0.8

        print(f"N sweep: N={N_values}, cv={cv_fixed}, n_seeds={args.n_seeds}", flush=True)
        t0_n = time.time()

        all_results_n = []
        def run_seed_n(seed, N_train):
            torch.manual_seed(seed)
            (X_tr, Z_tr, y_tr), (X_te, Z_te, y_te, fx_te, fz_te) = make_dgp(
                N_train, 200, D_IN, cv_fixed, seed_train=seed, num_covariates=nc)
            fx_target = (fx_te - fx_te.mean()).squeeze().numpy()
            fz_target = (fz_te - fz_te.mean()).squeeze().numpy()
            z_test = Z_te[:, 0].numpy()
            rows = []
            for name, train_fn in APPROACHES.items():
                model = train_fn(X_tr, Z_tr, y_tr, nc)
                fx_hat, fz_hat = predict_centered(model, X_te, Z_te)
                rows.append({'seed': seed, 'N': N_train, 'approach': name,
                             'fx_hat': fx_hat, 'fz_hat': fz_hat,
                             'fx_target': fx_target, 'fz_target': fz_target,
                             'corr_z_fx': float(np.corrcoef(z_test, fx_hat)[0, 1])})
            return rows

        with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            futures = {}
            for N in N_values:
                for s in range(args.n_seeds):
                    futures[pool.submit(run_seed_n, s, N)] = (N, s)
            done = 0
            for f in as_completed(futures):
                all_results_n.extend(f.result())
                done += 1
                if done % args.n_seeds == 0:
                    print(f"  N={futures[f][0]} batch done ({done}/{len(futures)})", flush=True)

        print(f"N sweep done in {time.time()-t0_n:.1f}s\n", flush=True)

        results_n = aggregate_results(all_results_n, 'N', N_values, approach_names)

        print(f"{'approach':>15s} {'N':>6s}  {'mspe_fx':>10s} {'mspe_fz':>10s} {'corr_Z':>10s}")
        print("-" * 55)
        for i, N in enumerate(N_values):
            for name in approach_names:
                print(f"{name:>15s} {N:6d}  {results_n[name]['mspe_fx'][i]:10.4f} "
                      f"{results_n[name]['mspe_fz'][i]:10.4f} "
                      f"{results_n[name]['corr_z_fx'][i]:+10.4f}", flush=True)
            print()

        # Save results as JSON (numpy arrays → lists).
        import json
        save_data = {}
        for name in approach_names:
            save_data[name] = {k: v for k, v in results_n[name].items()}
        with open(os.path.join(OUT_DIR, 'n_sweep_data.json'), 'w') as f:
            json.dump({'N_values': N_values, 'cv': cv_fixed, 'n_seeds': args.n_seeds,
                       'results': save_data}, f, indent=2)

        plot_sweep(results_n, N_values, 'N (training size)',
                   f'N sweep (cv={cv_fixed}, p={nc}, n_seeds={args.n_seeds})', 'n_sweep')
