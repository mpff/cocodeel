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
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cocodeel.dataset import CovarDataset
from cocodeel.model import BaseNetwork
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


# ═══════════════════════════════════════════════════════════════════════════════
# Backbone: MLP with higher-dimensional hidden representation
# ═══════════════════════════════════════════════════════════════════════════════
class TabularMLP(nn.Module):
    """Two-hidden-layer MLP. Deliberately over-parameterised relative to the
    three-dimensional input so the backbone can exhibit concurvity — the MLP's
    analogue of the cocodeel CNN's learned feature space.
    """
    def __init__(self, in_features=3, hidden=64, out_features=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_features),
        )
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        return self.net(x)


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
    """Return {mspe_y, mspe_fx, mspe_fz, corr_fx_z} on the test set."""
    model.eval()
    if getattr(model, 'num_covariates', 0) > 0:
        y_hat = model(X_te, Z_te)
    else:
        y_hat = model(X_te)
    fx_hat = model.predict_fx(X_te).squeeze()
    fz_hat = (model.predict_fz(Z_te).squeeze()
              if hasattr(model, 'predict_fz') else torch.zeros_like(fx_hat))
    return {
        'mspe_y':    mspe(y_hat.squeeze(), y_te.squeeze()),
        'mspe_fx':   mspe(fx_hat, fx_te.squeeze()),
        'mspe_fz':   mspe(fz_hat, fz_te.squeeze()),
        'corr_fx_z': float(np.corrcoef(fx_hat.numpy(),
                                       Z_te.squeeze().numpy())[0, 1]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# One training run (baseline + posthoc) for a given (N, seed)
# ═══════════════════════════════════════════════════════════════════════════════
def run_one(N_train, seed, cv1=0.8, cv2=0.5, N_test=2000,
            hidden=64, out_features=32):
    """Train BaseNetwork + PostHocCovarNetwork on one (N, seed).
    Returns {baseline, posthoc} metrics dicts.
    """
    torch.manual_seed(seed)

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

    model_params = {
        'backbone': TabularMLP,
        'backbone_params': {'in_features': 3, 'hidden': hidden,
                            'out_features': out_features},
    }

    # ── Baseline: BaseNetwork trained without covariates ──
    base_model = covar_trainer(
        model=BaseNetwork, model_params=model_params,
        train_loader=train_loader, val_loader=val_loader,
        epochs=500, lr=1e-3, patience=20,
    ).center_effects(train_loader)
    base_metrics = evaluate(base_model, X_te, Z_te, y_te, fx_te, fz_te)

    # ── PostHoc: refit last layer with covariates via FWL + ridge ──
    phm = PostHocCovarNetwork(base_model, num_covariates=1)
    phm.fit(train_loader, val_loader, n_lambdas=20)
    ph_metrics = evaluate(phm, X_te, Z_te, y_te, fx_te, fz_te)

    return {'baseline': base_metrics, 'posthoc': ph_metrics}


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


def plot_convergence(agg, N_values, true_corr, cv1, cv2, out_dir):
    """1×4 panel figure: MSPE(y), MSPE(fx), MSPE(fz), Corr(fx, Z) vs N."""
    metrics_cfg = [
        ('mspe_y',    r'MSPE($y$)',          True,  0.0),
        ('mspe_fx',   r'MSPE($f_x$)',        True,  0.0),
        ('mspe_fz',   r'MSPE($f_z$)',        True,  0.0),
        ('corr_fx_z', r'Corr($f_x$, $Z$)',   False, true_corr),
    ]
    colors = {'baseline': '#888888', 'posthoc': '#1f77b4'}
    labels = {'baseline': 'Baseline DNN (OVB)', 'posthoc': 'PostHoc (FWL)'}

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (metric, ylabel, log_y, hline) in zip(axes, metrics_cfg):
        for method in ('baseline', 'posthoc'):
            means = np.array([m for m, _ in agg[method][metric]])
            stds  = np.array([s for _, s in agg[method][metric]])
            ax.plot(N_values, means, '-o', color=colors[method],
                    label=labels[method], linewidth=2, markersize=5)
            ax.fill_between(N_values, means - stds, means + stds,
                            color=colors[method], alpha=0.2)
        ax.axhline(hline, color='red', linestyle=':', linewidth=1,
                   label=f'true ({hline:.3f})')
        ax.set_xlabel(r'$N_\mathrm{train}$')
        ax.set_ylabel(ylabel)
        ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')
    axes[0].legend(fontsize=8, loc='upper right')
    fig.suptitle(f'Traffic tabular DGP (cv1={cv1}, cv2={cv2}): '
                 f'baseline vs PostHoc convergence', fontsize=11)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'n_sweep.{ext}'),
                    bbox_inches='tight', dpi=150)
    print(f"Plot saved to {out_dir}/n_sweep.{{pdf,png}}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-seeds', type=int, default=5)
    parser.add_argument('--N-values', type=int, nargs='+',
                        default=[100, 200, 400, 800, 1600, 3200])
    parser.add_argument('--cv1', type=float, default=0.8)
    parser.add_argument('--cv2', type=float, default=0.5)
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(out_dir, exist_ok=True)

    true_corr = true_corr_fx_z(b2=1.0, b3=1.0, cv2=args.cv2)
    print(f"True Corr(fx, Z) = {true_corr:.4f}  (b2=b3=1, cv2={args.cv2})")
    print(f"N values: {args.N_values}   seeds: {args.n_seeds}\n")

    t0 = time.time()
    results = {'baseline': [], 'posthoc': []}
    for N in args.N_values:
        t_N = time.time()
        for seed in range(args.n_seeds):
            r = run_one(N, seed, cv1=args.cv1, cv2=args.cv2)
            for method in ('baseline', 'posthoc'):
                results[method].append({'N': N, 'seed': seed, **r[method]})
        print(f"  N={N:5d} done  ({time.time()-t_N:5.1f}s, total {time.time()-t0:5.1f}s)",
              flush=True)

    # Aggregate and print.
    metrics = ('mspe_y', 'mspe_fx', 'mspe_fz', 'corr_fx_z')
    agg = aggregate(results, ('baseline', 'posthoc'), metrics, args.N_values)

    header = (f"\n{'method':>10s} {'N':>6s}  "
              f"{'MSPE(y)':>14s} {'MSPE(fx)':>14s} {'MSPE(fz)':>14s} "
              f"{'Corr(fx,Z)':>16s}")
    subhdr = (f"{'':>10s} {'':>6s}  "
              f"{'(true=0)':>14s} {'(true=0)':>14s} {'(true=0)':>14s} "
              f"{'(true=' + f'{true_corr:.3f})':>16s}")
    print(header); print(subhdr); print("-" * len(header))
    for method in ('baseline', 'posthoc'):
        for i, N in enumerate(args.N_values):
            row = [agg[method][k][i] for k in metrics]
            cells = "  ".join(f"{m:.4f}±{s:.3f}" for m, s in row)
            print(f"{method:>10s} {N:6d}  {cells}")
        print()

    plot_convergence(agg, args.N_values, true_corr, args.cv1, args.cv2, out_dir)


if __name__ == '__main__':
    main()
