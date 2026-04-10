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
import multiprocessing as mp
import os
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
    """Return {mspe_y, mspe_fx, mspe_fz, corr_fx_z} on the test set.

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
    fx_hat = model.predict_fx(X_te_d).squeeze().cpu()
    fz_hat = (model.predict_fz(Z_te_d).squeeze().cpu()
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
            hidden=64, out_features=32,
            lr=1e-3, scheduler=None, scheduler_kwargs=None,
            device='cpu'):
    """Train BaseNetwork + PostHocCovarNetwork on one (N, seed).

    Args:
        lr: learning rate for the backbone training step.
        scheduler: scheduler class (not instance) forwarded to covar_trainer.
            None triggers covar_trainer's default ReduceLROnPlateau.
        scheduler_kwargs: kwargs dict for the scheduler class.
        device: 'cpu' or 'cuda'. Passed through to covar_trainer; evaluate()
            moves test tensors to the model's device automatically.

    Returns:
        {'baseline': metrics, 'posthoc': metrics}
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

    mlp_params = {
        'backbone': TabularMLP,
        'backbone_params': {'in_features': 3, 'hidden': hidden,
                            'out_features': out_features},
    }

    # ── Baseline: BaseNetwork with MLP trained without covariates ──
    base_model = covar_trainer(
        model=BaseNetwork, model_params=mlp_params,
        train_loader=train_loader, val_loader=val_loader,
        epochs=500, lr=lr, patience=20,
        scheduler=scheduler, scheduler_kwargs=scheduler_kwargs,
        device=device,
    ).center_effects(train_loader)
    base_metrics = evaluate(base_model, X_te, Z_te, y_te, fx_te, fz_te)
    base_val_loss = float(base_model.val_losses_[base_model.best_epoch_])

    # ── PostHoc: refit last layer of the trained MLP with FWL+ridge ──
    phm = PostHocCovarNetwork(base_model, num_covariates=1).to(device)
    phm.fit(train_loader, val_loader, n_lambdas=20)
    ph_metrics = evaluate(phm, X_te, Z_te, y_te, fx_te, fz_te)

    # ── Oracle: linear (identity backbone) + PostHoc FWL+ridge ──
    # No MLP training; PostHoc.fit does the ridge regression directly on H=X.
    oracle_base = BaseNetwork(
        backbone=LinearBackbone,
        backbone_params={'in_features': 3},
        num_covariates=0, link='identity',
    ).to(device)
    oracle = PostHocCovarNetwork(oracle_base, num_covariates=1).to(device)
    oracle.fit(train_loader, val_loader, n_lambdas=20)
    oracle_metrics = evaluate(oracle, X_te, Z_te, y_te, fx_te, fz_te)

    base_metrics['val_loss']   = base_val_loss
    ph_metrics['val_loss']     = base_val_loss  # posthoc reuses base's training
    oracle_metrics['val_loss'] = float('nan')   # no val-loss concept for the oracle
    return {'baseline': base_metrics, 'posthoc': ph_metrics, 'oracle': oracle_metrics}


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


METHODS = ('baseline', 'posthoc', 'oracle')
COLORS = {'baseline': '#888888', 'posthoc': '#1f77b4', 'oracle': '#2ca02c'}
LABELS = {'baseline': 'Baseline (MLP)',
          'posthoc':  'PostHoc (MLP + FWL)',
          'oracle':   'Oracle (Linear + FWL)'}


def plot_convergence(agg, N_values, true_corr, cv1, cv2, sdy, out_dir):
    """1×4 panel figure: MSPE(y), MSPE(fx), MSPE(fz), Corr(fx, Z) vs N.

    Reference lines:
      MSPE(y)    → sdy² (Bayes-optimal MSPE under the additive DGP)
      MSPE(fx)   → 0
      MSPE(fz)   → 0
      Corr(fx,Z) → analytic true value (e.g. 0.408 at cv2=0.5, b2=b3=1)

    X-axis ticks are set explicitly to the swept N values (no auto log-ticks).
    """
    metrics_cfg = [
        ('mspe_y',    r'MSPE($y$)',          True,  sdy**2),
        ('mspe_fx',   r'MSPE($f_x$)',        True,  0.0),
        ('mspe_fz',   r'MSPE($f_z$)',        True,  0.0),
        ('corr_fx_z', r'Corr($f_x$, $Z$)',   False, true_corr),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    for ax, (metric, ylabel, log_y, hline) in zip(axes, metrics_cfg):
        for method in METHODS:
            means = np.array([m for m, _ in agg[method][metric]])
            stds  = np.array([s for _, s in agg[method][metric]])
            ax.plot(N_values, means, '-o', color=COLORS[method],
                    label=LABELS[method], linewidth=2, markersize=5)
            ax.fill_between(N_values, means - stds, means + stds,
                            color=COLORS[method], alpha=0.2)
        ax.axhline(hline, color='red', linestyle=':', linewidth=1,
                   label=f'true ({hline:.3f})')
        ax.set_xlabel(r'$N_\mathrm{train}$')
        ax.set_ylabel(ylabel)
        ax.set_xscale('log')
        ax.set_xticks(N_values)
        ax.set_xticklabels([str(n) for n in N_values], rotation=30, fontsize=8)
        ax.minorticks_off()  # suppress default minor log ticks between our explicit ones
        if log_y:
            ax.set_yscale('log')
    axes[0].legend(fontsize=8, loc='upper right')
    fig.suptitle(f'Traffic tabular DGP (cv1={cv1}, cv2={cv2}): '
                 f'baseline vs PostHoc vs Oracle convergence', fontsize=11)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'n_sweep.{ext}'),
                    bbox_inches='tight', dpi=150)
    print(f"Plot saved to {out_dir}/n_sweep.{{pdf,png}}")


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
def n_sweep(args, device, lr, scheduler_name, out_dir):
    sdy = 1.0  # DGP noise std; Bayes-optimal MSPE(y) is sdy²
    true_corr = true_corr_fx_z(b2=1.0, b3=1.0, cv2=args.cv2)
    print(f"True Corr(fx, Z) = {true_corr:.4f}  "
          f"Bayes MSPE(y) = sdy² = {sdy**2:.4f}")
    print(f"N values: {args.N_values}   seeds: {args.n_seeds}   "
          f"device={device}  lr={lr:.0e}  sched={scheduler_name}\n", flush=True)

    sched_cls, sched_kw = resolve_scheduler(scheduler_name)

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
                    device=device,
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

    plot_convergence(agg, args.N_values, true_corr, args.cv1, args.cv2,
                     sdy=sdy, out_dir=out_dir)


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
                        default=[100, 200, 400, 800, 1600, 3200, 6400, 12800])
    parser.add_argument('--cv1', type=float, default=0.8)
    parser.add_argument('--cv2', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--scheduler', choices=['plateau', 'cosine', 'step'],
                        default='plateau')
    parser.add_argument('--device', default='auto',
                        help='"auto" picks cuda if available, else cpu')
    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(out_dir, exist_ok=True)

    if args.mode == 'hp_search':
        hp_search(device=device, cv1=args.cv1, cv2=args.cv2)
    else:
        n_sweep(args, device=device, lr=args.lr, scheduler_name=args.scheduler,
                out_dir=out_dir)


if __name__ == '__main__':
    main()
