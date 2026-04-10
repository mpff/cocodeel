#!/usr/bin/env python
"""Replication of Siems et al. (2023) Toy Example 1.

DGP: Y = 1·X₁ + 0·X₂ (X₂ has NO effect on Y).
Three correlation settings: independent, Corr(X₁,X₂)=0.9, X₁=X₂.
NAM with two MLPs: f₁(X₁) + f₂(X₂).

Measures: Corr(f₁(X₁), Y) and Corr(f₂(X₂), Y).
Under concurvity (correlated case), f₂ absorbs signal from f₁.

Reference: Siems et al., "Curve Your Enthusiasm", NeurIPS 2023, Section 4.1.
"""
import argparse
import copy  # noqa: F811
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════════════
# DGP (exact Siems Toy Example 1)
# ═══════════════════════════════════════════════════════════════════════════════
def make_data(N, corr, b2=0.0, seed=42):
    """Generate Y = 1·X₁ + b2·X₂ with controlled Corr(X₁, X₂).

    Args:
        corr: float in [0, 1]. 0 = independent, 0.9 = strongly correlated, 1.0 = X₁=X₂.
        b2: coefficient on X₂. 0.0 = original Siems (no X₂ effect),
            >0 = X₂ has a direct effect (our setting).

    Returns:
        X1, X2, Y as (N,1) tensors.
    """
    rng = np.random.default_rng(seed)

    X1 = torch.tensor(rng.uniform(-1, 1, (N, 1)), dtype=torch.float32)

    if corr >= 1.0:
        X2 = X1.clone()
    elif corr <= 0.0:
        X2 = torch.tensor(rng.uniform(-1, 1, (N, 1)), dtype=torch.float32)
    else:
        noise = torch.tensor(rng.uniform(-1, 1, (N, 1)), dtype=torch.float32)
        X2 = corr * X1 + np.sqrt(1 - corr**2) * noise

    Y = 1.0 * X1 + b2 * X2
    return X1, X2, Y


# ═══════════════════════════════════════════════════════════════════════════════
# NAM: two independent MLPs
# ═══════════════════════════════════════════════════════════════════════════════
class FeatureNet(nn.Module):
    """Single-feature MLP: R → R."""
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


class NAM(nn.Module):
    """Neural Additive Model: Y = f₁(X₁) + f₂(X₂) + β₀."""
    def __init__(self, hidden=64):
        super().__init__()
        self.f1 = FeatureNet(hidden)
        self.f2 = FeatureNet(hidden)
        self.intercept = nn.Parameter(torch.zeros(1))

    def forward(self, X1, X2):
        return self.intercept + self.f1(X1) + self.f2(X2)


# ═══════════════════════════════════════════════════════════════════════════════
# Siems concurvity regularizer: R⊥ = |Corr(f₁(X₁), f₂(X₂))|
# ═══════════════════════════════════════════════════════════════════════════════
def concurvity_penalty(f1_out, f2_out):
    """Pairwise |Corr| between two component outputs. Siems eq. (2)."""
    f1_c = f1_out - f1_out.mean()
    f2_c = f2_out - f2_out.mean()
    corr = (f1_c * f2_c).sum() / (f1_c.norm() * f2_c.norm() + 1e-8)
    return corr.abs()


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════
def train_nam(X1, X2, Y, method='standard', epochs=200, lr=0.01, lam_reg=0.0,
              lr_f2=None, batch_size=64, patience=20):
    """Train NAM with specified method, batching, and early stopping.

    Args:
        method: 'standard' | 'siems' | 'fast_f2'
        lam_reg: regularization strength for Siems penalty.
        lr_f2: learning rate for f₂ in fast_f2 method.
    """
    model = NAM(hidden=32)
    loss_fn = nn.MSELoss()
    N = X1.shape[0]

    if method == 'fast_f2' and lr_f2 is not None:
        optimizer = torch.optim.Adam([
            {'params': [*model.f1.parameters(), model.intercept], 'lr': lr, 'weight_decay': 1e-4},
            {'params': model.f2.parameters(), 'lr': lr_f2, 'weight_decay': 1e-4},
        ])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_loss = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    pctr = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(N)
        epoch_loss = 0; nb = 0
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            optimizer.zero_grad()
            f1_out = model.f1(X1[idx])
            f2_out = model.f2(X2[idx])
            pred = model.intercept + f1_out + f2_out
            loss = loss_fn(pred, Y[idx])

            if method == 'siems' and lam_reg > 0:
                loss = loss + lam_reg * concurvity_penalty(f1_out, f2_out)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item(); nb += 1

        epoch_loss /= nb
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = copy.deepcopy(model.state_dict())
            pctr = 0
        else:
            pctr += 1
            if pctr >= patience:
                break

    model.load_state_dict(best_state)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, X1, X2, Y):
    """Compute Corr(f₁(X₁), Y), Corr(f₂(X₂), Y), and Corr(f₁, f₂)."""
    f1 = model.f1(X1).squeeze().numpy()
    f2 = model.f2(X2).squeeze().numpy()
    y = Y.squeeze().numpy()
    return {
        'corr_f1_y': float(np.corrcoef(f1, y)[0, 1]),
        'corr_f2_y': float(np.corrcoef(f2, y)[0, 1]),
        'corr_f1_f2': float(np.corrcoef(f1, f2)[0, 1]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════
def run_seed(seed, corr_values, b2=0.0, N=1000):
    """Run all methods for one seed across correlation values."""
    torch.manual_seed(seed)
    rows = []

    methods = {
        'standard':      dict(method='standard'),
        'siems_0.1':     dict(method='siems', lam_reg=0.1),
        'siems_1.0':     dict(method='siems', lam_reg=1.0),
        'siems_10':      dict(method='siems', lam_reg=10.0),
        'fast_f2_0.1':   dict(method='fast_f2', lr_f2=0.1),
    }

    for corr in corr_values:
        X1, X2, Y = make_data(N, corr, b2=b2, seed=seed)

        for name, kwargs in methods.items():
            model = train_nam(X1, X2, Y, **kwargs)
            res = evaluate(model, X1, X2, Y)
            res['seed'] = seed
            res['corr_input'] = corr
            res['method'] = name
            res['b2'] = b2
            rows.append(res)

    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--n-seeds', type=int, default=40)
    parser.add_argument('--n-workers', type=int, default=20)
    parser.add_argument('--N', type=int, default=1000)
    args = parser.parse_args()

    OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(OUT_DIR, exist_ok=True)

    corr_values = [0.0, 0.3, 0.6, 0.9, 1.0]
    b2_values = [0.0, 0.5, 1.0]  # sweep X₂ effect strength
    method_names = ['standard', 'siems_0.1', 'siems_1.0', 'siems_10', 'fast_f2_0.1']

    colors = {
        'standard':    '#888888',
        'siems_0.1':   '#aec7e8',
        'siems_1.0':   '#1f77b4',
        'siems_10':    '#08306b',
        'fast_f2_0.1': '#d62728',
    }
    labels = {
        'standard':    'NAM (no reg)',
        'siems_0.1':   'Siems λ=0.1',
        'siems_1.0':   'Siems λ=1',
        'siems_10':    'Siems λ=10',
        'fast_f2_0.1': 'fast f₂ (ours)',
    }

    print(f"Siems replication: N={args.N}, n_seeds={args.n_seeds}, b2={b2_values}",
          flush=True)
    t0 = time.time()

    all_results = []
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        futures = {}
        for b2 in b2_values:
            for s in range(args.n_seeds):
                f = pool.submit(run_seed, s, corr_values, b2=b2, N=args.N)
                futures[f] = (b2, s)
        for f in as_completed(futures):
            all_results.extend(f.result())
    print(f"Done in {time.time()-t0:.1f}s\n", flush=True)

    # ── Print per b2 ──
    for b2 in b2_values:
        print(f"=== b2={b2} (Y = 1·X₁ + {b2}·X₂) ===")
        print(f"{'method':>15s} {'corr_in':>8s}  {'Corr(f1,Y)':>12s} {'Corr(f2,Y)':>12s} {'Corr(f1,f2)':>12s}")
        print("-" * 70)
        for corr in corr_values:
            for name in method_names:
                subset = [r for r in all_results
                          if r['corr_input'] == corr and r['method'] == name and r['b2'] == b2]
                c1y = [r['corr_f1_y'] for r in subset]
                c2y = [r['corr_f2_y'] for r in subset]
                c12 = [r['corr_f1_f2'] for r in subset]
                print(f"{name:>15s} {corr:8.1f}  "
                      f"{np.nanmean(c1y):+.4f}±{np.nanstd(c1y):.4f} "
                      f"{np.nanmean(c2y):+.4f}±{np.nanstd(c2y):.4f} "
                      f"{np.nanmean(c12):+.4f}±{np.nanstd(c12):.4f}", flush=True)
            print()

    # ── Plot: one row per b2, 3 columns ──
    metrics = [
        ('corr_f1_y', r'Corr($f_1(X_1)$, $Y$)'),
        ('corr_f2_y', r'Corr($f_2(X_2)$, $Y$)'),
        ('corr_f1_f2', r'Corr($f_1$, $f_2$)'),
    ]

    fig, axes = plt.subplots(len(b2_values), 3, figsize=(14, 4 * len(b2_values)))

    for row, b2 in enumerate(b2_values):
        for col, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, col] if len(b2_values) > 1 else axes[col]
            for name in method_names:
                vals = []
                for corr in corr_values:
                    subset = [r for r in all_results
                              if r['corr_input'] == corr and r['method'] == name and r['b2'] == b2]
                    vals.append(np.nanmean([r[metric] for r in subset]))
                style = '--' if name == 'standard' else '-'
                ax.plot(corr_values, vals, style, marker='o', color=colors[name],
                        label=labels[name], linewidth=2, markersize=4)
            ax.set_xlabel(r'Corr($X_1$, $X_2$)')
            ax.set_ylabel(ylabel)
            ax.set_xticks(corr_values)
            ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
            if row == 0 and col == 0:
                ax.legend(fontsize=6)
            ax.set_title(f'$b_2={b2}$' if col == 1 else '', fontsize=10)
        # Row label
        ax_left = axes[row, 0] if len(b2_values) > 1 else axes[0]
        ax_left.annotate(f'$Y = X_1 + {b2}X_2$', xy=(0, 0.5), xytext=(-0.35, 0.5),
                         xycoords='axes fraction', textcoords='axes fraction',
                         fontsize=10, ha='center', va='center', rotation=90)

    fig.suptitle(f'Siems Toy Example (N={args.N}, {args.n_seeds} seeds)', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'siems_replication.pdf'), bbox_inches='tight', dpi=150)
    fig.savefig(os.path.join(OUT_DIR, 'siems_replication.png'), bbox_inches='tight', dpi=150)
    print(f"Plot saved to {OUT_DIR}/siems_replication.{{pdf,png}}")
