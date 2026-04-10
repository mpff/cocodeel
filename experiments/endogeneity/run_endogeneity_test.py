#!/usr/bin/env python
"""Endogeneity test: compare training approaches for joint fx/fz estimation.

DGP with controlled confound strength (cv). Evaluates whether each approach
recovers the true direct effects fx and fz when the backbone is trained on
confounded data.

Multi-seed: fixed test set per cv, variable training sets. Computes bias²/var
decomposition of MSPE following cocodeel/experiments/simulation_images/utils.py.

Module layout:
    dataset/DGP:  this file (make_dgp + _generate)
    backbones:    backbones.py
    metrics:      metrics.py
    approaches:   approaches.py
    runner/CLI:   this file

Usage:
    python -m experiments.endogeneity.run_endogeneity_test \
        --num-covariates 1 --n-seeds 20 --n-workers 20
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

from experiments.endogeneity.approaches import APPROACHES, D_IN, D_OUT
from experiments.endogeneity.metrics import (
    bias_var_decomposition,
    concurvity,
    predict_centered,
)


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
# Runner: one seed, all approaches, all cv values
# ═══════════════════════════════════════════════════════════════════════════════
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

        # Concurvity diagnostic: compute on training data (property of the DGP, not the model).
        # Use a simple linear backbone to extract raw features for the concurvity measure.
        H_tr = X_tr  # for raw features; for trained backbone, would need model.backbone(X_tr)
        Z_tr_c = Z_tr - Z_tr.mean(dim=0, keepdim=True)
        H_tr_c = H_tr - H_tr.mean(dim=0, keepdim=True)
        cc = concurvity(H_tr_c, Z_tr_c)

        for name, train_fn in APPROACHES.items():
            model = train_fn(X_tr, Z_tr, y_tr, nc)
            fx_hat, fz_hat = predict_centered(model, X_te, Z_te)
            corr_z = float(np.corrcoef(z_test, fx_hat)[0, 1])
            rows.append({
                'seed': seed, 'cv': cv, 'approach': name,
                'fx_hat': fx_hat, 'fz_hat': fz_hat,
                'concurvity': cc,
                'fx_target': fx_target, 'fz_target': fz_target,
                'corr_z_fx': corr_z,
            })
    return rows


def run_seed_n(seed, N_train, cv_fixed, nc, N_test=200):
    """Variant of run_seed used by the N-sweep mode (fixed cv, variable N)."""
    torch.manual_seed(seed)
    (X_tr, Z_tr, y_tr), (X_te, Z_te, y_te, fx_te, fz_te) = make_dgp(
        N_train, N_test, D_IN, cv_fixed, seed_train=seed, num_covariates=nc)
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


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregation + plotting (endogeneity-specific, inline)
# ═══════════════════════════════════════════════════════════════════════════════
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


# Color/label palettes for plots — kept inline so the runner is self-contained.
COLORS = {
    '5_pretrained': '#000000',
    '6_posthoc':    '#888888',
    '2a_nam':       '#aec7e8',
    '2b_fast_fz':   '#1f77b4',
    '2c_precond':   '#6baed6',
    '2d_formula':   '#08306b',
    '3b_backfit':   '#d62728',
}
LABELS = {
    '5_pretrained': '5: pretrained (baseline)',
    '6_posthoc':    '6: post-hoc (current)',
    '2a_nam':       '2a: NAM (same lr)',
    '2b_fast_fz':   '2b: fast fz (oracle lr)',
    '2c_precond':   '2c: precond fz',
    '2d_formula':   '2d: formula lr (ours)',
    '3b_backfit':   '3b: SGD backfitting',
}
BASELINES = ('5_pretrained', '6_posthoc')


def plot_sweep(results, approach_names, x_values, x_label, title, filename, out_dir):
    """Plot a 1×3 panel (MSPE fx, MSPE fz, Corr(Z, fx)) as a function of sweep value."""
    import os
    import matplotlib.pyplot as plt

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
            style = '--' if name in BASELINES else '-'
            marker = 's' if name in BASELINES else 'o'
            ax.plot(x_values, vals, style, marker=marker, color=COLORS[name],
                    label=LABELS[name], linewidth=2, markersize=5)
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        if metric == 'corr_z_fx':
            ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    axes[0].legend(fontsize=6, loc='upper left')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename + '.pdf'), bbox_inches='tight', dpi=150)
    fig.savefig(os.path.join(out_dir, filename + '.png'), bbox_inches='tight', dpi=150)
    print(f"Plot saved to {out_dir}/{filename}.{{pdf,png}}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    import datetime
    import json
    import os

    import matplotlib
    matplotlib.use('Agg')

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-covariates', type=int, default=1, choices=[1, 2])
    parser.add_argument('--n-seeds', type=int, default=1)
    parser.add_argument('--n-workers', type=int, default=1)
    parser.add_argument('--mode', choices=['cv_sweep', 'n_sweep'], default='cv_sweep')
    args = parser.parse_args()
    nc = args.num_covariates
    cv_values = [0.0, 0.2, 0.4, 0.6, 0.8]
    approach_names = list(APPROACHES.keys())

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

    # ── Bias²/Var decomposition print loop ──
    results = {}
    for name in approach_names:
        results[name] = {m: [] for m in ['bias2_fx', 'var_fx', 'bias2_fz', 'var_fz',
                                          'corr_z_fx']}

    print(f"{'approach':>15s} {'cv':>4s}  {'bias2_fx':>10s} {'var_fx':>10s}  "
          f"{'bias2_fz':>10s} {'var_fz':>10s}  {'corr_Z':>10s}")
    print("-" * 80)

    for cv in cv_values:
        cc_vals = [r['concurvity'] for r in all_results
                   if r['cv'] == cv and r['approach'] == approach_names[0]]
        cc_mean = np.mean(cc_vals)
        print(f"  cv={cv:.1f}  concurvity(fx|fz)={cc_mean:.4f}", flush=True)

        for name in approach_names:
            subset = [r for r in all_results if r['cv'] == cv and r['approach'] == name]
            n_seeds = len(subset)

            fx_preds = np.stack([r['fx_hat'] for r in subset], axis=1)
            fz_preds = np.stack([r['fz_hat'] for r in subset], axis=1)
            fx_target = subset[0]['fx_target']
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
                print(f"    {name:>15s}  {dec_fx['bias2']:10.4f} {dec_fx['var']:10.4f}  "
                      f"{dec_fz['bias2']:10.4f} {dec_fz['var']:10.4f}  "
                      f"{mean_corr:+7.4f}±{np.std(corr_z_vals):.4f}", flush=True)
            else:
                print(f"    {name:>15s}  {dec_fx['bias2']:10.4f} {dec_fx['var']:10.4f}  "
                      f"{dec_fz['bias2']:10.4f} {dec_fz['var']:10.4f}  "
                      f"{mean_corr:+10.4f}", flush=True)
        print()

    # ── Dispatch to plotting for the requested mode ──
    if args.mode == 'cv_sweep':
        results = aggregate_results(all_results, 'cv', cv_values, approach_names)

        print(f"\n{'approach':>15s} {'cv':>4s}  {'mspe_fx':>10s} {'mspe_fz':>10s} {'corr_Z':>10s}")
        print("-" * 55)
        for i, cv in enumerate(cv_values):
            for name in approach_names:
                print(f"{name:>15s} {cv:4.1f}  {results[name]['mspe_fx'][i]:10.4f} "
                      f"{results[name]['mspe_fz'][i]:10.4f} "
                      f"{results[name]['corr_z_fx'][i]:+10.4f}", flush=True)
            print()

        plot_sweep(results, approach_names, cv_values, 'cv (confound strength)',
                   f'cv sweep (p={nc}, n_seeds={args.n_seeds})', 'cv_sweep', OUT_DIR)

    elif args.mode == 'n_sweep':
        N_values = [100, 200, 400, 800, 1600]
        cv_fixed = 0.8

        print(f"N sweep: N={N_values}, cv={cv_fixed}, n_seeds={args.n_seeds}", flush=True)
        t0_n = time.time()

        all_results_n = []
        with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            futures = {}
            for N in N_values:
                for s in range(args.n_seeds):
                    futures[pool.submit(run_seed_n, s, N, cv_fixed, nc)] = (N, s)
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

        save_data = {name: {k: v for k, v in results_n[name].items()}
                     for name in approach_names}
        with open(os.path.join(OUT_DIR, 'n_sweep_data.json'), 'w') as f:
            json.dump({'N_values': N_values, 'cv': cv_fixed, 'n_seeds': args.n_seeds,
                       'results': save_data}, f, indent=2)

        plot_sweep(results_n, approach_names, N_values, 'N (training size)',
                   f'N sweep (cv={cv_fixed}, p={nc}, n_seeds={args.n_seeds})', 'n_sweep',
                   OUT_DIR)


if __name__ == '__main__':
    main()
