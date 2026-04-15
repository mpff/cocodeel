"""Smoke test: same-sample vs. split posthoc recipe on the paper's simulation settings.

For each setting and sim draw, we train:
  - `base_full`: backbone on the full N draw (paper's reference).
  - `base_half`: backbone on the first N/2 (new-recipe backbone).
  - `posthoc`: PostHocCovarNetwork(base_half).fit(half_B)  → new, split recipe.
  - `posthoc_same_sample`: PostHocCovarNetwork(base_full).fit(full) → old, same-sample recipe.

Each sim evaluates MSPE of f̂_X and f̂_X^re on a shared test draw (seed 1234, n=800).
Outputs one row per (setting, method, sim_id) with MSPE breakdowns.

Usage:
    conda activate dl-mri
    python experiments/simulation_images/3-smoke_test_sample_splitting.py [--nsim N] [--n N] [--dry-run]

All output lands in `results/simulation_images/smoke_sample_splitting.csv`.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import MSELoss, BCELoss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cocodeel.model import BaseNetwork
from cocodeel.posthoc_model import PostHocCovarNetwork
from cocodeel.trainer import covar_trainer
from cocodeel.dataset import CovarDataset
from torch.utils.data import DataLoader

from experiments.simulation_images.backbone import TrafficBackbone
from experiments.simulation_images.dataset import simulate_traffic_light_data
from experiments.simulation_images.utils import simulate_dataloaders_split


# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_N = 400          # per-setting total sample size
DEFAULT_NSIM = 10        # sims per setting
DEFAULT_DEVICE = "cuda:1"
EPOCHS = 200
PATIENCE = 12
LR = 1e-3
WEIGHT_DECAY = 1e-4
TEST_SEED = 1234
TEST_N = 800

SETTINGS = {
    # Fig 1b: Gaussian at paper default, confounded and unconfounded.
    "fig1b_gauss_bz0":
        dict(bz=0.0,  b2=1., b3=1., cv1=0.8, cv2=0.5, sdy=1., outcome_type="continuous", q=32),
    "fig1b_gauss_bz1":
        dict(bz=1.0,  b2=1., b3=1., cv1=0.8, cv2=0.5, sdy=1., outcome_type="continuous", q=32),
    # Fig 4a: q-sweep at default cv1=0.8.
    "fig4a_q4":    dict(bz=1.0, b2=1., b3=1., cv1=0.8, cv2=0.5, sdy=1., outcome_type="continuous", q=4),
    "fig4a_q32":   dict(bz=1.0, b2=1., b3=1., cv1=0.8, cv2=0.5, sdy=1., outcome_type="continuous", q=32),
    "fig4a_q256":  dict(bz=1.0, b2=1., b3=1., cv1=0.8, cv2=0.5, sdy=1., outcome_type="continuous", q=256),
    "fig4a_q1024": dict(bz=1.0, b2=1., b3=1., cv1=0.8, cv2=0.5, sdy=1., outcome_type="continuous", q=1024),
    # Fig 4b: cv1-sweep at default q=32.
    "fig4b_cv0":   dict(bz=1.0, b2=1., b3=1., cv1=0.0, cv2=0.5, sdy=1., outcome_type="continuous", q=32),
    "fig4b_cv4":   dict(bz=1.0, b2=1., b3=1., cv1=0.4, cv2=0.5, sdy=1., outcome_type="continuous", q=32),
    "fig4b_cv8":   dict(bz=1.0, b2=1., b3=1., cv1=0.8, cv2=0.5, sdy=1., outcome_type="continuous", q=32),
    # Note: Fig 3 (Bernoulli concurvity) deferred — IRLS is unstable at N=400.
    # A separate run at paper-scale N (~25k) is needed for that setting.
}


# ── Utilities ─────────────────────────────────────────────────────────────────
def make_trainer_params(outcome_type: str, device: str) -> dict:
    return {
        "device": device,
        "loss_fn": BCELoss() if outcome_type == "binary" else MSELoss(),
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,
    }


def make_model_params(q: int, outcome_type: str) -> dict:
    return {
        "backbone": TrafficBackbone,
        "backbone_params": {"out_features": q},
        "num_covariates": 1,
        "link": "logit" if outcome_type == "binary" else "identity",
    }


def fit_one_sim(sim_params: dict, q: int, device: str, seed: int):
    """Train base_full, base_half, posthoc (split), posthoc_same_sample (old).

    Returns a dict {method_name: fitted_model}.
    """
    outcome_type = sim_params["outcome_type"]
    mp = make_model_params(q, outcome_type)
    tp = make_trainer_params(outcome_type, device)

    full, half_A, half_B = simulate_dataloaders_split(sim_params, seed=seed)
    full_tr, full_va = full
    hA_tr, hA_va = half_A
    hB_tr, hB_va = half_B

    base_full = covar_trainer(BaseNetwork, mp, train_loader=full_tr, val_loader=full_va, **tp)
    base_full = base_full.center_effects(full_tr)

    base_half = covar_trainer(BaseNetwork, mp, train_loader=hA_tr, val_loader=hA_va, **tp)
    base_half = base_half.center_effects(hA_tr)

    # Binary/IRLS case needs more iterations and a looser tolerance at small N.
    # Continuous/identity case converges after one iteration, so defaults are fine.
    fit_kwargs = dict(max_iters=100, tol=1e-4) if outcome_type == "binary" else dict()

    phm_split = PostHocCovarNetwork(base_half, num_covariates=1, orthogonalize=False).to(device)
    phm_split = phm_split.fit(hB_tr, hB_va, **fit_kwargs)

    phm_same = PostHocCovarNetwork(base_full, num_covariates=1, orthogonalize=False).to(device)
    phm_same = phm_same.fit(full_tr, full_va, **fit_kwargs)

    return {
        "base_full":           base_full,
        "base_half":           base_half,
        "posthoc":             phm_split,
        "posthoc_same_sample": phm_same,
    }


def eval_on_test(models: dict, sim_params: dict, q: int, device: str) -> list[dict]:
    """Evaluate each model on a shared unconfounded test draw.

    For each model, compute MSPE on fx (direct effect) and fr (residual / X^re).
    Note: f̂_X uses the covariate-free prediction, f̂_X^re would need orth — for
    the smoke we report MSPE(f̂_X - true_fx) and MSPE(f̂_X - true_fr) as two
    targets the model may or may not be estimating.
    """
    # Test data kept identical across (sim, method) — same seed, same n.
    X_te, Z_te, y_te, fx_te, fz_te, fr_te = simulate_traffic_light_data(
        n=TEST_N, bz=sim_params["bz"], b2=sim_params["b2"], b3=sim_params["b3"],
        cv1=sim_params["cv1"], cv2=sim_params["cv2"], sdy=sim_params["sdy"],
        outcome_type=sim_params["outcome_type"], seed=TEST_SEED,
    )
    ds = CovarDataset(X_te, Z_te, y_te)
    loader = DataLoader(ds, batch_size=min(200, len(ds)), shuffle=False)

    rows = []
    for name, model in models.items():
        model.eval()
        model.to(device)
        fx_hats = []
        y_hats = []
        with torch.no_grad():
            for b in loader:
                xb = b["X"].to(device)
                zb = b["Z"].to(device)
                fx_hats.append(model.predict_fx(xb, zb).cpu())
                y_hats.append(model(xb, zb).cpu())
        fx_hat = torch.cat(fx_hats, dim=0).view(-1).numpy()
        y_hat  = torch.cat(y_hats,  dim=0).view(-1).numpy()
        fx     = fx_te.view(-1).numpy()
        fr     = fr_te.view(-1).numpy()
        y_np   = y_te.view(-1).numpy()

        rows.append(dict(
            method=name,
            mspe_fx=float(((fx_hat - fx) ** 2).mean()),
            mspe_fr=float(((fx_hat - fr) ** 2).mean()),
            mspe_y =float(((y_hat  - y_np) ** 2).mean()),
            mean_fx_hat=float(fx_hat.mean()),
            var_fx_hat=float(fx_hat.var()),
        ))
    return rows


# ── Runner ────────────────────────────────────────────────────────────────────
def run(nsim: int, n_total: int, device: str, settings: dict, out_csv: Path,
        dry_run: bool = False) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "setting", "sim_id", "method", "mspe_fx", "mspe_fr", "mspe_y",
        "mean_fx_hat", "var_fx_hat", "n_total", "outcome_type", "q", "bz", "cv1",
    ]
    first_write = not out_csv.exists()
    f = out_csv.open("a" if not first_write else "w", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if first_write:
        writer.writeheader()

    total_settings = len(settings)
    t_start = time.time()
    for idx, (setting_name, cfg) in enumerate(settings.items(), 1):
        sim_params = {k: v for k, v in cfg.items() if k != "q"}
        sim_params["n"] = n_total
        q = cfg["q"]
        print(f"\n[{datetime.datetime.now():%H:%M:%S}] ({idx}/{total_settings}) setting={setting_name}  "
              f"q={q}  bz={sim_params['bz']}  cv1={sim_params['cv1']}  outcome={sim_params['outcome_type']}",
              flush=True)

        for sim_id in range(nsim):
            t_sim = time.time()
            models = fit_one_sim(sim_params, q=q, device=device, seed=sim_id)
            rows = eval_on_test(models, sim_params, q=q, device=device)
            for r in rows:
                r.update(setting=setting_name, sim_id=sim_id, n_total=n_total,
                         outcome_type=sim_params["outcome_type"], q=q,
                         bz=sim_params["bz"], cv1=sim_params["cv1"])
                writer.writerow(r)
            f.flush()
            print(f"  sim {sim_id}: {time.time() - t_sim:.1f}s  "
                  + "  ".join(f"{r['method']}={r['mspe_fx']:.3f}" for r in rows),
                  flush=True)
            if dry_run:
                print("  [dry-run: exiting after one sim in one setting]")
                return

    print(f"\nTotal wall time: {(time.time() - t_start) / 60:.1f} min")
    f.close()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsim", type=int, default=DEFAULT_NSIM)
    ap.add_argument("--n",    type=int, default=DEFAULT_N)
    ap.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    ap.add_argument("--only-setting", type=str, default=None,
                    help="Run only the named setting (for debugging).")
    ap.add_argument("--dry-run", action="store_true",
                    help="One sim, one setting, then exit (wall-time estimate).")
    ap.add_argument("--out", type=str,
                    default=str(ROOT / "results/simulation_images/smoke_sample_splitting.csv"))
    args = ap.parse_args()

    settings = SETTINGS if args.only_setting is None else {args.only_setting: SETTINGS[args.only_setting]}
    run(nsim=args.nsim, n_total=args.n, device=args.device, settings=settings,
        out_csv=Path(args.out), dry_run=args.dry_run)
