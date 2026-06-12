"""Per-q HP search for NAM (CovarNetwork end-to-end, lam_reg=0).

Sibling to `hp_search_q.py`, which searched HPs for the post-hoc xfit
backbone (BaseNetwork). NAM uses a different model class
(CovarNetwork) trained end-to-end via
`train_covar_with_concurvity_reg`. Although NAM and BaseNetwork share
the same backbone architecture (modulo +1 param for fz), explicitly
HP-searching NAM removes any unfair advantage to the post-hoc method
in the concurvity_q figure.

Protocol mirrors hp_search_q.py:
- Single anchor: continuous, N = 6400, bz = 1, defaults.
- Vary only `lr` over {1e-3, 3e-3, 1e-2}.
- Pin wd / early_pat / sched_pat to chosen_hps.json's continuous combo.
- Per (q, lr): train NAM end-to-end on the full N. Selection is by
  MSPE(f̂_X) of NAM's `predict_fx` on the shared test draw.
- 4 workers / spawn / cuda:0.

Outputs
-------
    experiments/simulation_images/chosen_hps_q_nam.json     (committed)
    results/simulation_images/hp_search_q_nam/hp_search_q_nam.csv
        (diagnostics, untracked)

Usage:
    ~/.conda/envs/dl-mri/bin/python -u \\
        experiments/simulation_images/hp_search_q_nam.py
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.nn import MSELoss
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cocodeel.benchmarking.model import CovarNetwork
from cocodeel.dataset import CovarDataset

from experiments.simulation_images.backbone import TrafficBackbone
from experiments.simulation_images.dataset import simulate_traffic_light_data
from experiments.simulation_images.utils import simulate_dataloaders_split
from experiments.simulation_images.concurvity_methods import (
    train_covar_with_concurvity_reg,
)


# ── Search config ────────────────────────────────────────────────────────────
DEVICE     = os.environ.get("COCODEEL_DEVICE", "cuda:0")
N_WORKERS  = 4
ANCHOR_N   = 6400
SEED       = 0
TEST_SEED  = 1234
TEST_N     = 800
EPOCHS_CAP = 1000

Q_GRID  = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
LR_GRID = [1e-3, 3e-3, 1e-2]

SIM_DEFAULTS = dict(bz=1., b2=1., b3=1., cv1=0.5, cv2=0.5, sdy=1.)

HP_DEFAULT_PATH = Path(__file__).resolve().parent / "chosen_hps.json"
HP_OUT_PATH     = Path(__file__).resolve().parent / "chosen_hps_q_nam.json"
DIAG_DIR        = ROOT / "results" / "simulation_images" / "hp_search_q_nam"


def _fit_one(task):
    q, lr, wd, early_pat, sched_pat = task
    try:
        device = torch.device(DEVICE)
        torch.manual_seed(SEED + q * 1000 + int(round(np.log10(lr) * 100)))

        sim_params = dict(SIM_DEFAULTS, n=ANCHOR_N, outcome_type="continuous")
        full, _, _ = simulate_dataloaders_split(sim_params, seed=SEED)
        full_tr, full_va = full

        model_params = dict(
            backbone=TrafficBackbone,
            backbone_params={"out_features": q},
            num_covariates=1,
            link="identity",
        )
        tp = dict(
            device=device, loss_fn=MSELoss(),
            epochs=EPOCHS_CAP,
            lr=lr, weight_decay=wd,
            patience=early_pat,
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
            scheduler_kwargs={"mode": "min", "patience": sched_pat, "factor": 0.5},
            use_amp=True,
        )

        t0 = time.time()
        nam = train_covar_with_concurvity_reg(
            CovarNetwork, model_params, full_tr, full_va,
            lam_reg=0.0, **tp,
        )
        nam = nam.center_effects(full_tr)
        t_nam = time.time() - t0

        # Shared test draw.
        test_sim = dict(sim_params, n=TEST_N)
        X_te, Z_te, y_te, fx_te, *_ = simulate_traffic_light_data(
            **test_sim, seed=TEST_SEED,
        )
        loader = DataLoader(CovarDataset(X_te, Z_te, y_te),
                            batch_size=min(200, TEST_N), shuffle=False)
        nam.eval()
        fx_preds = []
        with torch.no_grad():
            for b in loader:
                fx_preds.append(nam.predict_fx(b["X"].to(device)).cpu())
        fx_hat = torch.cat(fx_preds).view(-1).numpy()
        fx_truth = fx_te.view(-1).numpy()
        mspe_fx = float(((fx_hat - fx_truth) ** 2).mean())

        return dict(q=q, lr=lr, wd=wd, early_pat=early_pat, sched_pat=sched_pat,
                    t_nam_s=t_nam, t_total_s=t_nam,
                    epochs_nam=int(getattr(nam, "best_epoch_", -1)),
                    mspe_fx=mspe_fx, status="ok")
    except Exception as e:
        return dict(q=q, lr=lr, wd=wd, early_pat=early_pat, sched_pat=sched_pat,
                    error=str(e), traceback=traceback.format_exc()[-1000:],
                    status="error")


def _choose_winner_per_q(rows, default_combo):
    """Per q, pick the lr with the lowest MSPE(f̂_X)."""
    winners = {}
    for q in Q_GRID:
        candidates = [r for r in rows
                      if r.get("q") == q and r.get("status") == "ok"]
        if not candidates:
            print(f"  q={q}: no successful runs — falling back to default lr",
                  flush=True)
            winners[str(q)] = dict(default_combo)
            continue
        best = min(candidates, key=lambda r: r["mspe_fx"])
        winners[str(q)] = dict(
            q=q, lr=best["lr"], wd=best["wd"],
            early_pat=best["early_pat"], sched_pat=best["sched_pat"],
            mspe_fx=best["mspe_fx"], t_total_s=best["t_total_s"],
            epochs_nam=best["epochs_nam"],
        )
        print(f"  q={q}: lr={best['lr']:.0e}  mspe_fx={best['mspe_fx']:.4e}  "
              f"epochs={best['epochs_nam']}  t={best['t_total_s']:.1f}s",
              flush=True)
    return winners


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default=DEVICE)
    args = ap.parse_args()

    os.environ["COCODEEL_DEVICE"] = args.device
    globals()["DEVICE"] = args.device

    if not HP_DEFAULT_PATH.exists():
        sys.exit(f"chosen_hps.json not found at {HP_DEFAULT_PATH}")
    default_combo = json.loads(HP_DEFAULT_PATH.read_text())["continuous"]
    wd        = default_combo["wd"]
    early_pat = default_combo["early_pat"]
    sched_pat = default_combo["sched_pat"]

    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [(q, lr, wd, early_pat, sched_pat) for q in Q_GRID for lr in LR_GRID]
    print("=== HP search per q (NAM) ===")
    print(f"qs={Q_GRID}\nlrs={LR_GRID}\nwd={wd}, early_pat={early_pat}, "
          f"sched_pat={sched_pat}\nanchor_N={ANCHOR_N}, device={args.device}, "
          f"workers={N_WORKERS}\n{len(tasks)} tasks", flush=True)

    t_start = time.time()
    ctx = mp.get_context("spawn")
    rows = []
    with ctx.Pool(processes=N_WORKERS) as pool:
        for i, res in enumerate(pool.imap_unordered(_fit_one, tasks), start=1):
            rows.append(res)
            tag = res.get("status", "?").upper()
            extra = (f"mspe={res.get('mspe_fx', float('nan')):.4e} "
                     f"t={res.get('t_total_s', 0.0):.1f}s"
                     if tag == "OK"
                     else f"ERR: {res.get('error', '')}")
            print(f"[{datetime.datetime.now():%H:%M:%S}] {i}/{len(tasks)}  "
                  f"q={res.get('q','?')} lr={res.get('lr','?'):.0e}  {tag}  {extra}",
                  flush=True)

    fieldnames = sorted({k for r in rows for k in r.keys()})
    with (DIAG_DIR / "hp_search_q_nam.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {DIAG_DIR / 'hp_search_q_nam.csv'}", flush=True)

    print("\nPer-q NAM winners:")
    winners = _choose_winner_per_q(rows, default_combo)
    HP_OUT_PATH.write_text(json.dumps(winners, indent=2))
    print(f"\nWrote {HP_OUT_PATH}")
    print(f"\nTotal wall: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
