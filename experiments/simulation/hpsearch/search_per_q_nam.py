"""Per-q learning-rate search for the NAM competitor in the concurvity_q sweep.

Sibling of search_per_q.py with the same protocol, but the model is the
end-to-end CovarNetwork (NAM, lam_reg=0) trained on the full anchor sample.
Writes chosen_hps_q_nam.json; the winning rates are hardcoded as LR_Q_NAM in
study C.

Usage:  python experiments/simulation/hpsearch/search_per_q_nam.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from cocodeel.benchmarking.model import CovarNetwork
from cocodeel.dataset import CovarDataset

from experiments.simulation.common.backbone import TrafficBackbone
from experiments.simulation.common.dgp import simulate_traffic_light_data
from experiments.simulation.common.loaders import simulate_dataloaders_split
from experiments.simulation.study_c_concurvity_benchmark import train_covar_with_concurvity_reg
from experiments.simulation.hpsearch import _grid_search


# ── search config ─────────────────────────────────────────────────────────────
DEVICE = os.environ.get("COCODEEL_DEVICE", "cuda:0")
N_WORKERS = 4
ANCHOR_N = 6400
SEED = 0
TEST_SEED = 1234
TEST_N = 800
EPOCHS_CAP = 1000

Q_GRID = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
LR_GRID = [1e-3, 3e-3, 1e-2]
# pinned to chosen_hps.json's continuous winners
WD, EARLY_PAT, SCHED_PAT = 1e-5, 6, 5

SIM_DEFAULTS = dict(bz=1., b2=1., b3=1., cv1=0.5, cv2=0.5, sdy=1.)
OUT_DIR = ROOT / "experiments/simulation/output/hp_search_q_nam"
CHOSEN_PATH = Path(__file__).resolve().parent / "chosen_hps_q_nam.json"


def fit_one(task):
    q, lr = task
    try:
        device = torch.device(DEVICE)
        torch.manual_seed(SEED + q * 1000 + int(round(np.log10(lr) * 100)))

        # data and params
        sim_params = dict(SIM_DEFAULTS, n=ANCHOR_N, outcome_type="continuous")
        full, _, _, _ = simulate_dataloaders_split(sim_params, seed=SEED)
        full_tr, full_va = full
        model_params = dict(backbone=TrafficBackbone, backbone_params={"out_features": q},
                            num_covariates=1, link="identity")
        tp = dict(device=device, loss_fn=MSELoss(), epochs=EPOCHS_CAP,
                  lr=lr, weight_decay=WD, patience=EARLY_PAT,
                  scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
                  scheduler_kwargs={"mode": "min", "patience": SCHED_PAT, "factor": 0.5},
                  use_amp=True)

        # NAM end-to-end
        t0 = time.time()
        nam = train_covar_with_concurvity_reg(
            CovarNetwork, model_params, full_tr, full_va, lam_reg=0.0, **tp)
        nam = nam.center_effects(full_tr)

        # test MSPE(f_X)
        X_te, Z_te, y_te, fx_te, *_ = simulate_traffic_light_data(
            **dict(sim_params, n=TEST_N), seed=TEST_SEED)
        loader = DataLoader(CovarDataset(X_te, Z_te, y_te), batch_size=200, shuffle=False)
        nam.eval()
        xs = []
        with torch.no_grad():
            for b in loader:
                xs.append(nam.predict_fx(b["X"].to(device)).cpu())
        fx_hat = torch.cat(xs).view(-1).numpy()
        mspe_fx = float(((fx_hat - fx_te.view(-1).numpy()) ** 2).mean())

        return dict(q=q, lr=lr, mspe_fx=mspe_fx, t_total_s=time.time() - t0, status="ok")
    except Exception as e:
        import traceback
        return dict(q=q, lr=lr, error=str(e),
                    traceback=traceback.format_exc()[-1000:], status="error")


def choose_winner_per_q(rows):
    winners = {}
    for q in Q_GRID:
        candidates = [r for r in rows if r.get("q") == q and r["status"] == "ok"]
        best = min(candidates, key=lambda r: r["mspe_fx"])
        winners[str(q)] = dict(q=q, lr=best["lr"], wd=WD,
                               early_pat=EARLY_PAT, sched_pat=SCHED_PAT,
                               mspe_fx=best["mspe_fx"], t_total_s=best["t_total_s"])
        print(f"  q={q}: lr={best['lr']:.0e}  mspe_fx={best['mspe_fx']:.4e}", flush=True)
    return winners


def main():
    tasks = [(q, lr) for q in Q_GRID for lr in LR_GRID]
    rows = _grid_search.run_pool(
        tasks, fit_one, N_WORKERS,
        describe=lambda r: (f"q={r.get('q', '?')} lr={r.get('lr', float('nan')):.0e} "
                            f"{r.get('status', '?')} mspe={r.get('mspe_fx', float('nan')):.4e}"))
    _grid_search.write_csv(rows, OUT_DIR / "hp_search_q_nam.csv")
    winners = choose_winner_per_q(rows)
    CHOSEN_PATH.write_text(json.dumps(winners, indent=2))
    print(f"Wrote {CHOSEN_PATH}", flush=True)


if __name__ == "__main__":
    main()
