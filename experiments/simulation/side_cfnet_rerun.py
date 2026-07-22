"""Temporary side run: CF-Net under its source protocol for the concurvity sweep.

Refits only the cfnet_* methods of the concurvity sweep with the published
protocol — equal Adam rates 2e-4, a fixed budget of 2000 optimizer
iterations, no early stopping, no gradient clipping — writing per-seed NPZs
(same schema as the main study) to a separate run dir. The main study
scripts are untouched; side_aggregate_cfnet.py splices these predictions
over the main run's cfnet columns.

Usage:  NSIM=100 python experiments/simulation/side_cfnet_rerun.py
"""
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cocodeel.model import BaseNetwork
from cocodeel.benchmarking.adversarial_trainer import adversarial_trainer
from cocodeel.dataset import CovarDataset

from experiments.simulation.common.backbone import TrafficBackbone
from experiments.simulation.common.dgp import simulate_traffic_light_data
from experiments.simulation.common.loaders import simulate_dataloaders_split
from experiments.simulation.common import grid_runner
from experiments.simulation.study_c_concurvity_benchmark import (
    SIM_DEFAULTS, N_GRID_CONCURVITY, CFNET_LAMS, Q_DEFAULT, TEST_SEED, TEST_N,
    gather_predictions,
)


# ── run config ────────────────────────────────────────────────────────────────
DEVICE = os.environ.get("COCODEEL_DEVICE", "cuda:1")
N_WORKERS = 4
NSIM = int(os.environ.get("NSIM", "100"))
CFNET_LR = 2e-4
STEP_BUDGET = 2000
RUN_DIR = ROOT / "experiments/simulation/output/runs/study_c_side_cfnet"

SWEEP = "concurvity"
SETTINGS = [dict(n=n) for n in N_GRID_CONCURVITY]
sweep_key_fn = lambda s: f"n={s['n']}"


# ── one simulation ────────────────────────────────────────────────────────────
def run_one(sweep, setting, seed):
    key = sweep_key_fn(setting)
    outdir = RUN_DIR / sweep / "preds" / key
    outdir.mkdir(parents=True, exist_ok=True)
    npz_path = outdir / f"seed={seed}.npz"
    if npz_path.exists():
        return dict(sweep=sweep, sweep_key=key, seed=seed, status="cached", wall_s=0.0)

    device = torch.device(DEVICE)
    torch.manual_seed(seed)
    t0 = time.time()

    # data (same draw as the main run: identical sim params and seed)
    sim_params = dict(SIM_DEFAULTS, n=setting["n"], outcome_type="continuous")
    full, _, _, _ = simulate_dataloaders_split(sim_params, seed=seed)
    full_tr, full_va = full
    model_params = dict(
        backbone=TrafficBackbone,
        backbone_params={"out_features": Q_DEFAULT},
        num_covariates=1,
        link="identity",
    )

    # CF-Net at three strengths, published protocol: fixed iteration budget
    epochs = math.ceil(STEP_BUDGET / len(full_tr))
    models = {}
    for lam in CFNET_LAMS:
        adv = adversarial_trainer(
            BaseNetwork, model_params, num_covariates=1,
            train_loader=full_tr, val_loader=full_va,
            device=device, loss_fn=MSELoss(), epochs=epochs,
            lr_task=CFNET_LR, lr_cp=CFNET_LR, lr_adv=CFNET_LR, lam=lam,
            patience=None, max_grad_norm=None,
        )
        models[f"cfnet_{lam:g}"] = adv.center_effects(full_tr)

    # test evaluation (identical test population to the main run)
    test_sim = dict(sim_params, n=TEST_N)
    X_te, Z_te, y_te, fx_te, fz_te, fr_te = simulate_traffic_light_data(**test_sim, seed=TEST_SEED)
    loader = DataLoader(CovarDataset(X_te, Z_te, y_te), batch_size=min(200, TEST_N), shuffle=False)
    truths = {
        "y": y_te.view(-1).numpy().astype(np.float32),
        "fx": fx_te.view(-1).numpy().astype(np.float32),
        "fr": fr_te.view(-1).numpy().astype(np.float32),
        "fz": fz_te.view(-1).numpy().astype(np.float32),
    }
    arrays = {name: gather_predictions(m, loader, device) for name, m in models.items()}
    np.savez_compressed(
        npz_path,
        methods=np.array(list(arrays.keys())),
        **{f"{name}__{eff}": arr for name, d in arrays.items() for eff, arr in d.items()},
        **{f"truth__{eff}": arr for eff, arr in truths.items()},
        setting=np.array(json.dumps(setting)),
        sim_params=np.array(json.dumps(sim_params)),
    )

    # cleanup
    del models
    torch.cuda.empty_cache()
    gc.collect()
    return dict(sweep=sweep, sweep_key=key, seed=seed, status="ok",
                wall_s=time.time() - t0, n_methods=len(arrays))


def _worker(task):
    return grid_runner.catch_errors(run_one, task, sweep_key_fn(task["setting"]))


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    grid_runner.write_manifest(RUN_DIR, dict(
        study="c_side_cfnet_source_protocol", device=DEVICE, n_workers=N_WORKERS,
        nsim=NSIM, q_default=Q_DEFAULT, test=dict(seed=TEST_SEED, n=TEST_N),
        cfnet_lr=CFNET_LR, step_budget=STEP_BUDGET, cfnet_lams=CFNET_LAMS,
        protocol="fixed budget, no early stopping, no clipping (published)",
    ))
    grid_runner.write_settings_csv(RUN_DIR, {SWEEP: SETTINGS}, {SWEEP: sweep_key_fn})

    done = grid_runner.already_done(RUN_DIR)
    tasks = [dict(sweep=SWEEP, setting=setting, seed=seed)
             for setting in SETTINGS
             for seed in range(NSIM)
             if (SWEEP, sweep_key_fn(setting), seed) not in done]
    print(f"{len(done)} sims cached, {len(tasks)} to run.", flush=True)
    grid_runner.run_grid(RUN_DIR, tasks, _worker, N_WORKERS)


if __name__ == "__main__":
    main()
