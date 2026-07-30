"""Temporary side run: CIRCE (vendored original code) for the concurvity sweep.

Fits only the circe_* methods of the concurvity sweep — the vendored
Pogodin et al. pipeline via cocodeel.benchmarking.circe_adapter, released
dsprites_linear protocol, lam in CIRCE_LAMS — writing per-seed NPZs (same
schema as the main study, circe columns only) to a separate run dir.
side_aggregate_cfnet.py appends these columns to the main run's methods.

The vendored trainer trains on the first visible GPU, so the target device
is selected by restricting visibility:

Usage:  CUDA_VISIBLE_DEVICES=1 NSIM=100 python experiments/simulation/side_circe_rerun.py
"""
import contextlib
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cocodeel.benchmarking.circe_adapter import circe_fit, CirceRosterModel
from cocodeel.dataset import CovarDataset
from torch.utils.data import DataLoader

from experiments.simulation.common.dgp import simulate_traffic_light_data
from experiments.simulation.common.loaders import simulate_dataloaders_split
from experiments.simulation.common import grid_runner
from experiments.simulation.study_c_concurvity_benchmark import (
    SIM_DEFAULTS, N_GRID_CONCURVITY, CIRCE_LAMS, TEST_SEED, TEST_N,
    gather_predictions,
)


# ── run config ────────────────────────────────────────────────────────────────
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
N_WORKERS = 4
NSIM = int(os.environ.get("NSIM", "100"))
RUN_DIR = ROOT / "experiments/simulation/output/runs/study_c_side_circe"

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

    # CIRCE at three strengths, released protocol; their prints and tqdm
    # bars are silenced, LOO-selected KRR parameters are kept in the log,
    # the lam-independent heldout precompute is shared across the strengths
    models = {}
    loo_params = {}
    yz_cache = {}
    for lam in CIRCE_LAMS:
        ckpt_dir = RUN_DIR / sweep / "circe_ckpt" / key / f"seed={seed}_lam={lam:g}"
        with open(os.devnull, "w") as devnull, \
                contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            tr = circe_fit(
                full_tr.dataset.X, full_tr.dataset.Z, full_tr.dataset.y,
                full_va.dataset.X, full_va.dataset.Z, full_va.dataset.y,
                lam=lam, workdir=ckpt_dir, yz_cache=yz_cache)
        models[f"circe_{lam:g}"] = CirceRosterModel(tr, full_tr)
        loo_params[f"{lam:g}"] = dict(sigma2_y=tr.kernel_y_args["sigma2"],
                                      ridge=float(tr.model_cfg.ridge_lambda))

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
                wall_s=time.time() - t0, n_methods=len(arrays), loo=loo_params)


def _worker(task):
    return grid_runner.catch_errors(run_one, task, sweep_key_fn(task["setting"]))


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    grid_runner.write_manifest(RUN_DIR, dict(
        study="c_side_circe_vendored", device=DEVICE,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        n_workers=N_WORKERS, nsim=NSIM, test=dict(seed=TEST_SEED, n=TEST_N),
        circe_lams=CIRCE_LAMS,
        protocol="released dsprites_linear config via circe_adapter",
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
