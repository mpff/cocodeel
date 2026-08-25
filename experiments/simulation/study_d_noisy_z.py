"""Study D — refitting on a noisily measured control variable.

The DGP is held fixed at bz=1 (X and y are always generated from the true Z);
only the control handed to the estimators is corrupted, mirroring measurement
error: Z_tilde = (1-a) Z + a eps with eps ~ U([0,1]) iid, so Z_tilde stays in
[0,1] with E[Z_tilde] = 1/2 and reliability Corr(Z, Z_tilde)^2 =
(1-a)^2 / ((1-a)^2 + a^2). Methods mirror the Fig-1 bz sweep: refit and
refit_orth (2-fold cross-fit) plus the full-sample baselines base
(uncontrolled DNN, blind to Z_tilde) and posthoc_web (Weber). Endpoints anchor
to known results: a=0 reproduces study A at bz=1, a=1 controls for pure noise
and must match the uncontrolled DNN.

Usage:  NSIM=5 N_GRID=1600,12800 python experiments/simulation/study_d_noisy_z.py
"""
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cocodeel.model import BaseNetwork
from cocodeel.refit_model import RefitCovarNetwork
from cocodeel.crossfit import CrossFitEnsemble
from cocodeel.benchmarking.posthoc_model import PostHocOrthNetwork
from cocodeel.trainer import covar_trainer
from cocodeel.dataset import CovarDataset

from experiments.simulation.common.backbone import TrafficBackbone
from experiments.simulation.common.dgp import simulate_traffic_light_data
from experiments.simulation.common.loaders import simulate_dataloaders_split
from experiments.simulation.common import grid_runner

from experiments.simulation.study_a_linear_consistency import (
    HP, trainer_params, save_npz,
)


# ── run config ────────────────────────────────────────────────────────────────
DEVICE = os.environ.get("COCODEEL_DEVICE", "cuda:0")
N_WORKERS = 4
NSIM = int(os.environ.get("NSIM", "50"))
Q_DEFAULT = 32
TEST_SEED = 1234
TEST_N = 800
RUN_DIR = ROOT / "experiments/simulation/output/runs/study_d"

# ── sweep grids ───────────────────────────────────────────────────────────────
N_GRID = [400, 800, 1600, 3200, 6400, 12800, 25600]
if os.environ.get("N_GRID"):
    N_GRID = [int(v) for v in os.environ["N_GRID"].split(",")]
A_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

SWEEPS = {
    "increasing_a": dict(
        outcome_type="continuous",
        sim_defaults=dict(bz=1., b2=1., b3=1., cv1=0.5, cv2=0.5, sdy=1.),
        settings=[dict(n=n, a=a) for n in N_GRID for a in A_GRID],
        sweep_key_fn=lambda s: f"n={s['n']}_a={s['a']}",
    ),
}


# ── one simulation ────────────────────────────────────────────────────────────
def run_one(sweep, setting, seed):
    cfg = SWEEPS[sweep]
    sweep_key = cfg["sweep_key_fn"](setting)
    outdir = RUN_DIR / sweep / "preds" / sweep_key
    outdir.mkdir(parents=True, exist_ok=True)
    npz_path = outdir / f"seed={seed}.npz"
    if npz_path.exists():
        return dict(sweep=sweep, sweep_key=sweep_key, seed=seed, status="cached", wall_s=0.0)

    device = torch.device(DEVICE)
    torch.manual_seed(seed)
    t0 = time.time()

    # data with corrupted control
    # a fresh Generator seeded k replays the global stream seeded k, and the
    # DGP draws Z first — equal seeds would make eps identical to Z and the
    # corruption a no-op. Offsets keep the noise streams disjoint from the
    # DGP seeds ({0..NSIM-1} and TEST_SEED); the test stream is separate so
    # the test corruption is identical across n and a.
    a = setting["a"]
    noise_gen = torch.Generator().manual_seed(100_000 + seed)
    test_noise_gen = torch.Generator().manual_seed(200_000 + seed)

    def corrupt(Z, gen=noise_gen):
        return (1 - a) * Z + a * torch.rand(Z.shape, generator=gen)

    sim_params = dict(cfg["sim_defaults"])
    sim_params["n"] = setting["n"]
    sim_params["outcome_type"] = cfg["outcome_type"]
    full, half_A, half_B, pooled = simulate_dataloaders_split(
        sim_params, seed=seed, covar_transform=corrupt)
    full_tr, full_va = full
    hA_tr, hA_va = half_A
    hB_tr, hB_va = half_B

    # backbones
    model_params = dict(
        backbone=TrafficBackbone,
        backbone_params={"out_features": Q_DEFAULT},
        num_covariates=1,
        link="identity",
    )
    tp = dict(trainer_params(cfg["outcome_type"]), device=device)
    base_A = covar_trainer(BaseNetwork, model_params, train_loader=hA_tr, val_loader=hA_va, **tp)
    base_A = base_A.center_effects(hA_tr)
    base_B = covar_trainer(BaseNetwork, model_params, train_loader=hB_tr, val_loader=hB_va, **tp)
    base_B = base_B.center_effects(hB_tr)

    # refit variants (2-fold cross-fit)
    models = {}
    for name, orth in [("refit", False), ("refit_orth", True)]:
        m_AB = RefitCovarNetwork(base_A, num_covariates=1, orthogonalize=orth).to(device)
        m_AB = m_AB.fit(hB_tr, hB_va)
        m_BA = RefitCovarNetwork(base_B, num_covariates=1, orthogonalize=orth).to(device)
        m_BA = m_BA.fit(hA_tr, hA_va)
        models[name] = CrossFitEnsemble([m_AB, m_BA]).recenter(pooled)

    # full-sample baselines
    base_full = covar_trainer(BaseNetwork, model_params, train_loader=full_tr, val_loader=full_va, **tp)
    base_full = base_full.center_effects(full_tr)
    models["base"] = base_full
    web = PostHocOrthNetwork(base_full, num_covariates=1).to(device)
    models["posthoc_web"] = web.fit(full_tr, full_va)

    # test evaluation: truths from the true Z, predictions from a fresh corruption
    test_sim = dict(sim_params, n=TEST_N)
    X_te, Z_te, y_te, fx_te, fz_te, fr_te = simulate_traffic_light_data(**test_sim, seed=TEST_SEED)
    Z_te_tilde = corrupt(Z_te, gen=test_noise_gen)
    loader = DataLoader(CovarDataset(X_te, Z_te_tilde, y_te), batch_size=min(200, TEST_N), shuffle=False)
    truths = {
        "y": y_te.view(-1).numpy().astype(np.float32),
        "fx": fx_te.view(-1).numpy().astype(np.float32),
        "fr": fr_te.view(-1).numpy().astype(np.float32),
        "fz": fz_te.view(-1).numpy().astype(np.float32),
    }
    n_methods = save_npz(npz_path, models, loader, truths, device, setting, sim_params)

    # cleanup
    del models, base_A, base_B, base_full
    torch.cuda.empty_cache()
    gc.collect()
    return dict(sweep=sweep, sweep_key=sweep_key, seed=seed, status="ok",
                wall_s=time.time() - t0, n_methods=n_methods)


def _worker(task):
    sweep_key = SWEEPS[task["sweep"]]["sweep_key_fn"](task["setting"])
    return grid_runner.catch_errors(run_one, task, sweep_key)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    grid_runner.write_manifest(RUN_DIR, dict(
        study="d_noisy_z", device=DEVICE, n_workers=N_WORKERS,
        nsim=NSIM, q_default=Q_DEFAULT, test=dict(seed=TEST_SEED, n=TEST_N), hp=HP,
        sweeps={s: len(c["settings"]) for s, c in SWEEPS.items()},
    ))
    grid_runner.write_settings_csv(RUN_DIR, {s: c["settings"] for s, c in SWEEPS.items()},
                                   {s: c["sweep_key_fn"] for s, c in SWEEPS.items()})

    done = grid_runner.already_done(RUN_DIR)
    tasks = [dict(sweep=sweep, setting=setting, seed=seed)
             for sweep, cfg in SWEEPS.items()
             for setting in cfg["settings"]
             for seed in range(NSIM)
             if (sweep, cfg["sweep_key_fn"](setting), seed) not in done]
    print(f"{len(done)} sims cached, {len(tasks)} to run.", flush=True)
    grid_runner.run_grid(RUN_DIR, tasks, _worker, N_WORKERS)


if __name__ == "__main__":
    main()
