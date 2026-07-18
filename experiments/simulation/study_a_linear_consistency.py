"""Study A — consistency of the cross-fitted refit for linear f_Z.

Sweeps: increasing_bz (Fig 1), binary_increasing_bz (Fig 3), and the
adversarial settings increasing_q / increasing_cv / increasing_p (Fig 4).
Methods: refit and refit_orth (2-fold cross-fit); the bz sweeps add the
full-sample baselines base (uncontrolled DNN) and posthoc_web (Weber).

Usage:  NSIM=5 python experiments/simulation/study_a_linear_consistency.py
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import MSELoss, BCELoss
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


# ── run config ────────────────────────────────────────────────────────────────
DEVICE = os.environ.get("COCODEEL_DEVICE", "cuda:0")
N_WORKERS = 4
NSIM = int(os.environ.get("NSIM", "50"))
Q_DEFAULT = 32
TEST_SEED = 1234
TEST_N = 800
EPOCHS_CAP = 1000
RUN_DIR = ROOT / "experiments/simulation/output/runs/study_a"

# selected by hpsearch/search_default.py (hpsearch/chosen_hps.json)
HP = {
    "continuous": dict(lr=3e-3, wd=1e-5, early_pat=6, sched_pat=5),
    "binary":     dict(lr=3e-3, wd=1e-5, early_pat=6, sched_pat=3),
}

# ── sweep grids ───────────────────────────────────────────────────────────────
N_GRID = [400, 800, 1600, 3200, 6400, 12800, 25600]
BZ_GRID = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
Q_GRID = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
CV_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
P_GRID = [1, 2, 4, 8, 16]

SWEEPS = {
    "increasing_bz": dict(
        outcome_type="continuous",
        sim_defaults=dict(b2=1., b3=1., cv1=0.5, cv2=0.5, sdy=1.),
        settings=[dict(n=n, bz=bz) for n in N_GRID for bz in BZ_GRID],
        sweep_key_fn=lambda s: f"n={s['n']}_bz={s['bz']}",
        full_baselines=True,
    ),
    "binary_increasing_bz": dict(
        outcome_type="binary",
        sim_defaults=dict(b2=1., b3=1., cv1=0.5, cv2=0.5, sdy=1.),
        settings=[dict(n=n, bz=bz) for n in N_GRID for bz in BZ_GRID],
        sweep_key_fn=lambda s: f"n={s['n']}_bz={s['bz']}",
        full_baselines=True,
        fit_kwargs=dict(max_iters=25),
    ),
    "increasing_q": dict(
        outcome_type="continuous",
        sim_defaults=dict(bz=1., b2=1., b3=1., cv1=0.5, cv2=0.5, sdy=1.),
        settings=[dict(n=n, q=q) for n in N_GRID for q in Q_GRID],
        sweep_key_fn=lambda s: f"n={s['n']}_q={s['q']}",
        full_baselines=False,
    ),
    "increasing_cv": dict(
        outcome_type="continuous",
        sim_defaults=dict(bz=1., b2=1., b3=1., cv2=0.5, sdy=1.),
        settings=[dict(n=n, cv1=cv) for n in N_GRID for cv in CV_GRID],
        sweep_key_fn=lambda s: f"n={s['n']}_cv1={s['cv1']}",
        full_baselines=False,
    ),
    "increasing_p": dict(
        outcome_type="continuous",
        sim_defaults=dict(bz=1., b2=1., b3=1., cv1=0.5, cv2=0.5, sdy=1.),
        settings=[dict(n=n, n_covars=p) for n in N_GRID for p in P_GRID],
        sweep_key_fn=lambda s: f"n={s['n']}_p={s['n_covars']}",
        full_baselines=False,
    ),
}


# ── shared fitting pieces ─────────────────────────────────────────────────────
def trainer_params(outcome_type):
    h = HP[outcome_type]
    return dict(
        device=torch.device(DEVICE),
        loss_fn=BCELoss() if outcome_type == "binary" else MSELoss(),
        epochs=EPOCHS_CAP,
        lr=h["lr"],
        weight_decay=h["wd"],
        patience=h["early_pat"],
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_kwargs={"mode": "min", "patience": h["sched_pat"], "factor": 0.5},
        # BCELoss refuses bf16 autocast; MSE benefits from it
        use_amp=(outcome_type != "binary"),
    )


def gather_predictions(model, loader, device):
    """Collect y/fx/fz predictions; fr duplicates fx — a model's image effect targets fx or fr depending on its orthogonalization."""
    model.eval()
    ys, fxs, fzs = [], [], []
    with torch.no_grad():
        for b in loader:
            x = b["X"].to(device)
            z = b["Z"].to(device)
            y_hat = model(x, z) if getattr(model, "num_covariates", 0) > 0 else model(x)
            ys.append(y_hat.cpu())
            fxs.append(model.predict_fx(x, z).cpu())
            fzs.append(model.predict_fz(z).cpu())
    fx_arr = torch.cat(fxs).view(-1).numpy().astype(np.float32)
    return {
        "y": torch.cat(ys).view(-1).numpy().astype(np.float32),
        "fx": fx_arr,
        "fr": fx_arr,
        "fz": torch.cat(fzs).view(-1).numpy().astype(np.float32),
    }


def save_npz(npz_path, models, loader, truths, device, setting, sim_params):
    arrays = {name: gather_predictions(m, loader, device) for name, m in models.items()}
    np.savez_compressed(
        npz_path,
        methods=np.array(list(arrays.keys())),
        **{f"{name}__{eff}": arr for name, d in arrays.items() for eff, arr in d.items()},
        **{f"truth__{eff}": arr for eff, arr in truths.items()},
        setting=np.array(json.dumps(setting)),
        sim_params=np.array(json.dumps(sim_params)),
    )
    return len(arrays)


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

    # data
    sim_params = dict(cfg["sim_defaults"])
    sim_params.update({k: v for k, v in setting.items() if k != "q"})
    sim_params["outcome_type"] = cfg["outcome_type"]
    full, half_A, half_B, pooled = simulate_dataloaders_split(sim_params, seed=seed)
    full_tr, full_va = full
    hA_tr, hA_va = half_A
    hB_tr, hB_va = half_B

    # backbones
    q = setting.get("q", Q_DEFAULT)
    p = setting.get("n_covars", 1)
    model_params = dict(
        backbone=TrafficBackbone,
        backbone_params={"out_features": q},
        num_covariates=p,
        link=("logit" if cfg["outcome_type"] == "binary" else "identity"),
    )
    tp = trainer_params(cfg["outcome_type"])
    base_A = covar_trainer(BaseNetwork, model_params, train_loader=hA_tr, val_loader=hA_va, **tp)
    base_A = base_A.center_effects(hA_tr)
    base_B = covar_trainer(BaseNetwork, model_params, train_loader=hB_tr, val_loader=hB_va, **tp)
    base_B = base_B.center_effects(hB_tr)

    # refit variants (2-fold cross-fit)
    fit_kwargs = cfg.get("fit_kwargs", {})
    models = {}
    for name, orth in [("refit", False), ("refit_orth", True)]:
        m_AB = RefitCovarNetwork(base_A, num_covariates=p, orthogonalize=orth).to(device)
        m_AB = m_AB.fit(hB_tr, hB_va, **fit_kwargs)
        m_BA = RefitCovarNetwork(base_B, num_covariates=p, orthogonalize=orth).to(device)
        m_BA = m_BA.fit(hA_tr, hA_va, **fit_kwargs)
        models[name] = CrossFitEnsemble([m_AB, m_BA]).recenter(pooled)

    # full-sample baselines (bz sweeps only)
    if cfg["full_baselines"]:
        base_full = covar_trainer(BaseNetwork, model_params, train_loader=full_tr, val_loader=full_va, **tp)
        base_full = base_full.center_effects(full_tr)
        models["base"] = base_full
        web = PostHocOrthNetwork(base_full, num_covariates=p).to(device)
        models["posthoc_web"] = web.fit(full_tr, full_va)

    # test evaluation
    test_sim = dict(sim_params, n=TEST_N)
    X_te, Z_te, y_te, fx_te, fz_te, fr_te = simulate_traffic_light_data(**test_sim, seed=TEST_SEED)
    loader = DataLoader(CovarDataset(X_te, Z_te, y_te), batch_size=min(200, TEST_N), shuffle=False)
    truths = {
        "y": y_te.view(-1).numpy().astype(np.float32),
        "fx": fx_te.view(-1).numpy().astype(np.float32),
        "fr": fr_te.view(-1).numpy().astype(np.float32),
        "fz": fz_te.view(-1).numpy().astype(np.float32),
    }
    n_methods = save_npz(npz_path, models, loader, truths, device, setting, sim_params)

    # cleanup
    del models, base_A, base_B
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
        study="a_linear_consistency", device=DEVICE, n_workers=N_WORKERS,
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
