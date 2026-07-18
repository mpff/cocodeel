"""Study B — nonlinear f_Z: well-specified spline basis vs misspecified raw Z.

Same image DGP as study A but fz(Z) = bz * sin(2*pi*(Z - 0.5)). The
nonlinear_fz sweep feeds the refit a cubic B-spline basis of Z (correctly
specified); nonlinear_fz_misspec feeds raw 1-d Z, so the linear-in-Z refit can
only recover the linear projection of the sine and the residual leaks into
f_X. Methods: refit and refit_orth (2-fold cross-fit).

Usage:  NSIM=5 python experiments/simulation/study_b_misspecification.py
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import BSpline
from torch.nn import MSELoss
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cocodeel.model import BaseNetwork
from cocodeel.refit_model import RefitCovarNetwork
from cocodeel.crossfit import CrossFitEnsemble
from cocodeel.trainer import covar_trainer
from cocodeel.dataset import CovarDataset

from experiments.simulation.common.backbone import TrafficBackbone
from experiments.simulation.common.dgp import circle_mask
from experiments.simulation.common.loaders import simulate_dataloaders_split
from experiments.simulation.common import grid_runner


# ── run config ────────────────────────────────────────────────────────────────
DEVICE = os.environ.get("COCODEEL_DEVICE", "cuda:0")
N_WORKERS = 4
NSIM = int(os.environ.get("NSIM", "50"))
Q = 32
TEST_SEED = 1234
TEST_N = 800
EPOCHS_CAP = 1000
RUN_DIR = ROOT / "experiments/simulation/output/runs/study_b"

# selected by hpsearch/search_default.py (hpsearch/chosen_hps.json)
HP = dict(lr=3e-3, wd=1e-5, early_pat=6, sched_pat=5)

N_GRID = [400, 800, 1600, 3200, 6400, 12800, 25600]
BZ_GRID = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]

SIM_DEFAULTS = dict(b2=1., b3=1., cv1=0.5, cv2=0.5, sdy=1.)


# ── DGP: sinusoidal covariate effect ──────────────────────────────────────────
def simulate_data_nonlinear_fz(
        n=800, h=20, w=60, circle_radius=8,
        bz=1., b2=1., b3=1., cv1=0.5, cv2=0.5, sdy=1., seed=0,
        n_covars=1, outcome_type='continuous'):
    """Image DGP with fz(Z) = bz * sin(2*pi*(Z - 0.5)), one period over Z's support."""
    if n_covars != 1:
        raise ValueError("nonlinear fz is univariate: n_covars must be 1")
    torch.manual_seed(seed)

    # covariates and latents
    Z = torch.rand(n, 1)
    v1_raw = torch.rand(n, 1)
    v2_raw = torch.rand(n, 1)
    v3 = torch.rand(n, 1)
    v1 = (1 - cv1) * v1_raw + cv1 * Z
    v2 = (1 - cv2) * v2_raw + cv2 * Z

    # images
    X = torch.zeros((n, 1, h, w))
    centers = [(h // 2, w // 6), (h // 2, w // 2), (h // 2, 5 * w // 6)]
    masks = [circle_mask(h, w, c, circle_radius) for c in centers]
    for i in range(n):
        X[i, 0][masks[0]] = v1[i]
        X[i, 0][masks[1]] = v2[i]
        X[i, 0][masks[2]] = v3[i]

    # outcome
    fx = b2 * (v2 - 0.5) + b3 * (v3 - 0.5)
    fz = bz * torch.sin(2 * torch.pi * (Z - 0.5))
    eta = fx + fz
    if outcome_type == 'continuous':
        y = eta + sdy * torch.randn(n, 1)
    elif outcome_type == 'binary':
        y = torch.bernoulli(torch.sigmoid(eta))
    else:
        raise ValueError("outcome_type must be 'continuous' or 'binary'.")

    # residual image effect: v2's Z-dependence is linear regardless of fz's shape
    fr = fx - b2 * cv2 * (Z - 0.5)
    return X, Z, y, fx, fz, fr


# ── B-spline covariate basis ──────────────────────────────────────────────────
def make_bspline_basis(z, knots, degree=3):
    """Clamped B-spline design matrix (n, n_basis) for a 1-d covariate."""
    z = np.atleast_1d(z).astype(np.float32)
    knot_vector = np.concatenate([
        np.repeat(knots[0], degree + 1),
        knots[1:-1],
        np.repeat(knots[-1], degree + 1),
    ])
    n_basis = len(knot_vector) - degree - 1
    # clip to the support so boundary samples get a nonzero basis row
    z_clipped = np.clip(z, knot_vector[degree], knot_vector[-degree - 1])
    B = np.column_stack([
        BSpline.basis_element(knot_vector[i:i + degree + 2], extrapolate=False)(z_clipped)
        for i in range(n_basis)
    ])
    return np.nan_to_num(B, nan=0.0).astype(np.float32)


class BSplineBasisTransform:
    """Picklable callable mapping a covariate tensor (n, 1) to its B-spline basis (n, n_basis)."""

    def __init__(self, knots, degree=3):
        self.knots = np.asarray(knots, dtype=np.float32)
        self.degree = degree
        self.n_basis = len(self.knots) + degree - 1

    def __call__(self, Z):
        z = Z.numpy().ravel() if torch.is_tensor(Z) else np.asarray(Z).ravel()
        return torch.from_numpy(make_bspline_basis(z, self.knots, self.degree))


# cubic B-spline, 5 inner knots on [0, 1] -> 9 basis functions
SPLINE_BASIS = BSplineBasisTransform(knots=np.linspace(0.0, 1.0, 7), degree=3)

SWEEPS = {
    "nonlinear_fz": dict(covar_transform=SPLINE_BASIS, num_covariates=SPLINE_BASIS.n_basis),
    "nonlinear_fz_misspec": dict(covar_transform=None, num_covariates=1),
}
SETTINGS = [dict(n=n, bz=bz) for n in N_GRID for bz in BZ_GRID]
sweep_key = lambda s: f"n={s['n']}_bz={s['bz']}"


# ── shared fitting pieces ─────────────────────────────────────────────────────
def trainer_params():
    return dict(
        device=torch.device(DEVICE),
        loss_fn=MSELoss(),
        epochs=EPOCHS_CAP,
        lr=HP["lr"],
        weight_decay=HP["wd"],
        patience=HP["early_pat"],
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_kwargs={"mode": "min", "patience": HP["sched_pat"], "factor": 0.5},
        use_amp=True,
    )


def gather_predictions(model, loader, device):
    """Collect y/fx/fz predictions; fr duplicates fx — a model's image effect targets fx or fr depending on its orthogonalization."""
    model.eval()
    ys, fxs, fzs = [], [], []
    with torch.no_grad():
        for b in loader:
            x = b["X"].to(device)
            z = b["Z"].to(device)
            ys.append(model(x, z).cpu())
            fxs.append(model.predict_fx(x, z).cpu())
            fzs.append(model.predict_fz(z).cpu())
    fx_arr = torch.cat(fxs).view(-1).numpy().astype(np.float32)
    return {
        "y": torch.cat(ys).view(-1).numpy().astype(np.float32),
        "fx": fx_arr,
        "fr": fx_arr,
        "fz": torch.cat(fzs).view(-1).numpy().astype(np.float32),
    }


# ── one simulation ────────────────────────────────────────────────────────────
def run_one(sweep, setting, seed):
    cfg = SWEEPS[sweep]
    key = sweep_key(setting)
    outdir = RUN_DIR / sweep / "preds" / key
    outdir.mkdir(parents=True, exist_ok=True)
    npz_path = outdir / f"seed={seed}.npz"
    if npz_path.exists():
        return dict(sweep=sweep, sweep_key=key, seed=seed, status="cached", wall_s=0.0)

    device = torch.device(DEVICE)
    torch.manual_seed(seed)
    t0 = time.time()

    # data
    sim_params = dict(SIM_DEFAULTS, **setting, outcome_type="continuous")
    _, half_A, half_B, pooled = simulate_dataloaders_split(
        sim_params, seed=seed,
        dgp_fn=simulate_data_nonlinear_fz, covar_transform=cfg["covar_transform"],
    )
    hA_tr, hA_va = half_A
    hB_tr, hB_va = half_B

    # backbones
    p = cfg["num_covariates"]
    model_params = dict(
        backbone=TrafficBackbone,
        backbone_params={"out_features": Q},
        num_covariates=p,
        link="identity",
    )
    tp = trainer_params()
    base_A = covar_trainer(BaseNetwork, model_params, train_loader=hA_tr, val_loader=hA_va, **tp)
    base_A = base_A.center_effects(hA_tr)
    base_B = covar_trainer(BaseNetwork, model_params, train_loader=hB_tr, val_loader=hB_va, **tp)
    base_B = base_B.center_effects(hB_tr)

    # refit variants (2-fold cross-fit)
    models = {}
    for name, orth in [("refit", False), ("refit_orth", True)]:
        m_AB = RefitCovarNetwork(base_A, num_covariates=p, orthogonalize=orth).to(device)
        m_AB = m_AB.fit(hB_tr, hB_va)
        m_BA = RefitCovarNetwork(base_B, num_covariates=p, orthogonalize=orth).to(device)
        m_BA = m_BA.fit(hA_tr, hA_va)
        models[name] = CrossFitEnsemble([m_AB, m_BA]).recenter(pooled)

    # test evaluation (same Z structure as training: transform if the sweep does)
    test_sim = dict(sim_params, n=TEST_N)
    X_te, Z_te_raw, y_te, fx_te, fz_te, fr_te = simulate_data_nonlinear_fz(**test_sim, seed=TEST_SEED)
    Z_te = cfg["covar_transform"](Z_te_raw) if cfg["covar_transform"] is not None else Z_te_raw
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
    del models, base_A, base_B
    torch.cuda.empty_cache()
    gc.collect()
    return dict(sweep=sweep, sweep_key=key, seed=seed, status="ok",
                wall_s=time.time() - t0, n_methods=len(arrays))


def _worker(task):
    return grid_runner.catch_errors(run_one, task, sweep_key(task["setting"]))


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    grid_runner.write_manifest(RUN_DIR, dict(
        study="b_misspecification", device=DEVICE, n_workers=N_WORKERS,
        nsim=NSIM, q=Q, test=dict(seed=TEST_SEED, n=TEST_N), hp=HP,
        sweeps={s: len(SETTINGS) for s in SWEEPS},
    ))
    grid_runner.write_settings_csv(RUN_DIR, {s: SETTINGS for s in SWEEPS},
                                   {s: sweep_key for s in SWEEPS})

    done = grid_runner.already_done(RUN_DIR)
    tasks = [dict(sweep=sweep, setting=setting, seed=seed)
             for sweep in SWEEPS
             for setting in SETTINGS
             for seed in range(NSIM)
             if (sweep, sweep_key(setting), seed) not in done]
    print(f"{len(done)} sims cached, {len(tasks)} to run.", flush=True)
    grid_runner.run_grid(RUN_DIR, tasks, _worker, N_WORKERS)


if __name__ == "__main__":
    main()
