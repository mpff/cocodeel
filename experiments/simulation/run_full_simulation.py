"""Run the full paper simulation study under the split posthoc recipe.

Runs every simulation block with the split recipe (disjoint halves for
backbone vs posthoc refit) and the HP combo chosen by `hp_search.py`.
Nsim = 50 per setting, 4 workers on cuda:1.

Outputs (per run directory):
    <run>/manifest.json         — config snapshot, git commit, start time
    <run>/progress.log          — JSONL, one line per completed sim
    <run>/<block>/settings.csv  — one row per sweep setting
    <run>/<block>/preds/<sweep_key>/seed=<n>.npz  — raw test predictions

Aggregation to the R-plot-ready format (per-block CSV with columns
model, effect, n, sweep_value, mspe, bias2, var) is done in a separate
`aggregate_full_simulation.py` step (not in this runner).

Usage:
    /home/RDC/pfeuffma/.conda/envs/dl-mri/bin/python -u \
        experiments/simulation/run_full_simulation.py
"""
from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.nn import MSELoss, BCELoss
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cocodeel.model import BaseNetwork
from cocodeel.posthoc_model import PostHocCovarNetwork
from cocodeel.benchmarking.posthoc_model import PostHocOrthNetwork, SemiStructuredNetwork
from cocodeel.trainer import covar_trainer
from cocodeel.dataset import CovarDataset

from experiments.simulation.backbone import TrafficBackbone
from experiments.simulation.dataset import (
    simulate_traffic_light_data,
    simulate_data_nonlinear_fz,
    BSplineBasisTransform,
)
from experiments.simulation.utils import simulate_dataloaders_split

# Spline basis used by the `nonlinear_fz` block: cubic B-spline with 5
# inner knots on [0, 1] → 9 basis functions. Built once at import time
# and shared via the BLOCKS dict (class-based so it pickles cleanly when
# workers re-import this module).
NONLINEAR_FZ_BASIS = BSplineBasisTransform(
    knots=np.linspace(0.0, 1.0, 7), degree=3,
)
# n_basis = NONLINEAR_FZ_BASIS.n_basis = 9


# ── Run config ────────────────────────────────────────────────────────────────
DEVICE = os.environ.get("COCODEEL_DEVICE", "cuda:1")
N_WORKERS = 4
NSIM = 50
Q_DEFAULT = 32
TEST_SEED = 1234
TEST_N = 800
EPOCHS_CAP = 1000
HP_PATH           = Path(__file__).resolve().parent / "hpsearch" / "chosen_hps.json"
HP_PER_Q_PATH     = Path(__file__).resolve().parent / "hpsearch" / "chosen_hps_q.json"
HP_PER_Q_NAM_PATH = Path(__file__).resolve().parent / "hpsearch" / "chosen_hps_q_nam.json"

N_GRID            = [400, 800, 1600, 3200, 6400, 12800, 25600]
N_GRID_CONCURVITY = N_GRID + [51200, 102400]  # two extra log-steps for Fig 2
BZ_GRID     = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
CV_GRID     = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
Q_GRID      = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
P_GRID      = [1, 2, 4, 8, 16]


# ── Block definitions ─────────────────────────────────────────────────────────
BLOCKS = {
    "binary_increasing_bz": {
        "outcome_type": "binary",
        "sim_defaults": dict(b2=1., b3=1., cv1=0.5, cv2=0.5, sdy=1.),
        "posthoc_configs": {
            "posthoc":      dict(cls=PostHocCovarNetwork, init_kwargs={"orthogonalize": False}, fit_kwargs={"max_iters": 25}),
            "posthoc_orth": dict(cls=PostHocCovarNetwork, init_kwargs={"orthogonalize": True},  fit_kwargs={"max_iters": 25}),
            "posthoc_web":  dict(cls=PostHocOrthNetwork, recipe="full"),
        },
        "settings": [dict(n=n, bz=bz) for n in N_GRID for bz in BZ_GRID],
        "sweep_key_fn": lambda s: f"n={s['n']}_bz={s['bz']}",
    },
    "increasing_bz": {
        "outcome_type": "continuous",
        "sim_defaults": dict(b2=1., b3=1., cv1=0.5, cv2=0.5, sdy=1.),
        "posthoc_configs": {
            "posthoc":           dict(cls=PostHocCovarNetwork, init_kwargs={"orthogonalize": False}),
            "posthoc_lam0":      dict(cls=PostHocCovarNetwork, init_kwargs={"orthogonalize": False}, fit_kwargs={"lam": 0.0}),
            "posthoc_orth":      dict(cls=PostHocCovarNetwork, init_kwargs={"orthogonalize": True}),
            "posthoc_orth_lam0": dict(cls=PostHocCovarNetwork, init_kwargs={"orthogonalize": True},  fit_kwargs={"lam": 0.0}),
            "posthoc_web":       dict(cls=PostHocOrthNetwork, recipe="full"),
        },
        "settings": [dict(n=n, bz=bz) for n in N_GRID for bz in BZ_GRID],
        "sweep_key_fn": lambda s: f"n={s['n']}_bz={s['bz']}",
    },
    "increasing_cv": {
        "outcome_type": "continuous",
        "sim_defaults": dict(bz=1., b2=1., b3=1., cv2=0.5, sdy=1.),
        "posthoc_configs": {
            "posthoc":      dict(cls=PostHocCovarNetwork, init_kwargs={"orthogonalize": False}),
            "posthoc_orth": dict(cls=PostHocCovarNetwork, init_kwargs={"orthogonalize": True}),
        },
        "settings": [dict(n=n, cv1=cv) for n in N_GRID for cv in CV_GRID],
        "sweep_key_fn": lambda s: f"n={s['n']}_cv1={s['cv1']}",
    },
    "increasing_q": {
        "outcome_type": "continuous",
        "sim_defaults": dict(bz=1., b2=1., b3=1., cv1=0.5, cv2=0.5, sdy=1.),
        "posthoc_configs": {
            "posthoc":      dict(cls=PostHocCovarNetwork, init_kwargs={"orthogonalize": False}),
            "posthoc_orth": dict(cls=PostHocCovarNetwork, init_kwargs={"orthogonalize": True}),
        },
        "settings": [dict(n=n, q=q) for n in N_GRID for q in Q_GRID],
        "sweep_key_fn": lambda s: f"n={s['n']}_q={s['q']}",
    },
    "increasing_p": {
        "outcome_type": "continuous",
        "sim_defaults": dict(bz=1., b2=1., b3=1., cv1=0.5, cv2=0.5, sdy=1.),
        "posthoc_configs": {
            "posthoc":      dict(cls=PostHocCovarNetwork, init_kwargs={"orthogonalize": False}),
            "posthoc_orth": dict(cls=PostHocCovarNetwork, init_kwargs={"orthogonalize": True}),
        },
        "settings": [dict(n=n, n_covars=p) for n in N_GRID for p in P_GRID],
        "sweep_key_fn": lambda s: f"n={s['n']}_p={s['n_covars']}",
    },
    "concurvity": {
        "outcome_type": "continuous",
        "sim_defaults": dict(bz=1., b2=1., b3=1., cv1=0.5, cv2=0.5, sdy=1.),
        "posthoc_configs": {
            "posthoc":           dict(cls=PostHocCovarNetwork, init_kwargs={"orthogonalize": False}),
            "posthoc_lam0":      dict(cls=PostHocCovarNetwork, init_kwargs={"orthogonalize": False}, fit_kwargs={"lam": 0.0}),
            "posthoc_orth":      dict(cls=PostHocCovarNetwork, init_kwargs={"orthogonalize": True}),
            "posthoc_xfit":      dict(cls=PostHocCovarNetwork, init_kwargs={"orthogonalize": False}, recipe="xfit"),
            "posthoc_orth_xfit": dict(cls=PostHocCovarNetwork, init_kwargs={"orthogonalize": True},  recipe="xfit"),
            "posthoc_web":       dict(cls=PostHocOrthNetwork, recipe="full"),
        },
        # End-to-end NAM-style methods trained on the full sample. Fig 2
        # contrasts these "concurvity-exposed" fits with the split-recipe
        # posthoc refit.
        "sgd_configs": {
            "covar":          dict(lam_reg=0.0),    # plain NAM (CovarNetwork, no reg)
            "covar_conc_0.1": dict(lam_reg=0.1),    # Siems reg, weak
            "covar_conc_1":   dict(lam_reg=1.0),    # Siems reg, medium
            "covar_conc_10":  dict(lam_reg=10.0),   # Siems reg, strong
        },
        # SSN (Rügamer et al., 2023): wraps a fitted CovarNetwork and
        # post-hoc orthogonalises via lstsq(Z, fX). Each entry names the
        # sgd_config key whose model is wrapped.
        "ssn_configs": {
            "ssn": dict(wraps="covar"),
        },
        "settings": [dict(n=n) for n in N_GRID_CONCURVITY],
        "sweep_key_fn": lambda s: f"n={s['n']}",
    },
    "nonlinear_fz": {
        # Nonlinear-fz block. Same image DGP as the rest of the suite,
        # but fz is sinusoidal (one period over Z's [0,1] support);
        # paired with a cubic B-spline basis on Z (9 functions) so the
        # post-hoc model fits a spline regression for fz with no model
        # changes. Cross-fit (k=2) is the only recipe — both methods
        # share the second backbone trained on `half_B`.
        "outcome_type": "continuous",
        "dgp_fn":       simulate_data_nonlinear_fz,
        "covar_transform":              NONLINEAR_FZ_BASIS,
        "num_covariates_after_transform": NONLINEAR_FZ_BASIS.n_basis,
        "sim_defaults": dict(b2=1., b3=1., cv1=0.5, cv2=0.5, sdy=1.),
        "posthoc_configs": {
            "posthoc_xfit":      dict(cls=PostHocCovarNetwork, init_kwargs={"orthogonalize": False}, recipe="xfit"),
            "posthoc_orth_xfit": dict(cls=PostHocCovarNetwork, init_kwargs={"orthogonalize": True},  recipe="xfit"),
        },
        "settings": [dict(n=n, bz=bz) for n in N_GRID for bz in BZ_GRID],
        "sweep_key_fn": lambda s: f"n={s['n']}_bz={s['bz']}",
    },
    "concurvity_q": {
        # Concurvity vs backbone size: how does the NAM (covar) suffer
        # under increasing q (last-layer width / backbone capacity), and
        # does cocodeel's xfit refit (DNN w. Controls) stay consistent?
        # Same DGP defaults as the original concurvity block (linear fz,
        # bz=1) but sweep is over (n, q). Per-q HPs are loaded from
        # chosen_hps_q.json (set hp_per_q=True) — backbones with
        # 5k → 530k params need different lr.
        "outcome_type": "continuous",
        "hp_per_q": True,
        "sim_defaults": dict(bz=1., b2=1., b3=1., cv1=0.5, cv2=0.5, sdy=1.),
        "posthoc_configs": {
            "posthoc_xfit": dict(cls=PostHocCovarNetwork,
                                 init_kwargs={"orthogonalize": False},
                                 recipe="xfit"),
        },
        "sgd_configs": {
            "covar": dict(lam_reg=0.0),   # NAM (CovarNetwork, no reg)
        },
        "settings": [dict(n=n, q=q) for n in N_GRID for q in Q_GRID],
        "sweep_key_fn": lambda s: f"n={s['n']}_q={s['q']}",
    },
    "nonlinear_fz_misspec": {
        # Misspecification contrast to `nonlinear_fz`: same sin-fz DGP
        # and same xfit recipe, but the model is fed the RAW 1-d
        # covariate Z (no spline basis). The linear-in-Z refit can only
        # recover the linear projection of sin(2π(Z-0.5)) (coefficient
        # 6/π, captured variance 3/π² per unit bz²); the residual
        # ≈ 0.2·bz² nonlinear-fz variance is unfitted and leaks into
        # f̂_X via the X-Z correlation. Used in the appendix to show
        # the consequences of misspecifying the covariate basis.
        "outcome_type": "continuous",
        "dgp_fn":       simulate_data_nonlinear_fz,
        # No covar_transform / no num_covariates_after_transform —
        # defaults give n_covars=1 (raw Z), which is the
        # misspecification we are measuring.
        "sim_defaults": dict(b2=1., b3=1., cv1=0.5, cv2=0.5, sdy=1.),
        "posthoc_configs": {
            "posthoc_xfit":      dict(cls=PostHocCovarNetwork, init_kwargs={"orthogonalize": False}, recipe="xfit"),
            "posthoc_orth_xfit": dict(cls=PostHocCovarNetwork, init_kwargs={"orthogonalize": True},  recipe="xfit"),
        },
        "settings": [dict(n=n, bz=bz) for n in N_GRID for bz in BZ_GRID],
        "sweep_key_fn": lambda s: f"n={s['n']}_bz={s['bz']}",
    },
}


# ── Run directory + manifest ──────────────────────────────────────────────────
def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def setup_run_dir(suffix: str = "full") -> Path:
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = ROOT / f"experiments/simulation/output/runs/{ts}_{suffix}_nsim{NSIM}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_manifest(run_dir: Path, hp: dict) -> None:
    manifest = {
        "start": datetime.datetime.now().isoformat(),
        "host": os.uname().nodename,
        "git_commit": _git_commit(),
        "conda_env": "dl-mri",
        "device": DEVICE,
        "n_workers": N_WORKERS,
        "nsim": NSIM,
        "q_default": Q_DEFAULT,
        "test": {"seed": TEST_SEED, "n": TEST_N},
        "blocks": {b: {
            "outcome_type": B["outcome_type"],
            "posthoc_configs": list(B["posthoc_configs"].keys()),
            "n_settings": len(B["settings"]),
        } for b, B in BLOCKS.items()},
        "hp": hp,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


# ── Task helpers ──────────────────────────────────────────────────────────────
def _model_params_for(setting: dict, block_cfg: dict) -> dict:
    q = setting.get("q", Q_DEFAULT)
    # If the block applies a covariate transform (e.g. spline basis),
    # the model sees `num_covariates_after_transform` columns, not the
    # raw `n_covars` from the DGP.
    p = block_cfg.get("num_covariates_after_transform",
                      setting.get("n_covars", 1))
    return dict(
        backbone=TrafficBackbone,
        backbone_params={"out_features": q},
        num_covariates=p,
        link=("logit" if block_cfg["outcome_type"] == "binary" else "identity"),
    )


def _sim_params_for(setting: dict, block_cfg: dict) -> dict:
    sim = dict(block_cfg["sim_defaults"])
    sim.update({k: v for k, v in setting.items() if k not in ("q",)})
    sim["outcome_type"] = block_cfg["outcome_type"]
    return sim


def _trainer_params(outcome_type: str, hp: dict) -> dict:
    h = hp[outcome_type]
    return dict(
        device=torch.device(DEVICE),
        loss_fn=BCELoss() if outcome_type == "binary" else MSELoss(),
        epochs=EPOCHS_CAP,
        lr=h["lr"],
        weight_decay=h["wd"],
        patience=h["early_pat"],
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_kwargs={"mode": "min", "patience": h["sched_pat"], "factor": 0.5},
        use_amp=(outcome_type != "binary"),
    )


def _gather_predictions(model, loader, device):
    model.eval()
    ys, fxs, fzs = [], [], []
    with torch.no_grad():
        for b in loader:
            x = b["X"].to(device)
            z = b["Z"].to(device)
            y_hat = model(x, z) if getattr(model, "num_covariates", 0) > 0 else model(x)
            ys.append(y_hat.cpu())
            # predict_fx returns the model's configured image effect:
            # raw fx for orthogonalize=False, orthogonalized fx (= fr) for True.
            # Store the same tensor under both "fx" and "fr" — figures filter by
            # model name to pick the semantically correct row.
            fxs.append(model.predict_fx(x, z).cpu())
            fzs.append(model.predict_fz(z).cpu())
    fx_arr = torch.cat(fxs, dim=0).view(-1).numpy().astype(np.float32)
    return {
        "y":  torch.cat(ys,  dim=0).view(-1).numpy().astype(np.float32),
        "fx": fx_arr,
        "fr": fx_arr,
        "fz": torch.cat(fzs, dim=0).view(-1).numpy().astype(np.float32),
    }


class CrossFitAverageModel:
    """Two-fold cross-fit estimator. Holds two trained sub-models (each a
    regular post-hoc refit on a complementary half of the data) and
    returns the elementwise mean of their predictions on a shared test
    set. Exposes the same forward / predict_fx / predict_fz interface as
    the underlying models so it slots into `_gather_predictions`
    unchanged."""

    def __init__(self, m1, m2):
        self.m1 = m1
        self.m2 = m2
        # Surface num_covariates so the wrapper dispatches like its sub-models.
        self.num_covariates = getattr(m1, "num_covariates", 0)

    def eval(self):
        self.m1.eval(); self.m2.eval()
        return self

    def to(self, device):
        self.m1.to(device); self.m2.to(device)
        return self

    def __call__(self, x, z=None):
        if z is None:
            return (self.m1(x) + self.m2(x)) / 2
        return (self.m1(x, z) + self.m2(x, z)) / 2

    def predict_fx(self, x, z=None):
        if z is None:
            return (self.m1.predict_fx(x) + self.m2.predict_fx(x)) / 2
        return (self.m1.predict_fx(x, z) + self.m2.predict_fx(x, z)) / 2

    def predict_fz(self, z):
        return (self.m1.predict_fz(z) + self.m2.predict_fz(z)) / 2


def _fit_posthoc(model_cls, backbone, fit_tr, fit_va, init_kwargs,
                 fit_kwargs, num_covariates, device):
    """Construct and fit one post-hoc model on the refit sample."""
    m = model_cls(backbone, num_covariates=num_covariates,
                  **init_kwargs).to(device)
    return m.fit(fit_tr, fit_va, **fit_kwargs)


def _run_one_sim(block_name: str, setting: dict, seed: int, run_dir: Path, hp: dict) -> dict:
    """Train all models for one (block, setting, seed) triple, save predictions."""
    block_cfg = BLOCKS[block_name]
    sweep_key = block_cfg["sweep_key_fn"](setting)
    outdir = run_dir / block_name / "preds" / sweep_key
    outdir.mkdir(parents=True, exist_ok=True)
    npz_path = outdir / f"seed={seed}.npz"

    if npz_path.exists():
        return {"block": block_name, "sweep_key": sweep_key, "seed": seed,
                "status": "cached", "wall_s": 0.0}

    device = torch.device(DEVICE)
    torch.manual_seed(seed)

    sim_params = _sim_params_for(setting, block_cfg)
    model_params = _model_params_for(setting, block_cfg)

    # Per-q HP dispatch: if the block was configured with `hp_per_q`,
    # look up the lr / wd / patience combo from chosen_hps_q.json by
    # the current q value. We construct a minimal hp dict that
    # `_trainer_params` can consume without modification.
    #
    # NAM (sgd_configs) is searched separately from the posthoc backbone
    # (different model class — CovarNetwork end-to-end vs BaseNetwork
    # backbone), so when chosen_hps_q_nam.json exists, sgd_configs use
    # those HPs while posthoc_configs keep the original chosen_hps_q.json.
    if block_cfg.get("hp_per_q"):
        if not HP_PER_Q_PATH.exists():
            raise RuntimeError(
                f"Block {block_name} requires per-q HPs at {HP_PER_Q_PATH} — "
                "run hp_search_q.py first."
            )
        hp_q = json.loads(HP_PER_Q_PATH.read_text())
        combo = hp_q[str(setting["q"])]
        hp_local = {block_cfg["outcome_type"]: combo}

        if HP_PER_Q_NAM_PATH.exists():
            hp_q_nam = json.loads(HP_PER_Q_NAM_PATH.read_text())
            combo_nam = hp_q_nam[str(setting["q"])]
            hp_local_nam = {block_cfg["outcome_type"]: combo_nam}
        else:
            hp_local_nam = hp_local
    else:
        hp_local = hp
        hp_local_nam = hp
    tp = _trainer_params(block_cfg["outcome_type"], hp_local)
    tp_nam = _trainer_params(block_cfg["outcome_type"], hp_local_nam)

    t0 = time.time()

    # Optional per-block hooks: a custom DGP function and a covariate
    # transform (e.g. spline basis on Z). Defaults preserve the original
    # behaviour for the existing 6 blocks.
    dgp_fn          = block_cfg.get("dgp_fn")
    covar_transform = block_cfg.get("covar_transform")

    full, half_A, half_B = simulate_dataloaders_split(
        sim_params, seed=seed,
        dgp_fn=dgp_fn, covar_transform=covar_transform,
    )
    full_tr, full_va = full
    hA_tr,  hA_va  = half_A
    hB_tr,  hB_va  = half_B

    posthoc_configs = block_cfg["posthoc_configs"]
    recipes = {cfg.get("recipe", "split") for cfg in posthoc_configs.values()}

    # Train only the backbones the block's recipes actually need. Saves
    # ~one full-N backbone per sim for blocks that don't use the "full"
    # recipe (e.g. nonlinear_fz uses xfit only).
    base_full = None
    if "full" in recipes:
        base_full = covar_trainer(BaseNetwork, model_params,
                                  train_loader=full_tr, val_loader=full_va, **tp)
        base_full = base_full.center_effects(full_tr)

    base_half = None
    if recipes & {"split", "xfit"}:
        base_half = covar_trainer(BaseNetwork, model_params,
                                  train_loader=hA_tr, val_loader=hA_va, **tp)
        base_half = base_half.center_effects(hA_tr)

    # Cross-fit (recipe="xfit") needs a second backbone trained on the
    # OTHER half. Shared across all xfit methods, since posthoc and
    # posthoc_orth differ only in their refit step.
    base_half_B = None
    if "xfit" in recipes:
        base_half_B = covar_trainer(BaseNetwork, model_params,
                                    train_loader=hB_tr, val_loader=hB_va, **tp)
        base_half_B = base_half_B.center_effects(hB_tr)

    p = model_params["num_covariates"]
    posthoc_models = {}
    for name, cfg in posthoc_configs.items():
        init_kwargs = cfg.get("init_kwargs", {})
        fit_kwargs  = cfg.get("fit_kwargs", {})
        model_cls   = cfg["cls"]
        # recipe="split" (default): backbone half_A, refit half_B.
        # recipe="full":            backbone full, refit full.
        # recipe="xfit":            two-fold cross-fit — fold 1 (AB):
        #                           backbone half_A, refit half_B; fold 2
        #                           (BA): backbone half_B, refit half_A;
        #                           predictions averaged.
        # The Weber baseline (PostHocOrthNetwork / "posthoc_web") uses full
        # — cocodeel's posthoc refit is the only method whose unbiasedness
        # argument requires splitting; full-sample fit is the published
        # Weber recipe and the fair apples-to-apples baseline for it.
        recipe = cfg.get("recipe", "split")

        if recipe == "xfit":
            m_AB = _fit_posthoc(model_cls, base_half,   hB_tr, hB_va,
                                init_kwargs, fit_kwargs, p, device)
            m_BA = _fit_posthoc(model_cls, base_half_B, hA_tr, hA_va,
                                init_kwargs, fit_kwargs, p, device)
            m = CrossFitAverageModel(m_AB, m_BA)
        elif recipe == "full":
            m = _fit_posthoc(model_cls, base_full, full_tr, full_va,
                             init_kwargs, fit_kwargs, p, device)
        else:  # "split"
            m = _fit_posthoc(model_cls, base_half, hB_tr, hB_va,
                             init_kwargs, fit_kwargs, p, device)
        posthoc_models[name] = m

    # End-to-end (no-refit) SGD-style fits: train on full N (not split).
    # Currently only used by the concurvity block to expose NAM/Siems
    # benchmarks alongside the posthoc refit.
    sgd_models = {}
    sgd_configs = block_cfg.get("sgd_configs", {})
    if sgd_configs:
        from cocodeel.benchmarking.model import CovarNetwork
        from experiments.simulation.concurvity_methods import (
            train_covar_with_concurvity_reg,
        )
        for name, cfg in sgd_configs.items():
            m = train_covar_with_concurvity_reg(
                CovarNetwork, model_params, full_tr, full_va,
                lam_reg=cfg["lam_reg"], **tp_nam,
            )
            m = m.center_effects(full_tr)
            sgd_models[name] = m

    # SSN (SemiStructuredNetwork) wraps an already-fitted CovarNetwork and
    # adds a post-hoc orthogonalisation term via lstsq(Z, fX). The wrapped
    # model is identified by name in `ssn_configs[name]["wraps"]`.
    ssn_models = {}
    for name, cfg in block_cfg.get("ssn_configs", {}).items():
        wraps = cfg["wraps"]
        if wraps not in sgd_models:
            raise ValueError(f"ssn_configs['{name}'] wraps '{wraps}', "
                             f"but no sgd_config produced that model.")
        m = SemiStructuredNetwork(sgd_models[wraps]).to(device)
        m = m.fit(full_tr)
        ssn_models[name] = m

    # --- evaluate on shared test draw ---
    # For the multi-covariate p-sweep, the test draw must use n_covars=p too
    # (otherwise Z dims mismatch). Keep everything else at defaults. For
    # blocks with a custom DGP / covariate transform, mirror them on the
    # test set so the model sees the same Z structure.
    test_sim = dict(sim_params); test_sim["n"] = TEST_N
    test_dgp = dgp_fn if dgp_fn is not None else simulate_traffic_light_data
    X_te, Z_te_raw, y_te, fx_te, fz_te, fr_te = test_dgp(**test_sim, seed=TEST_SEED)
    Z_te = covar_transform(Z_te_raw) if covar_transform is not None else Z_te_raw
    loader = DataLoader(CovarDataset(X_te, Z_te, y_te), batch_size=min(200, TEST_N), shuffle=False)

    arrays = {}
    truths = {
        "y":  y_te.view(-1).numpy().astype(np.float32),
        "fx": fx_te.view(-1).numpy().astype(np.float32),
        "fr": fr_te.view(-1).numpy().astype(np.float32),
        "fz": fz_te.view(-1).numpy().astype(np.float32),
    }
    to_eval = []
    if base_full is not None: to_eval.append(("base_full", base_full))
    if base_half is not None: to_eval.append(("base_half", base_half))
    to_eval += list(sgd_models.items()) + list(ssn_models.items()) + list(posthoc_models.items())
    for name, m in to_eval:
        arrays[name] = _gather_predictions(m, loader, device)

    # Save one NPZ: per-method predictions + truths + metadata.
    np.savez_compressed(
        npz_path,
        methods=np.array(list(arrays.keys())),
        **{f"{name}__{eff}": arr for name, d in arrays.items() for eff, arr in d.items()},
        **{f"truth__{eff}": arr for eff, arr in truths.items()},
        setting=np.array(json.dumps(setting)),
        sim_params=np.array(json.dumps({k: v for k, v in sim_params.items()})),
    )

    # Clean up GPU memory between sims.
    del base_full, base_half, posthoc_models, sgd_models, ssn_models
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "block": block_name, "sweep_key": sweep_key, "seed": seed,
        "status": "ok", "wall_s": time.time() - t0,
        "n_methods": len(arrays),
    }


def _worker(task):
    """Wrap _run_one_sim with exception handling so a single bad sim
    does not kill the pool."""
    try:
        return _run_one_sim(**task)
    except Exception as e:
        return {
            "block": task["block_name"], "sweep_key": BLOCKS[task["block_name"]]["sweep_key_fn"](task["setting"]),
            "seed": task["seed"], "status": "error",
            "error": str(e), "traceback": traceback.format_exc()[-1500:],
        }


def _already_done(run_dir: Path) -> set:
    """Scan existing NPZ files and return the set of (block, sweep_key, seed) already complete."""
    done = set()
    for block in BLOCKS:
        base = run_dir / block / "preds"
        if not base.exists():
            continue
        for sweep_dir in base.iterdir():
            if not sweep_dir.is_dir():
                continue
            for p in sweep_dir.glob("seed=*.npz"):
                try:
                    seed = int(p.stem.split("=")[1])
                    done.add((block, sweep_dir.name, seed))
                except ValueError:
                    continue
    return done


# ── Settings CSV (one row per sweep) ──────────────────────────────────────────
def _write_settings_csv(run_dir: Path):
    import csv
    for block, cfg in BLOCKS.items():
        outp = run_dir / block / "settings.csv"
        outp.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for s in cfg["settings"] for k in s.keys()})
        with outp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["sweep_key", *keys])
            w.writeheader()
            for s in cfg["settings"]:
                w.writerow({"sweep_key": cfg["sweep_key_fn"](s), **s})


# ── Runner ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, default=None,
                    help="Resume an existing run dir. If None, create a new one.")
    ap.add_argument("--only-block", type=str, default=None,
                    help="Restrict to a single block (e.g. 'increasing_bz').")
    ap.add_argument("--nsim", type=int, default=NSIM)
    ap.add_argument("--device", type=str, default=DEVICE,
                    help="CUDA device (e.g. 'cuda:0'). Overrides module default.")
    args = ap.parse_args()

    # Override module-level DEVICE so spawned workers pick up the chosen device
    # at import time (they re-import this module; the module sees the CLI arg
    # via an env var set here before Pool spawn).
    globals()["DEVICE"] = args.device
    os.environ["COCODEEL_DEVICE"] = args.device

    if not HP_PATH.exists():
        print(f"Chosen HPs not found at {HP_PATH} — run hp_search.py first.", file=sys.stderr)
        sys.exit(1)
    hp = json.loads(HP_PATH.read_text())

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        print(f"Resuming {run_dir}")
    else:
        run_dir = setup_run_dir("full")
        write_manifest(run_dir, hp)
        _write_settings_csv(run_dir)
        print(f"New run dir: {run_dir}")

    # Build task list.
    blocks = [args.only_block] if args.only_block else list(BLOCKS)
    tasks = []
    for block in blocks:
        cfg = BLOCKS[block]
        for setting in cfg["settings"]:
            for seed in range(args.nsim):
                tasks.append(dict(block_name=block, setting=setting, seed=seed, run_dir=run_dir, hp=hp))

    done = _already_done(run_dir)
    tasks = [t for t in tasks
             if (t["block_name"], BLOCKS[t["block_name"]]["sweep_key_fn"](t["setting"]), t["seed"]) not in done]

    print(f"[{datetime.datetime.now():%H:%M:%S}] "
          f"{len(tasks)} tasks to run ({len(done)} already complete). "
          f"Workers={N_WORKERS} on {DEVICE}.", flush=True)

    progress_path = run_dir / "progress.log"
    t_start = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=N_WORKERS) as pool:
        for i, res in enumerate(pool.imap_unordered(_worker, tasks), start=1):
            line = json.dumps({"t": datetime.datetime.now().isoformat(), **res})
            with progress_path.open("a") as f:
                f.write(line + "\n")
            tag = res.get("status", "?").upper()
            print(f"[{datetime.datetime.now():%H:%M:%S}] {i}/{len(tasks)} {tag}  "
                  f"{res.get('block','?')}/{res.get('sweep_key','?')} seed={res.get('seed','?')} "
                  f"t={res.get('wall_s', 0.0):.1f}s"
                  + (f"  ERR: {res.get('error','')}" if tag == "ERROR" else ""),
                  flush=True)

    print(f"\n[{datetime.datetime.now():%H:%M:%S}] Done in "
          f"{(time.time() - t_start)/3600:.2f} h. Run: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
