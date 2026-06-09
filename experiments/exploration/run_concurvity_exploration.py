"""Concurvity exploration study — further work, not part of the current paper.

Reruns the concurvity block of the paper simulation
(`simulation_images/run_full_simulation.py`) with two changes:

  * the only end-to-end NAM-style fits are an SGD-trained CovarNetwork
    (linear f_z), a NAM with an MLP shape function for f_z, and a ridge
    sweep over that NAM (AdamW weight decay on all parameters); no
    Siems penalty, no backfitting variants.
  * results are written under `results/exploration/`, never the paper tree.

Methods per (n, seed):
    sgd            end-to-end CovarNetwork (linear f_z), MSE only
    nam            end-to-end CovarNetworkMLPfz (MLP f_z), MSE only
    nam_ridge_<λ>  the NAM, AdamW ridge on all params, one fit per λ in RIDGE_LAMBDAS
    posthoc        sample-split refit (backbone half_A, refit half_B)
    posthoc_xfit   two-fold cross-fit refit (folds averaged)

DGP, backbone, split helper, and the post-hoc fitting machinery are imported
from the paper experiment; HPs are read from its `chosen_hps.json`. The
methods are fit in clearly delimited blocks inside `_run_one_sim`; comment
out a block to skip those methods on a fresh run.

Outputs (per run directory):
    <run>/manifest.json              config snapshot, git commit, start time
    <run>/progress.log               JSONL, one line per completed sim
    <run>/n=<n>/seed=<s>.npz         per-method f_x/f_z/y predictions + truths

Usage:
    /home/RDC/pfeuffma/.conda/envs/dl-mri/bin/python -u \
        experiments/exploration/run_concurvity_exploration.py [--nsim N] [--device cuda:1]
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
from torch.nn import BCELoss, MSELoss
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cocodeel.model import BaseNetwork, CovarNetwork
from cocodeel.posthoc_model import PostHocCovarNetwork
from cocodeel.trainer import covar_trainer
from cocodeel.dataset import CovarDataset

from experiments.simulation_images.backbone import TrafficBackbone
from experiments.simulation_images.dataset import simulate_traffic_light_data
from experiments.simulation_images.utils import simulate_dataloaders_split
from experiments.simulation_images.run_full_simulation import (
    _fit_posthoc, CrossFitAverageModel, _accepts_two,
)

from experiments.exploration.covar_mlp_fz import CovarNetworkMLPfz

# ── Run config ──────────────────────────────────────────────────────────────
# DEVICE / OUTCOME_TYPE are read from the environment so spawned workers
# (which re-import this module) pick up the runner's CLI choices at import
# time. binary → logit + BCE; continuous → identity + MSE.
DEVICE = os.environ.get("COCODEEL_DEVICE", "cuda:1")
OUTCOME_TYPE = os.environ.get("COCODEEL_OUTCOME", "binary")
N_WORKERS = 4
NSIM = 50
Q_DEFAULT = 32
TEST_SEED = 1234
TEST_N = 800
EPOCHS_CAP = 1000
HP_PATH = ROOT / "experiments/simulation_images/chosen_hps.json"

# Subset of the paper concurvity block's N grid (n=102400 dropped — too slow
# for the exploration's value).
N_GRID = [400, 800, 1600, 3200, 6400, 12800, 25600, 51200]

# Ridge sweep for the `nam_ridge_<λ>` method: AdamW weight decay (decoupled
# L2 on all parameters). λ=100 over-regularises to a near-constant predictor
# (flat, high-bias, ~zero-variance), so the sweep stops at 0.1.
RIDGE_LAMBDAS = [0.001, 0.1]

SIM_DEFAULTS = dict(bz=1., b2=1., b3=1., cv1=0.5, cv2=0.5, sdy=1.)

MODEL_PARAMS = dict(
    backbone=TrafficBackbone,
    backbone_params={"out_features": Q_DEFAULT},
    num_covariates=1,
    link=("logit" if OUTCOME_TYPE == "binary" else "identity"),
)


def _trainer_params(hp: dict) -> dict:
    h = hp[OUTCOME_TYPE]
    return dict(
        device=torch.device(DEVICE),
        loss_fn=BCELoss() if OUTCOME_TYPE == "binary" else MSELoss(),
        epochs=EPOCHS_CAP,
        lr=h["lr"],
        weight_decay=h["wd"],
        patience=h["early_pat"],
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_kwargs={"mode": "min", "patience": h["sched_pat"], "factor": 0.5},
        use_amp=(OUTCOME_TYPE != "binary"),
    )


def _gather(model, loader, device):
    """Test-set predictions of y, f_x, f_z for one model."""
    model.eval()
    ys, fxs, fzs = [], [], []
    with torch.no_grad():
        for b in loader:
            x = b["X"].to(device)
            z = b["Z"].to(device)
            y_hat = model(x, z) if getattr(model, "num_covariates", 0) > 0 else model(x)
            fx_pred = (model.predict_fx(x, z).cpu() if _accepts_two(model.predict_fx)
                       else model.predict_fx(x).cpu())
            ys.append(y_hat.cpu())
            fxs.append(fx_pred)
            fzs.append(model.predict_fz(z).cpu())
    return {
        "y":  torch.cat(ys,  dim=0).view(-1).numpy().astype(np.float32),
        "fx": torch.cat(fxs, dim=0).view(-1).numpy().astype(np.float32),
        "fz": torch.cat(fzs, dim=0).view(-1).numpy().astype(np.float32),
    }


# ── Run directory + manifest ──────────────────────────────────────────────────
def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def setup_run_dir() -> Path:
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = ROOT / f"results/exploration/runs/{ts}_concurvity_{OUTCOME_TYPE}_nsim{NSIM}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_manifest(run_dir: Path, hp: dict) -> None:
    manifest = {
        "start": datetime.datetime.now().isoformat(),
        "host": os.uname().nodename,
        "git_commit": _git_commit(),
        "conda_env": "dl-mri",
        "device": DEVICE,
        "outcome_type": OUTCOME_TYPE,
        "n_workers": N_WORKERS,
        "nsim": NSIM,
        "q_default": Q_DEFAULT,
        "test": {"seed": TEST_SEED, "n": TEST_N},
        "n_grid": N_GRID,
        "ridge_lambdas": RIDGE_LAMBDAS,
        "sim_defaults": SIM_DEFAULTS,
        "effects": ["y", "fx", "fz"],
        "hp": hp,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


# ── One simulation ────────────────────────────────────────────────────────────
def _run_one_sim(n: int, seed: int, run_dir: Path, hp: dict) -> dict:
    sweep_key = f"n={n}"
    outdir = run_dir / sweep_key
    outdir.mkdir(parents=True, exist_ok=True)
    npz_path = outdir / f"seed={seed}.npz"
    if npz_path.exists():
        return {"sweep_key": sweep_key, "seed": seed, "status": "cached", "wall_s": 0.0}

    device = torch.device(DEVICE)
    torch.manual_seed(seed)

    sim_params = dict(SIM_DEFAULTS, n=n, outcome_type=OUTCOME_TYPE)
    tp = _trainer_params(hp)
    t0 = time.time()

    full, half_A, half_B = simulate_dataloaders_split(sim_params, seed=seed)
    full_tr, full_va = full
    hA_tr, hA_va = half_A
    hB_tr, hB_va = half_B

    arrays: dict[str, dict[str, np.ndarray]] = {}

    # Shared test draw (same for every method).
    test_sim = dict(sim_params, n=TEST_N)
    X_te, Z_te, y_te, fx_te, fz_te, _fr_te = simulate_traffic_light_data(**test_sim, seed=TEST_SEED)
    loader = DataLoader(CovarDataset(X_te, Z_te, y_te), batch_size=min(200, TEST_N), shuffle=False)

    truths = {
        "y":  y_te.view(-1).numpy().astype(np.float32),
        "fx": fx_te.view(-1).numpy().astype(np.float32),
        "fz": fz_te.view(-1).numpy().astype(np.float32),
    }

    # ── sgd: end-to-end CovarNetwork (linear f_z), MSE only ──────────────────
    m_sgd = covar_trainer(CovarNetwork, MODEL_PARAMS, full_tr, full_va, **tp)
    m_sgd = m_sgd.center_effects(full_tr)
    arrays["sgd"] = _gather(m_sgd, loader, device)
    del m_sgd

    # ── nam: end-to-end CovarNetworkMLPfz (MLP f_z), MSE only ────────────────
    m_nam = covar_trainer(CovarNetworkMLPfz, MODEL_PARAMS, full_tr, full_va, **tp)
    m_nam = m_nam.center_effects(full_tr)
    arrays["nam"] = _gather(m_nam, loader, device)
    del m_nam

    # ── nam_ridge_<λ>: the NAM, AdamW ridge on all params, one fit per λ ──────
    for lam in RIDGE_LAMBDAS:
        tp_ridge = dict(tp, weight_decay=lam, optimizer_cls=torch.optim.AdamW)
        m_ridge = covar_trainer(CovarNetworkMLPfz, MODEL_PARAMS, full_tr, full_va, **tp_ridge)
        m_ridge = m_ridge.center_effects(full_tr)
        arrays[f"nam_ridge_{lam:g}"] = _gather(m_ridge, loader, device)
        del m_ridge

    # ── posthoc / posthoc_xfit: shared backbones, then refit on half_B/A ────
    base_A = covar_trainer(BaseNetwork, MODEL_PARAMS,
                           train_loader=hA_tr, val_loader=hA_va, **tp).center_effects(hA_tr)
    base_B = covar_trainer(BaseNetwork, MODEL_PARAMS,
                           train_loader=hB_tr, val_loader=hB_va, **tp).center_effects(hB_tr)
    m_AB = _fit_posthoc(PostHocCovarNetwork, base_A, hB_tr, hB_va, {"orthogonalize": False}, {}, 1, device)
    m_BA = _fit_posthoc(PostHocCovarNetwork, base_B, hA_tr, hA_va, {"orthogonalize": False}, {}, 1, device)
    arrays["posthoc"] = _gather(m_AB, loader, device)
    arrays["posthoc_xfit"] = _gather(CrossFitAverageModel(m_AB, m_BA), loader, device)
    del base_A, base_B, m_AB, m_BA

    np.savez_compressed(
        npz_path,
        methods=np.array(list(arrays.keys())),
        **{f"{name}__{eff}": arr for name, d in arrays.items() for eff, arr in d.items()},
        **{f"truth__{eff}": arr for eff, arr in truths.items()},
        setting=np.array(json.dumps({"n": n})),
        sim_params=np.array(json.dumps({k: v for k, v in sim_params.items()})),
    )

    torch.cuda.empty_cache()
    gc.collect()

    return {"sweep_key": sweep_key, "seed": seed, "status": "ok",
            "wall_s": time.time() - t0, "n_methods": len(arrays)}


def _worker(task):
    try:
        return _run_one_sim(**task)
    except Exception as e:
        return {"sweep_key": f"n={task['n']}", "seed": task["seed"], "status": "error",
                "error": str(e), "traceback": traceback.format_exc()[-1500:]}


def _already_done(run_dir: Path) -> set:
    done = set()
    for sweep_dir in run_dir.glob("n=*"):
        if not sweep_dir.is_dir():
            continue
        for p in sweep_dir.glob("seed=*.npz"):
            try:
                done.add((sweep_dir.name, int(p.stem.split("=")[1])))
            except ValueError:
                continue
    return done


# ── Runner ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, default=None,
                    help="Resume an existing run dir. If None, create a new one.")
    ap.add_argument("--nsim", type=int, default=NSIM)
    ap.add_argument("--seed-start", type=int, default=0,
                    help="First seed, inclusive (shard a run across GPUs).")
    ap.add_argument("--seed-end", type=int, default=None,
                    help="Last seed, exclusive. Defaults to --nsim.")
    ap.add_argument("--n-values", type=str, default=None,
                    help="Comma-separated subset of the N grid (default: full grid).")
    ap.add_argument("--device", type=str, default=DEVICE,
                    help="CUDA device (e.g. 'cuda:0'). Overrides module default.")
    ap.add_argument("--outcome-type", type=str, default=OUTCOME_TYPE,
                    choices=["binary", "continuous"],
                    help="binary → logit link + BCE loss; continuous → identity + MSE.")
    args = ap.parse_args()

    # Override module-level DEVICE / OUTCOME_TYPE so spawned workers pick up the
    # chosen values at import time (they re-import this module and read the env).
    globals()["DEVICE"] = args.device
    os.environ["COCODEEL_DEVICE"] = args.device
    globals()["OUTCOME_TYPE"] = args.outcome_type
    os.environ["COCODEEL_OUTCOME"] = args.outcome_type

    if not HP_PATH.exists():
        print(f"Chosen HPs not found at {HP_PATH} — run simulation_images/hp_search.py first.",
              file=sys.stderr)
        sys.exit(1)
    hp = json.loads(HP_PATH.read_text())

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using {run_dir}")
    else:
        run_dir = setup_run_dir()
        print(f"New run dir: {run_dir}")
    # Concurrent shards point at the same dir; the first to arrive writes it.
    if not (run_dir / "manifest.json").exists():
        write_manifest(run_dir, hp)

    seed_end = args.seed_end if args.seed_end is not None else args.nsim
    n_values = [int(v) for v in args.n_values.split(",")] if args.n_values else N_GRID
    tasks = [dict(n=n, seed=seed, run_dir=run_dir, hp=hp)
             for n in n_values for seed in range(args.seed_start, seed_end)]
    done = _already_done(run_dir)
    tasks = [t for t in tasks if (f"n={t['n']}", t["seed"]) not in done]

    print(f"[{datetime.datetime.now():%H:%M:%S}] {len(tasks)} tasks to run "
          f"({len(done)} already complete). seeds[{args.seed_start},{seed_end}) "
          f"on {DEVICE}. Workers={N_WORKERS}.", flush=True)

    progress_path = run_dir / "progress.log"
    t_start = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=N_WORKERS) as pool:
        for i, res in enumerate(pool.imap_unordered(_worker, tasks), start=1):
            with progress_path.open("a") as f:
                f.write(json.dumps({"t": datetime.datetime.now().isoformat(), **res}) + "\n")
            tag = res.get("status", "?").upper()
            print(f"[{datetime.datetime.now():%H:%M:%S}] {i}/{len(tasks)} {tag}  "
                  f"{res.get('sweep_key','?')} seed={res.get('seed','?')} "
                  f"t={res.get('wall_s', 0.0):.1f}s"
                  + (f"  ERR: {res.get('error','')}" if tag == "ERROR" else ""),
                  flush=True)

    print(f"\n[{datetime.datetime.now():%H:%M:%S}] Done in "
          f"{(time.time() - t_start)/3600:.2f} h. Run: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
