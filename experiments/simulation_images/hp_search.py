"""Hyperparameter search for the paper simulation study runner.

Goal: find training-speed-friendly values of `lr`, `weight_decay`,
early-stopping patience, and scheduler patience that do not hurt MSPE.
The chosen combo is persisted to `chosen_hps.json` and baked into the
main simulation runner's config.

Protocol
--------
- 4 anchors: outcome ∈ {continuous, binary} × n ∈ {1600, 25600}.
- Grid: lr × weight_decay × early_patience × sched_patience (3×2×3×2 = 36).
- 1 seed per (combo, anchor) (user instruction).
- 4 workers on cuda:1 via torch.multiprocessing spawn.
- Per run: train base_full + base_half + posthoc (split recipe), record
  wall time, epochs trained, MSPE(f̂_X) on a shared test draw.

Selection rule (per outcome type)
---------------------------------
Among combos whose MSPE(f̂_X) on `posthoc` is within 5 % of the best combo,
pick the combo with the shortest total wall time across the two N anchors.

Usage
-----
    /home/RDC/pfeuffma/.conda/envs/dl-mri/bin/python \
        experiments/simulation_images/hp_search.py

Outputs
-------
    results/simulation_images/hp_search/hp_search.csv
    results/simulation_images/hp_search/chosen_hps.json
    results/simulation_images/hp_search/manifest.json
"""
from __future__ import annotations

import argparse
import csv
import datetime
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
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
DEVICE = "cuda:1"
N_WORKERS = 4
Q = 32
SEED = 0

# Grid: 3 * 2 * 3 * 2 = 36 combos
LRS           = [3e-4, 1e-3, 3e-3]
WEIGHT_DECAYS = [1e-5, 1e-4]
EARLY_PATS    = [6, 10, 16]
SCHED_PATS    = [3, 5]

# Anchors: 4 points that span the full-run regime.
ANCHORS = [
    dict(outcome_type="continuous", n=1600,  bz=1.0, b2=1., b3=1., cv1=0.5, cv2=0.5, sdy=1.),
    dict(outcome_type="continuous", n=25600, bz=1.0, b2=1., b3=1., cv1=0.5, cv2=0.5, sdy=1.),
    dict(outcome_type="binary",     n=1600,  bz=1.0, b2=1., b3=1., cv1=0.5, cv2=0.5, sdy=1.),
    dict(outcome_type="binary",     n=25600, bz=1.0, b2=1., b3=1., cv1=0.5, cv2=0.5, sdy=1.),
]

TEST_SEED = 1234
TEST_N = 800
EPOCHS_CAP = 1000   # never train longer than this

OUT_DIR = ROOT / "results/simulation_images/hp_search"


# ── Fit one anchor under one combo ────────────────────────────────────────────
def _fit_one(task):
    combo_id, lr, wd, early_pat, sched_pat, anchor = task
    sim_params = {k: v for k, v in anchor.items()}
    outcome_type = sim_params["outcome_type"]
    device = torch.device(DEVICE)

    torch.manual_seed(SEED)
    mp_seed = SEED + hash((combo_id, anchor["n"])) % (2**31)
    torch.manual_seed(mp_seed)

    full, half_A, half_B = simulate_dataloaders_split(sim_params, seed=mp_seed)
    full_tr, full_va = full
    hA_tr,  hA_va  = half_A
    hB_tr,  hB_va  = half_B

    mp_ = dict(
        backbone=TrafficBackbone,
        backbone_params={"out_features": Q},
        num_covariates=1,
        link=("logit" if outcome_type == "binary" else "identity"),
    )
    # BCELoss is not autocast-safe (PyTorch refuses under bf16).  Disable AMP
    # for binary; continuous/MSE is autocast-safe and benefits from bf16.
    tp = dict(
        device=device,
        loss_fn=BCELoss() if outcome_type == "binary" else MSELoss(),
        epochs=EPOCHS_CAP,
        lr=lr,
        weight_decay=wd,
        patience=early_pat,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_kwargs={"mode": "min", "patience": sched_pat, "factor": 0.5},
        use_amp=(outcome_type != "binary"),
    )

    t0 = time.time()
    base_full = covar_trainer(BaseNetwork, mp_, train_loader=full_tr, val_loader=full_va, **tp)
    base_full = base_full.center_effects(full_tr)
    t_base_full = time.time() - t0

    t1 = time.time()
    base_half = covar_trainer(BaseNetwork, mp_, train_loader=hA_tr, val_loader=hA_va, **tp)
    base_half = base_half.center_effects(hA_tr)
    t_base_half = time.time() - t1

    fit_kwargs = dict(max_iters=100, tol=1e-4) if outcome_type == "binary" else dict()
    t2 = time.time()
    phm = PostHocCovarNetwork(base_half, num_covariates=1, orthogonalize=False).to(device)
    phm = phm.fit(hB_tr, hB_va, **fit_kwargs)
    t_posthoc = time.time() - t2

    # Evaluate on the shared unconfounded test draw.
    X_te, Z_te, y_te, fx_te, fz_te, fr_te = simulate_traffic_light_data(
        n=TEST_N, bz=sim_params["bz"], b2=sim_params["b2"], b3=sim_params["b3"],
        cv1=sim_params["cv1"], cv2=sim_params["cv2"], sdy=sim_params["sdy"],
        outcome_type=outcome_type, seed=TEST_SEED,
    )
    loader = DataLoader(CovarDataset(X_te, Z_te, y_te), batch_size=min(200, TEST_N), shuffle=False)

    def _fx_hat(model):
        model.eval(); model.to(device)
        xs = []
        with torch.no_grad():
            for b in loader:
                xs.append(model.predict_fx(b["X"].to(device), b["Z"].to(device)).cpu())
        return torch.cat(xs, dim=0).view(-1).numpy()

    fx_hat_full = _fx_hat(base_full)
    fx_hat_ph   = _fx_hat(phm)
    fx = fx_te.view(-1).numpy()

    return dict(
        combo_id=combo_id, lr=lr, wd=wd, early_pat=early_pat, sched_pat=sched_pat,
        outcome_type=outcome_type, n=anchor["n"], seed=mp_seed,
        t_base_full_s=t_base_full, t_base_half_s=t_base_half, t_posthoc_s=t_posthoc,
        t_total_s=t_base_full + t_base_half + t_posthoc,
        epochs_base_full=int(getattr(base_full, "best_epoch_", -1)),
        epochs_base_half=int(getattr(base_half, "best_epoch_", -1)),
        mspe_fx_base_full=float(((fx_hat_full - fx) ** 2).mean()),
        mspe_fx_posthoc=float(((fx_hat_ph   - fx) ** 2).mean()),
    )


# ── Worker driver ─────────────────────────────────────────────────────────────
def _worker(args):
    try:
        return _fit_one(args)
    except Exception as e:
        import traceback
        return dict(error=str(e), traceback=traceback.format_exc(), task=repr(args))


# ── Selection ─────────────────────────────────────────────────────────────────
def _choose_winners(rows):
    """Per outcome type, find best combo: fastest with MSPE within 5 % of best."""
    winners = {}
    for outcome in ("continuous", "binary"):
        rs = [r for r in rows if r.get("outcome_type") == outcome and "error" not in r]
        combos = {}
        for r in rs:
            cid = r["combo_id"]
            c = combos.setdefault(cid, {"combo_id": cid, "lr": r["lr"], "wd": r["wd"],
                                         "early_pat": r["early_pat"], "sched_pat": r["sched_pat"],
                                         "t_sum": 0.0, "mspe": []})
            c["t_sum"] += r["t_total_s"]
            c["mspe"].append(r["mspe_fx_posthoc"])
        if not combos:
            winners[outcome] = {"outcome_type": outcome, "error": "no successful runs"}
            continue
        for c in combos.values():
            c["mspe_mean"] = float(np.mean(c["mspe"]))
        best_mspe = min(c["mspe_mean"] for c in combos.values())
        tol = 1.05 * best_mspe
        eligible = [c for c in combos.values() if c["mspe_mean"] <= tol]
        winner = min(eligible, key=lambda c: c["t_sum"])
        winner["outcome_type"] = outcome
        winner["best_mspe"] = best_mspe
        winner["tol_cap"] = tol
        winners[outcome] = winner
    return winners


# ── Runner ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(OUT_DIR))
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    combos = list(itertools.product(LRS, WEIGHT_DECAYS, EARLY_PATS, SCHED_PATS))
    tasks = []
    for i, (lr, wd, ep, sp) in enumerate(combos):
        for anchor in ANCHORS:
            tasks.append((i, lr, wd, ep, sp, anchor))
    n_tasks = len(tasks)

    manifest = {
        "start": datetime.datetime.now().isoformat(),
        "host": os.uname().nodename,
        "device": DEVICE, "n_workers": N_WORKERS,
        "grid": {"lr": LRS, "weight_decay": WEIGHT_DECAYS,
                 "early_patience": EARLY_PATS, "sched_patience": SCHED_PATS},
        "anchors": ANCHORS, "q": Q, "seeds": 1,
        "epochs_cap": EPOCHS_CAP,
        "n_combos": len(combos), "n_tasks": n_tasks,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    ctx = mp.get_context("spawn")
    t0 = time.time()
    print(f"[{datetime.datetime.now():%H:%M:%S}] launching {n_tasks} tasks "
          f"({len(combos)} combos × {len(ANCHORS)} anchors) on {N_WORKERS} workers",
          flush=True)

    rows = []
    with ctx.Pool(processes=N_WORKERS) as pool:
        for i, res in enumerate(pool.imap_unordered(_worker, tasks), start=1):
            rows.append(res)
            tag = "ERR" if "error" in res else "OK"
            summary = (f"combo={res['combo_id']} n={res.get('n','?')} "
                       f"outcome={res.get('outcome_type','?')} "
                       f"t={res.get('t_total_s',float('nan')):.1f}s "
                       f"mspe_ph={res.get('mspe_fx_posthoc',float('nan')):.4f}"
                       if tag == "OK" else res.get("error", "?"))
            print(f"[{datetime.datetime.now():%H:%M:%S}] {i}/{n_tasks} {tag}  {summary}",
                  flush=True)

    fieldnames = sorted({k for r in rows for k in r.keys()})
    with (out / "hp_search.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[{datetime.datetime.now():%H:%M:%S}] wrote hp_search.csv", flush=True)

    winners = _choose_winners([r for r in rows if "error" not in r])
    (out / "chosen_hps.json").write_text(json.dumps(winners, indent=2))
    print(f"[{datetime.datetime.now():%H:%M:%S}] wrote chosen_hps.json:\n"
          + json.dumps(winners, indent=2), flush=True)
    print(f"Total wall: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
