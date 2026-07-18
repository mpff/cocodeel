"""Default HP search: lr/wd/patience grid for the study runners, per outcome type.

Four anchors (outcome x n) span the sweep regime; per combo one backbone+refit
pipeline runs per anchor. Selection per outcome type: fastest combo whose
refit MSPE(f_X) is within 5% of the best. Writes chosen_hps.json; the winning
values are hardcoded in the study scripts.

Usage:  python experiments/simulation/hpsearch/search_default.py
"""
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import MSELoss, BCELoss
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from cocodeel.model import BaseNetwork
from cocodeel.refit_model import RefitCovarNetwork
from cocodeel.trainer import covar_trainer
from cocodeel.dataset import CovarDataset

from experiments.simulation.common.backbone import TrafficBackbone
from experiments.simulation.common.dgp import simulate_traffic_light_data
from experiments.simulation.common.loaders import simulate_dataloaders_split
from experiments.simulation.hpsearch import _grid_search


# ── search config ─────────────────────────────────────────────────────────────
DEVICE = os.environ.get("COCODEEL_DEVICE", "cuda:0")
N_WORKERS = 4
Q = 32
SEED = 0
TEST_SEED = 1234
TEST_N = 800
EPOCHS_CAP = 1000

LRS = [3e-4, 1e-3, 3e-3]
WEIGHT_DECAYS = [1e-5, 1e-4]
EARLY_PATS = [6, 10, 16]
SCHED_PATS = [3, 5]

ANCHORS = [
    dict(outcome_type=o, n=n, bz=1.0, b2=1., b3=1., cv1=0.5, cv2=0.5, sdy=1.)
    for o in ("continuous", "binary") for n in (1600, 25600)
]

OUT_DIR = ROOT / "experiments/simulation/output/hp_search"
CHOSEN_PATH = Path(__file__).resolve().parent / "chosen_hps.json"


def fit_one(task):
    combo_id, lr, wd, early_pat, sched_pat, anchor = task
    try:
        outcome_type = anchor["outcome_type"]
        device = torch.device(DEVICE)
        mp_seed = SEED + hash((combo_id, anchor["n"])) % (2 ** 31)
        torch.manual_seed(mp_seed)

        # data and params
        full, half_A, half_B, _ = simulate_dataloaders_split(anchor, seed=mp_seed)
        full_tr, full_va = full
        hA_tr, hA_va = half_A
        hB_tr, hB_va = half_B
        model_params = dict(
            backbone=TrafficBackbone, backbone_params={"out_features": Q},
            num_covariates=1,
            link=("logit" if outcome_type == "binary" else "identity"),
        )
        tp = dict(
            device=device,
            loss_fn=BCELoss() if outcome_type == "binary" else MSELoss(),
            epochs=EPOCHS_CAP, lr=lr, weight_decay=wd, patience=early_pat,
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
            scheduler_kwargs={"mode": "min", "patience": sched_pat, "factor": 0.5},
            use_amp=(outcome_type != "binary"),
        )

        # backbones + refit
        t0 = time.time()
        base_full = covar_trainer(BaseNetwork, model_params, train_loader=full_tr, val_loader=full_va, **tp)
        base_full = base_full.center_effects(full_tr)
        t_base_full = time.time() - t0
        t1 = time.time()
        base_half = covar_trainer(BaseNetwork, model_params, train_loader=hA_tr, val_loader=hA_va, **tp)
        base_half = base_half.center_effects(hA_tr)
        t_base_half = time.time() - t1
        t2 = time.time()
        fit_kwargs = dict(max_iters=100, tol=1e-4) if outcome_type == "binary" else {}
        refit = RefitCovarNetwork(base_half, num_covariates=1, orthogonalize=False).to(device)
        refit = refit.fit(hB_tr, hB_va, **fit_kwargs)
        t_refit = time.time() - t2

        # test MSPE(f_X)
        X_te, Z_te, y_te, fx_te, *_ = simulate_traffic_light_data(
            n=TEST_N, bz=anchor["bz"], b2=anchor["b2"], b3=anchor["b3"],
            cv1=anchor["cv1"], cv2=anchor["cv2"], sdy=anchor["sdy"],
            outcome_type=outcome_type, seed=TEST_SEED)
        loader = DataLoader(CovarDataset(X_te, Z_te, y_te), batch_size=200, shuffle=False)

        def mspe_fx(model):
            model.eval().to(device)
            xs = []
            with torch.no_grad():
                for b in loader:
                    xs.append(model.predict_fx(b["X"].to(device), b["Z"].to(device)).cpu())
            fx_hat = torch.cat(xs).view(-1).numpy()
            return float(((fx_hat - fx_te.view(-1).numpy()) ** 2).mean())

        return dict(
            combo_id=combo_id, lr=lr, wd=wd, early_pat=early_pat, sched_pat=sched_pat,
            outcome_type=outcome_type, n=anchor["n"], seed=mp_seed,
            t_total_s=t_base_full + t_base_half + t_refit,
            mspe_fx_base_full=mspe_fx(base_full), mspe_fx_refit=mspe_fx(refit),
            status="ok",
        )
    except Exception as e:
        import traceback
        return dict(combo_id=combo_id, error=str(e),
                    traceback=traceback.format_exc()[-1000:], status="error")


def choose_winners(rows):
    """Per outcome type: fastest combo with mean MSPE(f_X) within 5% of the best."""
    winners = {}
    for outcome in ("continuous", "binary"):
        combos = {}
        for r in rows:
            if r.get("outcome_type") != outcome or r["status"] != "ok":
                continue
            c = combos.setdefault(r["combo_id"], dict(
                combo_id=r["combo_id"], lr=r["lr"], wd=r["wd"],
                early_pat=r["early_pat"], sched_pat=r["sched_pat"],
                t_sum=0.0, mspe=[]))
            c["t_sum"] += r["t_total_s"]
            c["mspe"].append(r["mspe_fx_refit"])
        for c in combos.values():
            c["mspe_mean"] = float(np.mean(c["mspe"]))
        best = min(c["mspe_mean"] for c in combos.values())
        eligible = [c for c in combos.values() if c["mspe_mean"] <= 1.05 * best]
        winner = min(eligible, key=lambda c: c["t_sum"])
        winner["outcome_type"] = outcome
        winners[outcome] = winner
    return winners


def main():
    combos = list(itertools.product(LRS, WEIGHT_DECAYS, EARLY_PATS, SCHED_PATS))
    tasks = [(i, lr, wd, ep, sp, anchor)
             for i, (lr, wd, ep, sp) in enumerate(combos) for anchor in ANCHORS]
    rows = _grid_search.run_pool(
        tasks, fit_one, N_WORKERS,
        describe=lambda r: (f"combo={r.get('combo_id', '?')} {r.get('status', '?')} "
                            f"mspe_refit={r.get('mspe_fx_refit', float('nan')):.4f}"))
    _grid_search.write_csv(rows, OUT_DIR / "hp_search.csv")
    winners = choose_winners(rows)
    CHOSEN_PATH.write_text(json.dumps(winners, indent=2))
    print(f"Wrote {CHOSEN_PATH}:\n" + json.dumps(winners, indent=2), flush=True)


if __name__ == "__main__":
    main()
