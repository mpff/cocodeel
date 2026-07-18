"""HP search for the concurvity benchmark: per-N anchors, replicated draws.

Three log-spaced anchors span N_GRID_CONCURVITY; every other n inherits the
nearest anchor's winners (log scale) in the study script. Per anchor, three
trained method classes (backbone, nam, nam_mlp) sweep an lr x wd grid and
cfnet sweeps its shared learning rate at lam=1; each combo is fit on R=5
independent simulation draws and selected by mean best validation
prediction loss. Regularization strengths are never tuned here — they are
reported as method variants. Writes search_study_c.csv and
chosen_hps_study_c.json; the winners are hardcoded in the study script.

Usage:  python experiments/simulation/hpsearch/search_study_c.py
"""
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import MSELoss

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from cocodeel.model import BaseNetwork
from cocodeel.trainer import covar_trainer
from cocodeel.benchmarking.model import CovarNetwork, MLPCovarNetwork
from cocodeel.benchmarking.adversarial_trainer import adversarial_trainer

from experiments.simulation.common.backbone import TrafficBackbone
from experiments.simulation.common.loaders import simulate_dataloaders_split
from experiments.simulation.hpsearch import _grid_search


# ── search config ─────────────────────────────────────────────────────────────
DEVICE = os.environ.get("COCODEEL_DEVICE", "cuda:0")
N_WORKERS = 4
Q = 32
N_DRAWS = 5
DRAW_SEED_BASE = 1000
EPOCHS_CAP = 1000

ANCHOR_NS = [400, 6400, 102400]
LRS = [3e-4, 1e-3, 3e-3, 1e-2]
WEIGHT_DECAYS = [1e-5, 1e-4]
CFNET_LRS = [1e-4, 2e-4, 1e-3]

SIM_DEFAULTS = dict(bz=1., b2=1., b3=1., cv1=0.5, cv2=0.5, sdy=1.,
                    outcome_type="continuous")

OUT_DIR = ROOT / "experiments/simulation/output/hp_search"
CHOSEN_PATH = Path(__file__).resolve().parent / "chosen_hps_study_c.json"

MODEL_CLASSES = {"backbone": BaseNetwork, "nam": CovarNetwork, "nam_mlp": MLPCovarNetwork}


# ── one fit ───────────────────────────────────────────────────────────────────
def fit_one(task):
    kind, n, hp, draw = task["kind"], task["n"], task["hp"], task["draw"]
    try:
        device = torch.device(DEVICE)
        seed = DRAW_SEED_BASE + draw
        torch.manual_seed(seed)

        # data (same draw for every combo: paired comparison)
        full, _, _, _ = simulate_dataloaders_split(dict(SIM_DEFAULTS, n=n), seed=seed)
        full_tr, full_va = full
        model_params = dict(
            backbone=TrafficBackbone, backbone_params={"out_features": Q},
            num_covariates=1, link="identity",
        )

        # fit
        t0 = time.time()
        if kind == "cfnet":
            model = adversarial_trainer(
                BaseNetwork, model_params, num_covariates=1,
                train_loader=full_tr, val_loader=full_va,
                device=device, loss_fn=MSELoss(), epochs=EPOCHS_CAP,
                lr_task=hp["lr"], lr_cp=hp["lr"], lr_adv=hp["lr"], lam=1.0,
            )
        else:
            model = covar_trainer(
                MODEL_CLASSES[kind], model_params,
                train_loader=full_tr, val_loader=full_va,
                device=device, loss_fn=MSELoss(), epochs=EPOCHS_CAP,
                lr=hp["lr"], weight_decay=hp["wd"], patience=6,
                scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
                scheduler_kwargs={"mode": "min", "patience": 5, "factor": 0.5},
                use_amp=True,
            )
        return dict(
            kind=kind, n=n, draw=draw, seed=seed, **hp,
            val_loss=model.val_losses_[model.best_epoch_],
            best_epoch=model.best_epoch_, n_epochs=model.n_epochs_run_,
            wall_s=time.time() - t0, status="ok",
        )
    except Exception as e:
        import traceback
        return dict(kind=kind, n=n, draw=draw, **hp, error=str(e),
                    traceback=traceback.format_exc()[-1000:], status="error")


# ── selection ─────────────────────────────────────────────────────────────────
def choose_winners(rows):
    """Per (anchor, kind): the combo with the lowest mean val loss over draws."""
    winners = {}
    for n in ANCHOR_NS:
        winners[str(n)] = {}
        for kind in list(MODEL_CLASSES) + ["cfnet"]:
            combos = {}
            for r in rows:
                if r["n"] != n or r["kind"] != kind or r["status"] != "ok":
                    continue
                key = (r["lr"], r.get("wd"))
                combos.setdefault(key, []).append(r["val_loss"])
            means = {k: float(np.mean(v)) for k, v in combos.items()}
            (lr, wd), val = min(means.items(), key=lambda kv: kv[1])
            winner = dict(lr=lr, val_loss=val, n_draws=len(combos[(lr, wd)]))
            if wd is not None:
                winner["wd"] = wd
            winners[str(n)][kind] = winner
    return winners


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    grid = [dict(lr=lr, wd=wd) for lr, wd in itertools.product(LRS, WEIGHT_DECAYS)]
    tasks = [dict(kind=kind, n=n, hp=hp, draw=d)
             for n in ANCHOR_NS
             for kind, hps in [("backbone", grid), ("nam", grid), ("nam_mlp", grid),
                               ("cfnet", [dict(lr=lr) for lr in CFNET_LRS])]
             for hp in hps
             for d in range(N_DRAWS)]
    # largest fits first so the pool tail is short
    tasks.sort(key=lambda t: -t["n"])
    rows = _grid_search.run_pool(
        tasks, fit_one, N_WORKERS,
        describe=lambda r: (f"{r.get('kind')} n={r.get('n')} lr={r.get('lr')} "
                            f"draw={r.get('draw')} {r.get('status')} "
                            f"val={r.get('val_loss', float('nan')):.4f}"))
    _grid_search.write_csv(rows, OUT_DIR / "search_study_c.csv")
    winners = choose_winners(rows)
    CHOSEN_PATH.write_text(json.dumps(winners, indent=2))
    print(f"Wrote {CHOSEN_PATH}:\n" + json.dumps(winners, indent=2), flush=True)


if __name__ == "__main__":
    main()
