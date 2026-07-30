#!/usr/bin/env python
"""Source-of-truth backbone training for the UKBB sample-split experiment.

Trains the three ResNet50 backbones per (coef, fold) that the downstream refit stages consume:
base_full (on h1 u h2), base_half (on h1), base_half_B (on h2). Checkpoints are the irreplaceable
artefact — refit_from_checkpoints.py and run_crossfit.py regenerate every head and figure output
from them without retraining. This script is NOT part of the reproducible loop; it documents how
the released checkpoints were produced. Skip-if-exists resume; one backbone file per role.

Usage:  python experiments/ukbb/train_backbones.py --coefs 0.0,2.0 --folds 0,1,2,3,4
        add --int-coef/--rho-coef for the non-additive DGP variant, which writes to its
        own run dir (see nonadditivity_study.md)
"""
import sys
import gc
import argparse
import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import sklearn.utils.class_weight
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.ukbb.common.backbone import ResNet50
from experiments.ukbb.common.data import (
    RANDOM_STATE, PRETRAINED_RESNET, seed_everything, load_ukbb_data,
    resample_synthetic, NumpyCovarDataset, fast_loader, default_transforms,
)
from cocodeel.model import BaseNetwork
from cocodeel.trainer import covar_trainer

# ── run config ────────────────────────────────────────────────────────────────
N_SPLITS = 5
NTRAIN_PER_HALF = 2500
BATCH_SIZE = 48
NUM_WORKERS = 16
# tuned values from the released final_v2 run (fixed backbone recipe; not re-searched here)
LR = 1.67e-6
WEIGHT_DECAY = 1.02e-5
EPOCHS = 128
PATIENCE = 20
MODEL_PARAMS = dict(backbone=ResNet50, backbone_params={"pretrained_model": str(PRETRAINED_RESNET)},
                    num_covariates=0, link="identity")
OUT_RUN = ROOT / "experiments/ukbb/runs/final_v2"

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--coefs", type=str, default="0.0,2.0")
parser.add_argument("--folds", type=str, default="0,1,2,3,4")
parser.add_argument("--int-coef", type=float, default=0.0)
parser.add_argument("--rho-coef", type=float, default=0.0)
parser.add_argument("--out-run", type=str, default=None)
args = parser.parse_args()
COEFS = [float(c) for c in args.coefs.split(",")]
FOLDS = [int(f) for f in args.folds.split(",")]
DGP = dict(int_coef=args.int_coef, rho_coef=args.rho_coef)

# non-additive DGP variants land in their own run dir, so final_v2 is never written into
TAG = "final_v2" if not any(DGP.values()) else f"nonadd_int{args.int_coef:g}_rho{args.rho_coef:g}"
OUT_RUN = Path(args.out_run) if args.out_run else ROOT / "experiments/ukbb/runs" / TAG

# ── setup ─────────────────────────────────────────────────────────────────────
seed_everything(RANDOM_STATE)
device = torch.device(f"cuda:{args.gpu}")
tform = default_transforms()
d = load_ukbb_data()
X, y, Z = d["X"], d["y"], d["Z"]
print(f"[{datetime.datetime.now():%H:%M:%S}] device=cuda:{args.gpu}  out={OUT_RUN.name}", flush=True)


def loaders(idx, seed):
    """Stratified 80/20 train/val loaders over global indices; returns loaders and train labels."""
    tr, va = train_test_split(np.arange(len(idx)), test_size=0.2, random_state=seed, stratify=y[idx])
    tr_ld = fast_loader(NumpyCovarDataset(X[idx[tr]], y[idx[tr]], Z[idx[tr]], tform),
                        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    va_ld = fast_loader(NumpyCovarDataset(X[idx[va]], y[idx[va]], Z[idx[va]], tform),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return tr_ld, va_ld, y[idx[tr]]


def train_backbone(tr_ld, va_ld, y_tr, path):
    """Train a BaseNetwork with class-balanced BCE on raw logits, center, save as a logit model."""
    cw = sklearn.utils.class_weight.compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    pos_weight = torch.tensor(cw[1] / cw[0]).to(device)
    model = covar_trainer(
        model=BaseNetwork, model_params=MODEL_PARAMS,
        train_loader=tr_ld, val_loader=va_ld,
        device=f"cuda:{args.gpu}", epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY, patience=PATIENCE,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_kwargs={"patience": 5, "factor": 0.8}, use_amp=True,
        loss_fn=nn.BCEWithLogitsLoss(pos_weight=pos_weight),
    )
    # trained on raw logits (link=identity); switch to logit so model() returns probabilities
    model = model.center_effects(tr_ld)
    model.link = "logit"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[{datetime.datetime.now():%H:%M:%S}]   saved {path.relative_to(OUT_RUN)}  "
          f"best_epoch={model.best_epoch_}", flush=True)
    del model
    torch.cuda.empty_cache()
    gc.collect()


# ── main ──────────────────────────────────────────────────────────────────────
for coef in COEFS:
    for fold in FOLDS:
        fold_seed = RANDOM_STATE + fold
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        train_ix = list(cv.split(np.zeros(len(y)), y))[fold][0]

        # partition (split pool before resampling)
        pool_A, pool_B = train_test_split(train_ix, test_size=0.5, random_state=fold_seed, stratify=y[train_ix])
        h1_idx = pool_A[resample_synthetic(y[pool_A], Z[pool_A], NTRAIN_PER_HALF, coef, fold_seed, **DGP)]
        h2_idx = pool_B[resample_synthetic(y[pool_B], Z[pool_B], NTRAIN_PER_HALF, coef, fold_seed + 100, **DGP)]
        full_idx = np.concatenate([h1_idx, h2_idx])
        out_dir = OUT_RUN / f"coef={coef}" / f"fold={fold}"

        # backbones: base_full (h1 u h2), base_half (h1), base_half_B (h2)
        for name, idx, seed in [("base_full", full_idx, fold_seed + 200),
                                ("base_half", h1_idx, fold_seed + 201),
                                ("base_half_B", h2_idx, fold_seed + 201)]:
            path = out_dir / f"{name}.pt"
            if path.exists():
                print(f"[{datetime.datetime.now():%H:%M:%S}] coef={coef} fold={fold} {name} cached — skip.", flush=True)
                continue
            print(f"\n[{datetime.datetime.now():%H:%M:%S}] ══ coef={coef} fold={fold} training {name} ══", flush=True)
            tr_ld, va_ld, y_tr = loaders(idx, seed)
            train_backbone(tr_ld, va_ld, y_tr, path)
print(f"\n[{datetime.datetime.now():%H:%M:%S}] Done. Backbones in {OUT_RUN}", flush=True)
