#!/usr/bin/env python
"""Train the ResNet50 backbones of the UKBB application (Section 6).

Per synthetic confounding strength coef (beta_age = -coef) and outer CV fold, the outer training fold is
split into pools and each pool is resampled to the logistic DGP (common/data.py). The backbones are
DNNs without controls, phi plus a linear head, trained on highalc from the image alone:
  base_full    on all NTRAIN observations (No sample split, and the post-hoc competitors);
  base_half    on half h1, base_half_B on half h2 (Sample split and Cross-fit (K=2));
  k3/backbone_k{k}, k = 0, 1, 2, on all thirds but the k-th (Cross-fit (K=3)).
refit_from_checkpoints.py, run_crossfit.py and competitors_from_checkpoints.py refit their heads.

Usage:  python experiments/ukbb/train_backbones.py --gpu 0
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
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.ukbb.common.backbone import ResNet50
from experiments.ukbb.common.data import (
    RANDOM_STATE, BACKBONE_DIR, outer_fold, resample_partition, PRETRAINED_RESNET, seed_everything, load_ukbb_data,
    NumpyCovarDataset, fast_loader, default_transforms,
)
from cocodeel.model import BaseNetwork
from cocodeel.trainer import covar_trainer

# ── run config ────────────────────────────────────────────────────────────────
BATCH_SIZE = 48
# the full image array (~53 GB) is held in RAM and every call site slices it with fancy
# indexing, which copies; 16 forked workers inherit all of it and exhausted host memory
NUM_WORKERS = 4
LR = 1.67e-6
WEIGHT_DECAY = 1.02e-5
EPOCHS = 128
PATIENCE = 20
MODEL_PARAMS = dict(backbone=ResNet50, backbone_params={"pretrained_model": str(PRETRAINED_RESNET)},
                    num_covariates=0, link="identity")
K_CROSSFIT = 3

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--coefs", type=str, default="0.0,2.0")
parser.add_argument("--folds", type=str, default="0,1,2,3,4")
args = parser.parse_args()
COEFS = [float(c) for c in args.coefs.split(",")]
FOLDS = [int(f) for f in args.folds.split(",")]
OUT_RUN = BACKBONE_DIR

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
        train_ix = outer_fold(y, fold)

        # split the outer training fold into pools before resampling, so the parts are disjoint
        h1_idx, h2_idx = resample_partition(train_ix, y, Z, coef, fold_seed, K=2)
        thirds = resample_partition(train_ix, y, Z, coef, fold_seed, K=K_CROSSFIT)
        out_dir = OUT_RUN / f"coef={coef}" / f"fold={fold}"

        jobs = [("base_full", np.concatenate([h1_idx, h2_idx]), fold_seed + 200),
                ("base_half", h1_idx, fold_seed + 201),
                ("base_half_B", h2_idx, fold_seed + 201)]
        jobs += [(f"k{K_CROSSFIT}/backbone_k{k}", np.concatenate([t for j, t in enumerate(thirds) if j != k]),
                  fold_seed + 300 + k) for k in range(K_CROSSFIT)]
        for name, idx, seed in jobs:
            path = out_dir / f"{name}.pt"
            if path.exists():
                print(f"[{datetime.datetime.now():%H:%M:%S}] coef={coef} fold={fold} {name} cached — skip.", flush=True)
                continue
            print(f"\n[{datetime.datetime.now():%H:%M:%S}] ══ coef={coef} fold={fold} training {name} ══", flush=True)
            tr_ld, va_ld, y_tr = loaders(idx, seed)
            train_backbone(tr_ld, va_ld, y_tr, path)
print(f"\n[{datetime.datetime.now():%H:%M:%S}] Done. Backbones in {OUT_RUN}", flush=True)
