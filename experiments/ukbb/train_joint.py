#!/usr/bin/env python
"""Joint fusion model: a DNN that takes the image X and the covariate Z as explicit inputs.

Unlike the additive refit/cross-fit estimators (eta = intercept + f_X(X) + f_Z(Z)), this baseline
concatenates the ResNet50 image features with age and passes them through an MLP head, so the
network can learn image x age interactions (non-additive). Like base_full, the backbone starts
from the Kinetics-pretrained init and is trained end-to-end, here jointly with age. Because there is no additive f_Z to drop, age is marginalised by averaging predictions over
the training age distribution:
    y_marg(x_i) = mean_j sigmoid( DNN(x_i, age_j) ).
Writes one row per (coef, fold) to joint_results.csv (method "joint_age") for the figure's
"Joint DNN" box.

Usage:  python experiments/ukbb/train_joint.py --gpu 0
"""
import sys
import gc
import csv
import argparse
import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import sklearn.utils.class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.ukbb.common.backbone import ResNet50
from experiments.ukbb.common.data import (
    RANDOM_STATE, NTEST, RUN_DIR, outer_fold, resample_partition, PRETRAINED_RESNET, seed_everything, load_ukbb_data,
    resample_synthetic, NumpyCovarDataset, fast_loader, default_transforms,
)
from cocodeel.links import LINKS
from cocodeel.trainer import covar_trainer

# ── run config ────────────────────────────────────────────────────────────────
BATCH_SIZE = 48
NUM_WORKERS = 16
HIDDEN = 64                    # MLP fusion head width
# the fresh MLP head needs a larger LR than the 1.67e-6 of the base_full training; 1e-4 diverges
LR = 1e-5
WEIGHT_DECAY = 1e-5
EPOCHS = 100
PATIENCE = 15
OUT_RUN = RUN_DIR / "refit"

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--coefs", type=str, default="0.0,2.0")
parser.add_argument("--folds", type=str, default="0,1,2,3,4")
args = parser.parse_args()
COEFS = [float(c) for c in args.coefs.split(",")]
FOLDS = [int(f) for f in args.folds.split(",")]

# ── setup ─────────────────────────────────────────────────────────────────────
seed_everything(RANDOM_STATE)
device = torch.device(f"cuda:{args.gpu}")
tform = default_transforms()
d = load_ukbb_data()
X, y, Z = d["X"], d["y"], d["Z"]
X_test, y_test, Z_test = d["X_test"], d["y_test"], d["Z_test"]
idx_test = resample_synthetic(y_test, Z_test, NTEST, 0.0, RANDOM_STATE)
X_te, y_te, Z_te = X_test[idx_test], y_test[idx_test], Z_test[idx_test]
print(f"[{datetime.datetime.now():%H:%M:%S}] device=cuda:{args.gpu}  out={OUT_RUN.name}", flush=True)


# ── fusion model ──────────────────────────────────────────────────────────────
class FusionNetwork(nn.Module):
    """Late-fusion joint model: ResNet50 image features concatenated with Z into an MLP head."""

    def __init__(self, backbone, backbone_params=None, num_covariates=1, link="identity",
                 hidden=HIDDEN):
        super().__init__()
        self.num_covariates = num_covariates
        self.link = link
        self._link = LINKS[link]
        self.backbone = backbone(**(backbone_params or {}))
        nf = self.backbone.out_features
        self.head = nn.Sequential(
            nn.Linear(nf + num_covariates, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, z):
        eta = self.head(torch.cat([self.backbone(x), z], dim=1))
        return self._link.inverse(eta)


# ── data + standardization helpers ────────────────────────────────────────────
def std_age(a, mean, std):
    """Standardize raw age with the training mean/std (age enters the MLP alongside 2048 features)."""
    return ((a.astype(np.float32) - mean) / std).reshape(-1, 1)


def loaders(idx, seed, mean, std):
    tr, va = train_test_split(np.arange(len(idx)), test_size=0.2, random_state=seed, stratify=y[idx])
    tr_ld = fast_loader(NumpyCovarDataset(X[idx[tr]], y[idx[tr]], std_age(Z[idx[tr], 0], mean, std), tform),
                        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    va_ld = fast_loader(NumpyCovarDataset(X[idx[va]], y[idx[va]], std_age(Z[idx[va], 0], mean, std), tform),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return tr_ld, va_ld, y[idx[tr]]


# ── evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def joint_metrics(model, coef, fold, mean, std, grid_ages):
    """Regular AUC (test's own age) + age-marginalised AUC (mean sigmoid over the training age grid)."""
    model.eval()
    test_ld = fast_loader(NumpyCovarDataset(X_te, y_te, std_age(Z_te[:, 0], mean, std), tform),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    feats, y_reg = [], []
    for b in test_ld:
        x = b["X"].to(device, non_blocking=True)
        z = b["Z"].to(device, non_blocking=True)
        f = model.backbone(x)
        feats.append(f)
        y_reg.append(torch.sigmoid(model.head(torch.cat([f, z], dim=1))).cpu().numpy())
    feats = torch.cat(feats, dim=0)
    y_reg = np.concatenate(y_reg).flatten()
    # marginalize each test image over the training age grid
    zg = torch.tensor(std_age(grid_ages, mean, std), device=device)
    y_marg = np.empty(feats.shape[0], dtype=np.float32)
    G = zg.shape[0]
    for i0 in range(0, feats.shape[0], 32):
        fb = feats[i0:i0 + 32]
        cat = torch.cat([fb[:, None, :].expand(-1, G, -1), zg[None].expand(fb.shape[0], -1, -1)], dim=-1)
        p = torch.sigmoid(model.head(cat.reshape(-1, cat.shape[-1]))).reshape(fb.shape[0], G)
        y_marg[i0:i0 + 32] = p.mean(dim=1).cpu().numpy()
    return dict(method="joint_age", coef=coef, fold=fold,
                auc=float(roc_auc_score(y_te, y_reg)),
                bacc=float(balanced_accuracy_score(y_te, (y_reg >= 0.5).astype(int))),
                auc_marg=float(roc_auc_score(y_te, y_marg)),
                bacc_marg=float(balanced_accuracy_score(y_te, (y_marg >= 0.5).astype(int))),
                corr_age=float("nan"), corr_sex=float("nan"),
                b_age=float("nan"), b_sex=float("nan"), lam=float("nan"))


# ── one (coef, fold) ──────────────────────────────────────────────────────────
def run_fold(coef, fold):
    fold_seed = RANDOM_STATE + fold
    train_ix = outer_fold(y, fold)

    # split the outer training fold into two pools before resampling, so the halves are disjoint
    h1, h2 = resample_partition(train_ix, y, Z, coef, fold_seed, K=2)
    full_idx = np.concatenate([h1, h2])
    mean, std = float(Z[full_idx, 0].mean()), float(Z[full_idx, 0].std())

    out_dir = OUT_RUN / f"coef={coef}" / f"fold={fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "joint_age.pt"
    if ckpt.exists():
        # resume: reload a trained fusion model, no retraining
        model = FusionNetwork(ResNet50, {"pretrained_model": ""}, num_covariates=1,
                              link="logit", hidden=HIDDEN).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"[{datetime.datetime.now():%H:%M:%S}]   loaded existing checkpoint", flush=True)
    else:
        tr_ld, va_ld, y_tr = loaders(full_idx, fold_seed + 200, mean, std)
        cw = sklearn.utils.class_weight.compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
        pos_weight = torch.tensor(cw[1] / cw[0]).to(device)
        # backbone from the Kinetics-pretrained init, trained jointly with the head on (X, age)
        model_params = dict(backbone=ResNet50, backbone_params={"pretrained_model": str(PRETRAINED_RESNET)},
                            num_covariates=1, link="identity", hidden=HIDDEN)
        model = covar_trainer(
            model=FusionNetwork, model_params=model_params, train_loader=tr_ld, val_loader=va_ld,
            device=f"cuda:{args.gpu}", epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY, patience=PATIENCE,
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau, scheduler_kwargs={"patience": 3, "factor": 0.5},
            use_amp=True, loss_fn=nn.BCEWithLogitsLoss(pos_weight=pos_weight),
        )
        vl = model.val_losses_
        print(f"[{datetime.datetime.now():%H:%M:%S}]   trained best_epoch={model.best_epoch_} n={model.n_epochs_run_} "
              f"val[:6]={[round(v, 4) for v in vl[:6]]} best={min(vl):.4f}", flush=True)
        torch.save(model.state_dict(), ckpt)

    # marginalized evaluation over the training age grid
    row = joint_metrics(model, coef, fold, mean, std, Z[full_idx, 0])
    print(f"[{datetime.datetime.now():%H:%M:%S}]   joint_age: auc={row['auc']:.4f} "
          f"auc_marg={row['auc_marg']:.4f}", flush=True)
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return row


# ── main (append to joint_results.csv, resumable per fold) ─────────────────────
OUT_CSV = OUT_RUN / "joint_results.csv"
FIELDS = ["method", "coef", "fold", "auc", "bacc", "auc_marg", "bacc_marg",
          "corr_age", "corr_sex", "b_age", "b_sex", "lam"]
OUT_RUN.mkdir(parents=True, exist_ok=True)
done = set()
if OUT_CSV.exists():
    for r in csv.DictReader(open(OUT_CSV)):
        done.add((float(r["coef"]), int(r["fold"])))
else:
    with open(OUT_CSV, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()
n_written = 0
for coef in COEFS:
    for fold in FOLDS:
        if (coef, fold) in done:
            print(f"[{datetime.datetime.now():%H:%M:%S}] coef={coef} fold={fold} in results — skip.", flush=True)
            continue
        print(f"\n[{datetime.datetime.now():%H:%M:%S}] ══ coef={coef} fold={fold} ══", flush=True)
        # run_fold reloads a fold's checkpoint if present, else trains; row appended immediately (resumable)
        row = run_fold(coef, fold)
        with open(OUT_CSV, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        n_written += 1
print(f"\n[{datetime.datetime.now():%H:%M:%S}] Done. Wrote {n_written} rows to {OUT_CSV}", flush=True)
