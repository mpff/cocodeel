#!/usr/bin/env python
"""K-fold cross-fit from released backbone checkpoints, ensembled with CrossFitEnsemble.

For each (coef, outer fold) this loads the per-rotation backbones (each trained on the union
of the other inner folds), refits a linear f_Z head on the held-out inner fold, and forms the
cross-fit ensemble eta = (1/K) sum_k eta_k. It also reports the no-sample-split refit (head and
backbone share all NTRAIN obs) and the single-split refit (rotation k=0 alone) — the two biased
estimators the figure contrasts against the cross-fit. Backbones are never retrained. aggregate.py
reduces the records to k{K}_crossfit_results.csv.

Usage:  python experiments/ukbb/run_crossfit.py --K 2 --run experiments/ukbb/runs/2026-05-03_16-43-32_n5k_noreplace_k2 \
            --ntrain 5000 --no-replace --coefs 0.0,2.0 --folds 0,1,2,3,4
"""
import gc
import sys
import json
import argparse
import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.ukbb.common.backbone import ResNet50
from experiments.ukbb.common.data import (
    RANDOM_STATE, seed_everything, load_ukbb_data, resample_synthetic,
    NumpyCovarDataset, fast_loader, default_transforms,
)
from cocodeel.model import BaseNetwork
from cocodeel.refit_model import RefitCovarNetwork
from cocodeel.crossfit import CrossFitEnsemble

# ── run config ────────────────────────────────────────────────────────────────
N_SPLITS = 5                   # outer CV folds
NTEST = 2500
BATCH_SIZE = 48
NUM_WORKERS = 16
REFIT_MAX_ITERS = 400          # IRLS backfitting controls (match the released run)
REFIT_TOL = 1e-3
MODEL_PARAMS = dict(backbone=ResNet50, backbone_params={"pretrained_model": ""},
                    num_covariates=0, link="logit")

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--K", type=int, default=2)
parser.add_argument("--coefs", type=str, default="0.0,2.0")
parser.add_argument("--folds", type=str, default="0,1,2,3,4")
parser.add_argument("--run", type=str, required=True, help="run dir holding coef=*/fold=*/{full,kK}/ checkpoints")
parser.add_argument("--ntrain", type=int, default=5000)
parser.add_argument("--no-replace", dest="replace", action="store_false")
parser.add_argument("--replace", dest="replace", action="store_true")
parser.set_defaults(replace=False)
args = parser.parse_args()
K = args.K
COEFS = [float(c) for c in args.coefs.split(",")]
FOLDS = [int(f) for f in args.folds.split(",")]
NTRAIN = args.ntrain
REPLACE = args.replace
RUN = (ROOT / args.run) if not Path(args.run).is_absolute() else Path(args.run)
OUT_RUN = ROOT / "experiments/ukbb/runs" / f"{RUN.name}_refit"

# ── setup ─────────────────────────────────────────────────────────────────────
seed_everything(RANDOM_STATE)
device = torch.device(f"cuda:{args.gpu}")
tform = default_transforms()
print(f"[{datetime.datetime.now():%H:%M:%S}] device=cuda:{args.gpu}  K={K}  ntrain={NTRAIN}  "
      f"replace={REPLACE}  run={RUN.name}  out={OUT_RUN.name}", flush=True)

# ── data ──────────────────────────────────────────────────────────────────────
d = load_ukbb_data()
X, y, Z = d["X"], d["y"], d["Z"]
X_test, y_test, Z_test = d["X_test"], d["y_test"], d["Z_test"]
idx_test = resample_synthetic(y_test, Z_test, NTEST, 0.0, RANDOM_STATE)
X_te, y_te, Z_te = X_test[idx_test], y_test[idx_test], Z_test[idx_test]


def loader(idx, z_cols=None, shuffle=False):
    Zc = Z[idx] if z_cols is None else Z[idx][:, z_cols]
    return fast_loader(NumpyCovarDataset(X[idx], y[idx], Zc, tform),
                       batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=NUM_WORKERS)


def refit_loaders(idx, seed, z_cols):
    tr, va = train_test_split(np.arange(len(idx)), test_size=0.2, random_state=seed, stratify=y[idx])
    return loader(idx[tr], z_cols, shuffle=True), loader(idx[va], z_cols, shuffle=False)


def load_backbone(path):
    net = BaseNetwork(**MODEL_PARAMS).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    net.eval()
    return net


# ── metrics ───────────────────────────────────────────────────────────────────
def base_metrics(name, coef, fold, y_hat, fx_hat, y_marg=None, b_age=float("nan"), b_sex=float("nan")):
    row = dict(method=name, coef=coef, fold=fold,
               auc=float(roc_auc_score(y_te, y_hat)),
               bacc=float(balanced_accuracy_score(y_te, (y_hat >= 0.5).astype(int))),
               auc_marg=float("nan"), bacc_marg=float("nan"),
               corr_age=float(np.corrcoef(fx_hat, Z_te[:, 0])[0, 1]),
               corr_sex=float(np.corrcoef(fx_hat, Z_te[:, 1])[0, 1]),
               b_age=b_age, b_sex=b_sex, lam=float("nan"))
    if y_marg is not None:
        row["auc_marg"] = float(roc_auc_score(y_te, y_marg))
        row["bacc_marg"] = float(balanced_accuracy_score(y_te, (y_marg >= 0.5).astype(int)))
    return row


@torch.no_grad()
def predict_dnn(model, test_ld):
    """y = sigmoid(intercept + fx) and fx for an uncontrolled backbone."""
    model.eval()
    ys, fxs = [], []
    for b in test_ld:
        x = b["X"].to(device, non_blocking=True)
        ys.append(model(x).cpu().numpy())
        fxs.append(model.predict_fx(x).cpu().numpy())
    return np.concatenate(ys).flatten(), np.concatenate(fxs).flatten()


@torch.no_grad()
def ensemble_metrics(name, coef, fold, ens, ncov, test_ld, train_grid):
    """Cross-fit-ensemble predictions: regular AUC + AUC marginalized over the training-Z grid."""
    ens.eval()
    eta_reg, eta_x, fx = [], [], []
    for b in test_ld:
        x = b["X"].to(device, non_blocking=True)
        z = b["Z"].to(device, non_blocking=True)[:, :ncov]
        e = ens.predict_eta(x, z)
        eta_reg.append(e.cpu().numpy())
        # eta without fz: predict_eta(x, z) - predict_fz(z), exact since eta is linear in z
        eta_x.append((e - ens.predict_fz(z)).cpu().numpy())
        fx.append(ens.predict_fx(x).cpu().numpy())
    eta_reg = np.concatenate(eta_reg).flatten()
    eta_x = np.concatenate(eta_x).flatten()
    fx = np.concatenate(fx).flatten()
    fz_train = []
    for b in train_grid:
        z = b["Z"].to(device, non_blocking=True)[:, :ncov]
        fz_train.append(ens.predict_fz(z).cpu().numpy())
    fz_train = np.concatenate(fz_train).flatten()
    y_reg = 1.0 / (1.0 + np.exp(-eta_reg))
    y_marg = (1.0 / (1.0 + np.exp(-(eta_x[:, None] + fz_train[None, :])))).mean(axis=1)
    w = np.mean([m.fz.weight.data.flatten().cpu().numpy() for m in ens.models], axis=0)
    b_age = float(w[0])
    b_sex = float(w[1]) if w.size > 1 else float("nan")
    return base_metrics(name, coef, fold, y_reg, fx, y_marg, b_age, b_sex)


# ── one (coef, fold) ──────────────────────────────────────────────────────────
def run_fold(coef, fold):
    fold_seed = RANDOM_STATE + fold
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    train_ix = list(cv.split(np.zeros(len(y)), y))[fold][0]

    # resample + inner K-fold partition (mirror the released run)
    full_idx = train_ix[resample_synthetic(y[train_ix], Z[train_ix], NTRAIN, coef, fold_seed, replace=REPLACE)]
    inner = StratifiedKFold(n_splits=K, shuffle=True, random_state=fold_seed)
    h_idx = [full_idx[fi] for _, fi in inner.split(np.zeros(NTRAIN), y[full_idx])]

    ckpt = RUN / f"coef={coef}" / f"fold={fold}"
    test_ld = fast_loader(NumpyCovarDataset(X_te, y_te, Z_te, tform),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    rows = []
    # dnn baseline (uncontrolled backbone on all NTRAIN)
    base_full = load_backbone(ckpt / "full" / "base_full.pt")
    y_dnn, fx_dnn = predict_dnn(base_full, test_ld)
    rows.append(base_metrics("dnn", coef, fold, y_dnn, fx_dnn))
    print(f"[{datetime.datetime.now():%H:%M:%S}]    {'dnn':>20}: auc={rows[-1]['auc']:.4f}", flush=True)
    del base_full

    for tag, z_cols in [("age", [0]), ("age_sex", [0, 1])]:
        ncov = len(z_cols)
        grid = loader(full_idx, z_cols=z_cols)  # recenter/marginalization grid at this covariate count
        # per-rotation refits over the loaded backbones
        members = []
        for k in range(K):
            bb = load_backbone(ckpt / f"k{K}" / f"backbone_k{k}.pt")
            tr, va = refit_loaders(h_idx[k], fold_seed + (2000 if ncov == 1 else 3000) + k, z_cols)
            m = RefitCovarNetwork(bb, num_covariates=ncov, orthogonalize=False).to(device)
            members.append(m.fit(tr, va, max_iters=REFIT_MAX_ITERS, tol=REFIT_TOL))
        # no-sample-split refit (head + backbone share all NTRAIN)
        bb_full = load_backbone(ckpt / "full" / "base_full.pt")
        tr, va = refit_loaders(full_idx, fold_seed + (700 if ncov == 1 else 800), z_cols)
        nosamp = RefitCovarNetwork(bb_full, num_covariates=ncov, orthogonalize=False).to(device)
        nosamp = nosamp.fit(tr, va, max_iters=REFIT_MAX_ITERS, tol=REFIT_TOL)

        # recenter each fitted model once on the full-training grid (leaves eta invariant);
        # members[0] feeds both refit_split and the cross-fit, so recenter it a single time
        for m in members:
            m.recenter(grid)
        nosamp.recenter(grid)
        rows.append(ensemble_metrics(f"refit_nosamp_{tag}", coef, fold,
                                     CrossFitEnsemble([nosamp]), ncov, test_ld, grid))
        rows.append(ensemble_metrics(f"refit_split_{tag}", coef, fold,
                                     CrossFitEnsemble([members[0]]), ncov, test_ld, grid))
        rows.append(ensemble_metrics(f"crossfit_k{K}_{tag}", coef, fold,
                                     CrossFitEnsemble(members), ncov, test_ld, grid))
        for r in rows[-3:]:
            print(f"[{datetime.datetime.now():%H:%M:%S}]    {r['method']:>20}: auc={r['auc']:.4f} "
                  f"auc_marg={r['auc_marg']:.4f} b_age={r['b_age']:+.4f} b_sex={r['b_sex']:+.4f}", flush=True)
        del members, nosamp, bb_full, grid

    out_dir = OUT_RUN / f"coef={coef}" / f"fold={fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "record.npz", coef=coef, fold=fold, K=K, rows=np.array(json.dumps(rows)))
    del test_ld
    torch.cuda.empty_cache()
    gc.collect()


# ── main ──────────────────────────────────────────────────────────────────────
for coef in COEFS:
    for fold in FOLDS:
        rec = OUT_RUN / f"coef={coef}" / f"fold={fold}" / "record.npz"
        if rec.exists():
            print(f"[{datetime.datetime.now():%H:%M:%S}] coef={coef} fold={fold} cached — skip.", flush=True)
            continue
        print(f"\n[{datetime.datetime.now():%H:%M:%S}] ══ K={K} coef={coef} fold={fold} ══", flush=True)
        run_fold(coef, fold)
print(f"\n[{datetime.datetime.now():%H:%M:%S}] Done. Records in {OUT_RUN}", flush=True)
