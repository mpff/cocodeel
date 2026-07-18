#!/usr/bin/env python
"""K-fold cross-fit on UKBB — source of truth for the figure: dnn, refit_nosamp, refit_split, and the crossfit_k{K} ensemble per (coef, outer fold)."""
import os
import gc
import json
import argparse
import datetime
import sklearn.utils.class_weight

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from ukbb_common import (
    RANDOM_STATE,
    seed_everything, load_ukbb_data,
    resample_synthetic, NumpyCovarDataset,
    fast_loader,
    default_model_params, default_trainer_params, default_transforms,
)
from cocodeel.model import BaseNetwork
from cocodeel.refit_model import RefitCovarNetwork
from cocodeel.trainer import covar_trainer

# ── Config ────────────────────────────────────────────────────────────────────
N_SPLITS = 5                        # Crossvalidation folds (outer)
K = 2                               # Crossfit folds (inner)
NTRAIN = 5000                       # total resampled obs per outer fold
NTRAIN_PER_FOLD = NTRAIN // K       # per crossfit-fold draw
RESAMPLE_REPLACE = False            # Resample with or without replacement
NTEST = 2500
BATCH_SIZE = 48
NUM_WORKERS = 16

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--coef", type=float, required=True, choices=[0.0, 2.0])
parser.add_argument("--folds", type=str, default="0,1,2,3,4",
                    help="comma-separated outer-fold indices (default: all 5)")
args = parser.parse_args()

GPU = args.gpu
COEF = args.coef
FOLDS = [int(f) for f in args.folds.split(",")]

RUN_DIR = (
    "/home/RDC/pfeuffma/Research/proj-orthogonalisation/"
    "experiments/ukbb/runs/2026-05-03_16-43-32_n5k_noreplace_k2/"
)
OUT_CSV = RUN_DIR + f"k{K}_crossfit_results.csv"
PROGRESS = RUN_DIR + f"progress_k{K}_crossfit.log"


# ── Setup ─────────────────────────────────────────────────────────────────────
seed_everything(RANDOM_STATE)
device = torch.device(f"cuda:{GPU}")
print(f"[{datetime.datetime.now():%H:%M:%S}] device=cuda:{GPU}  K={K}  "
      f"coef={COEF}  folds={FOLDS}  NTRAIN={NTRAIN}  NTRAIN_PER_FOLD={NTRAIN_PER_FOLD}  "
      f"replace={RESAMPLE_REPLACE}", flush=True)

print(f"[{datetime.datetime.now():%H:%M:%S}] Loading data ...", flush=True)
d = load_ukbb_data()
# ToDo: Remove "Z_full". Use just "Z" and if necessary only take the first or
# second column, inside the script. Z_full adds mental noise (why is there a
# full?!)
X, y, Z_full = d["X"], d["y"], d["Z_full"]
X_test, y_test, Z_full_test = d["X_test"], d["y_test"], d["Z_full_test"]

tform = default_transforms()
model_params = default_model_params()
trainer_params = default_trainer_params(GPU)
trainer_params["device"] = f"cuda:{GPU}"

# Resample test set to be balanced with respect to 'age'.
idx_test = resample_synthetic(y_test, Z_full_test, NTEST, 0.0, RANDOM_STATE)
X_te, y_te, Z_te = X_test[idx_test], y_test[idx_test], Z_full_test[idx_test]


def _test_ld():
    ds = NumpyCovarDataset(X_te, y_te, Z_te, tform)
    return fast_loader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


def _train_ld(idx):
    ds = NumpyCovarDataset(X[idx], y[idx], Z_full[idx], tform)
    return fast_loader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


def _make_loaders(idx_all, fold_seed_inner, Z=None):
    if Z is None:
        Z = Z_full
    tr_loc, va_loc = train_test_split(
        np.arange(len(idx_all)), test_size=0.2,
        random_state=fold_seed_inner, stratify=y[idx_all],
    )
    idx_tr, idx_va = idx_all[tr_loc], idx_all[va_loc]
    tr_ds = NumpyCovarDataset(X[idx_tr], y[idx_tr], Z[idx_tr], tform)
    va_ds = NumpyCovarDataset(X[idx_va], y[idx_va], Z[idx_va], tform)
    tr_ld = fast_loader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    va_ld = fast_loader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return tr_ld, va_ld, y[idx_tr]


# ── Prediction helpers ────────────────────────────────────────────────────────
@torch.no_grad()
def predict_eta(phm, loader):
    """η = intercept + fx(X) + fz(Z) per obs in `loader`. Shape (N,)."""
    phm.eval()
    out = []
    for batch in loader:
        x = batch["X"].to(device, non_blocking=True)
        z = batch["Z"].to(device, non_blocking=True)
        z_model = z[:, : phm.num_covariates]
        eta = phm.intercept + phm.predict_fx(x) + phm.predict_fz(z_model)
        out.append(eta.cpu().numpy())
    return np.concatenate(out).flatten()


@torch.no_grad()
def predict_fx(phm, loader):
    """fx(X) per obs in `loader`. Shape (N,)."""
    phm.eval()
    out = []
    for batch in loader:
        x = batch["X"].to(device, non_blocking=True)
        out.append(phm.predict_fx(x).cpu().numpy())
    return np.concatenate(out).flatten()


@torch.no_grad()
def predict_fz(phm, loader):
    """fz(Z) per obs in `loader`. Shape (N,)."""
    phm.eval()
    out = []
    for batch in loader:
        z = batch["Z"].to(device, non_blocking=True)
        z_model = z[:, : phm.num_covariates]
        out.append(phm.predict_fz(z_model).cpu().numpy())
    return np.concatenate(out).flatten()


@torch.no_grad()
def _recenter_ensemble(phm, train_idx):
    """Recenter phm on the full train set (∪_k h_k) — unabsorb, then reabsorb."""
    phm.eval()
    train_ds = NumpyCovarDataset(
        X[train_idx], y[train_idx], Z_full[train_idx], tform
    )
    train_ld = fast_loader(
        train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    b0_old = phm.intercept.data.clone()
    cx_old = phm.center_x.mean.clone()
    cz_old = phm.center_z.mean.clone()
    bX = phm.fx.weight.data.flatten().clone()
    bZ = phm.fz.weight.data.flatten().clone()

    feats, Zs = [], []
    for batch in train_ld:
        x = batch["X"].to(device, non_blocking=True)
        z = batch["Z"].to(device, non_blocking=True)
        feats.append(phm.backbone(x))
        Zs.append(z[:, : phm.num_covariates])
    feats = torch.cat(feats, dim=0)
    Zall  = torch.cat(Zs,    dim=0)
    cx_new = feats.mean(dim=0)
    cz_new = Zall.mean(dim=0)

    alpha  = b0_old - cx_old @ bX - cz_old @ bZ
    b0_new = alpha + cx_new @ bX + cz_new @ bZ

    phm.intercept.data.copy_(b0_new)
    phm.center_x.mean.copy_(cx_new)
    phm.center_z.mean.copy_(cz_new)
    return phm


def _eval_metrics(y_hat, fx_hat=None, y_marg=None):
    """auc/bacc + marg-over-train-Z variants + corr(fx, Z_te)."""
    y_bin = (y_hat >= 0.5).astype(int)
    out = {
        "auc": float(roc_auc_score(y_te, y_hat)),
        "bacc": float(balanced_accuracy_score(y_te, y_bin)),
        "auc_marg": float("nan"),
        "bacc_marg": float("nan"),
        "corr_age": float("nan"),
        "corr_sex": float("nan"),
    }
    if y_marg is not None:
        y_marg_bin = (y_marg >= 0.5).astype(int)
        out["auc_marg"] = float(roc_auc_score(y_te, y_marg))
        out["bacc_marg"] = float(balanced_accuracy_score(y_te, y_marg_bin))
    if fx_hat is not None:
        out["corr_age"] = float(np.corrcoef(fx_hat, Z_te[:, 0])[0, 1])
        out["corr_sex"] = float(np.corrcoef(fx_hat, Z_te[:, 1])[0, 1])
    return out


@torch.no_grad()
def _predict_dnn(model, loader):
    """y_hat = sigmoid(intercept + fx(X)) for a centered BaseNetwork (link=logit).
    Returns (y_hat, fx_hat) per obs."""
    model.eval()
    y_hats, fx_hats = [], []
    for batch in loader:
        x = batch["X"].to(device, non_blocking=True)
        y_hats.append(model(x).cpu().numpy())
        fx_hats.append(model.predict_fx(x).cpu().numpy())
    return (
        np.concatenate(y_hats).flatten(),
        np.concatenate(fx_hats).flatten(),
    )


# ── Main loop ─────────────────────────────────────────────────────────────────
results_rows = []

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
all_train_ix = list(cv.split(np.zeros(len(y)), y))

for fold in FOLDS:
    train_ix = all_train_ix[fold][0]
    fold_seed = RANDOM_STATE + fold
    ckpt_dir = RUN_DIR + f"coef={COEF}/fold={fold}/k{K}/"
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n[{datetime.datetime.now():%H:%M:%S}] ══ K={K} coef={COEF} fold={fold} ══", flush=True)

    # ── 1. Resample synthetic-confounded subsample of train_ix.
    #      Safe to split AFTER resampling because RESAMPLE_REPLACE=False yields
    #      unique original obs (see "Split before resampling with replacement"
    #      rule in CLAUDE.md).
    local = resample_synthetic(
        y[train_ix], Z_full[train_ix],
        NTRAIN, COEF, fold_seed,
        replace=RESAMPLE_REPLACE,
    )
    full_train_idx = train_ix[local]

    # ── 2. Inner K-fold partition of resampled set into disjoint h_k.
    inner_cv = StratifiedKFold(n_splits=K, shuffle=True, random_state=fold_seed)
    h_idx = []
    for _, fold_idx in inner_cv.split(np.zeros(NTRAIN), y[full_train_idx]):
        h_idx.append(full_train_idx[fold_idx])
    print(f"[{datetime.datetime.now():%H:%M:%S}]  full_train_idx: {len(full_train_idx)} "
          f"({len(np.unique(full_train_idx))} unique) → h_idx sizes: "
          f"{[len(h) for h in h_idx]}", flush=True)

    # ── 2b. K-independent fits on the full 5000 (dnn baseline + no-samp refit).
    #       Stored under coef/fold/full/ so they're shared across K=2, K=3, ...
    full_dir = RUN_DIR + f"coef={COEF}/fold={fold}/full/"
    os.makedirs(full_dir, exist_ok=True)

    # base_full: DNN backbone on full_train_idx (5000 obs). No control.
    bb_full_path = full_dir + "base_full.pt"
    bb_full_tr_ld, bb_full_va_ld, y_bb_full_tr = _make_loaders(full_train_idx, fold_seed + 500)
    if os.path.exists(bb_full_path):
        print(f"[{datetime.datetime.now():%H:%M:%S}]   base_full exists — loading.", flush=True)
        base_full = BaseNetwork(**model_params).to(device)
        base_full.load_state_dict(torch.load(bb_full_path, map_location=device))
        base_full = base_full.center_effects(bb_full_tr_ld)
        base_full.link = "logit"
        base_full.eval()
    else:
        print(f"[{datetime.datetime.now():%H:%M:%S}]   training base_full on "
              f"{len(full_train_idx)} obs ...", flush=True)
        cw_bb = sklearn.utils.class_weight.compute_class_weight(
            "balanced", classes=np.unique(y_bb_full_tr), y=y_bb_full_tr
        )
        pw_bb = torch.tensor(cw_bb[1] / cw_bb[0])
        tp_bb = {
            **trainer_params,
            "loss_fn": nn.BCEWithLogitsLoss(pos_weight=pw_bb.to(device)),
            "patience": 10,
        }
        base_full = covar_trainer(
            model=BaseNetwork,
            model_params={**model_params, "link": "identity"},
            train_loader=bb_full_tr_ld, val_loader=bb_full_va_ld,
            **tp_bb,
        )
        base_full = base_full.center_effects(bb_full_tr_ld)
        base_full.link = "logit"
        print(f"[{datetime.datetime.now():%H:%M:%S}]   base_full done. "
              f"best_epoch={base_full.best_epoch_}", flush=True)
        torch.save(base_full.state_dict(), bb_full_path)
    del bb_full_tr_ld, bb_full_va_ld; gc.collect()

    # refit_full_age, refit_full_age_sex: fitted on the SAME 5000 obs as
    # base_full. By construction this violates sample-splitting; the bias of
    # this estimator is the comparison the figure is meant to expose.
    ph_full_age_path = full_dir + "refit_full_age.pt"
    ph_full_age_tr_ld, ph_full_age_va_ld, _ = _make_loaders(
        full_train_idx, fold_seed + 700, Z=Z_full[:, 0:1]
    )
    phm_full_age = RefitCovarNetwork(base_full, num_covariates=1, orthogonalize=False).to(device)
    if os.path.exists(ph_full_age_path):
        print(f"[{datetime.datetime.now():%H:%M:%S}]   refit_full_age exists — loading.", flush=True)
        phm_full_age.load_state_dict(torch.load(ph_full_age_path, map_location=device))
        phm_full_age.eval()
    else:
        print(f"[{datetime.datetime.now():%H:%M:%S}]   fitting refit_full_age on "
              f"{len(full_train_idx)} obs ...", flush=True)
        phm_full_age = phm_full_age.fit(ph_full_age_tr_ld, ph_full_age_va_ld, max_iters=400, tol=1e-3)
        print(f"[{datetime.datetime.now():%H:%M:%S}]   refit_full_age done. "
              f"lam={float(phm_full_age.lam.data):.3e}", flush=True)
        torch.save(phm_full_age.state_dict(), ph_full_age_path)
    del ph_full_age_tr_ld, ph_full_age_va_ld; gc.collect()

    ph_full_sex_path = full_dir + "refit_full_age_sex.pt"
    ph_full_sex_tr_ld, ph_full_sex_va_ld, _ = _make_loaders(
        full_train_idx, fold_seed + 800, Z=Z_full
    )
    phm_full_sex = RefitCovarNetwork(base_full, num_covariates=2, orthogonalize=False).to(device)
    if os.path.exists(ph_full_sex_path):
        print(f"[{datetime.datetime.now():%H:%M:%S}]   refit_full_age_sex exists — loading.", flush=True)
        phm_full_sex.load_state_dict(torch.load(ph_full_sex_path, map_location=device))
        phm_full_sex.eval()
    else:
        print(f"[{datetime.datetime.now():%H:%M:%S}]   fitting refit_full_age_sex on "
              f"{len(full_train_idx)} obs ...", flush=True)
        phm_full_sex = phm_full_sex.fit(ph_full_sex_tr_ld, ph_full_sex_va_ld, max_iters=400, tol=1e-3)
        print(f"[{datetime.datetime.now():%H:%M:%S}]   refit_full_age_sex done. "
              f"lam={float(phm_full_sex.lam.data):.3e}", flush=True)
        torch.save(phm_full_sex.state_dict(), ph_full_sex_path)
    del ph_full_sex_tr_ld, ph_full_sex_va_ld; gc.collect()

    # ── 3. Per rotation: train backbone on ∪_{j≠k} h_j, fit refit on h_k. ──
    phms_age = []  # Refit models controlling for age.
    phms_sex = []  # Refit models controlling for age + sex. (Sanity check)
    for k in range(K):
        dnn_idx = np.concatenate([h_idx[j] for j in range(K) if j != k])
        post_idx = h_idx[k]

        # ── 3a. Backbone on dnn_idx (skip if checkpoint exists) ──────────────
        bb_path = ckpt_dir + f"backbone_k{k}.pt"
        bb_ld_tr, bb_ld_va, y_bb_tr = _make_loaders(dnn_idx, fold_seed + 1000 + k)
        if os.path.exists(bb_path):
            print(f"[{datetime.datetime.now():%H:%M:%S}]   k={k}: backbone exists — loading.", flush=True)
            backbone_k = BaseNetwork(**model_params).to(device)
            backbone_k.load_state_dict(torch.load(bb_path, map_location=device))
            backbone_k = backbone_k.center_effects(bb_ld_tr)
            backbone_k.link = "logit"
            backbone_k.eval()
        else:
            print(f"[{datetime.datetime.now():%H:%M:%S}]   k={k}: training backbone on "
                  f"{len(dnn_idx)} obs ...", flush=True)
            cw_bb = sklearn.utils.class_weight.compute_class_weight(
                "balanced", classes=np.unique(y_bb_tr), y=y_bb_tr
            )
            pw_bb = torch.tensor(cw_bb[1] / cw_bb[0])
            tp_bb = {
                **trainer_params,
                "loss_fn": nn.BCEWithLogitsLoss(pos_weight=pw_bb.to(device)),
                "patience": 10,  # smoke best_epoch=7-9; default=20 over-runs
            }
            backbone_k = covar_trainer(
                model=BaseNetwork,
                model_params={**model_params, "link": "identity"},
                train_loader=bb_ld_tr, val_loader=bb_ld_va,
                **tp_bb,
                ) # Todo: Fix link function hack. Works for now, but not clean.
            backbone_k = backbone_k.center_effects(bb_ld_tr)
            backbone_k.link = "logit"
            print(f"[{datetime.datetime.now():%H:%M:%S}]   k={k}: backbone done. "
                  f"best_epoch={backbone_k.best_epoch_}", flush=True)
            torch.save(backbone_k.state_dict(), bb_path)
        del bb_ld_tr, bb_ld_va; gc.collect()

        # ── 3b. Refit (age-only) on post_idx ─────────────────────────────────
        ph_age_path = ckpt_dir + f"refit_age_k{k}.pt"
        ph_age_tr_ld, ph_age_va_ld, _ = _make_loaders(
            post_idx, fold_seed + 2000 + k, Z=Z_full[:, 0:1]
        )
        phm_k_age = RefitCovarNetwork(backbone_k, num_covariates=1, orthogonalize=False).to(device)
        if os.path.exists(ph_age_path):
            print(f"[{datetime.datetime.now():%H:%M:%S}]   k={k}: refit_age exists — loading.", flush=True)
            phm_k_age.load_state_dict(torch.load(ph_age_path, map_location=device))
            phm_k_age.eval()
        else:
            print(f"[{datetime.datetime.now():%H:%M:%S}]   k={k}: fitting refit_age on "
                  f"{len(post_idx)} obs ...", flush=True)
            phm_k_age = phm_k_age.fit(ph_age_tr_ld, ph_age_va_ld, max_iters=400, tol=1e-3)
            print(f"[{datetime.datetime.now():%H:%M:%S}]   k={k}: refit_age done. "
                  f"lam={float(phm_k_age.lam.data):.3e}", flush=True)
            torch.save(phm_k_age.state_dict(), ph_age_path)
        phms_age.append(phm_k_age)
        del ph_age_tr_ld, ph_age_va_ld; gc.collect()

        # ── 3c. Refit (age+sex) on post_idx ──────────────────────────────────
        ph_sex_path = ckpt_dir + f"refit_age_sex_k{k}.pt"
        ph_sex_tr_ld, ph_sex_va_ld, _ = _make_loaders(
            post_idx, fold_seed + 3000 + k, Z=Z_full
        )
        phm_k_sex = RefitCovarNetwork(backbone_k, num_covariates=2, orthogonalize=False).to(device)
        if os.path.exists(ph_sex_path):
            print(f"[{datetime.datetime.now():%H:%M:%S}]   k={k}: refit_age_sex exists — loading.", flush=True)
            phm_k_sex.load_state_dict(torch.load(ph_sex_path, map_location=device))
            phm_k_sex.eval()
        else:
            print(f"[{datetime.datetime.now():%H:%M:%S}]   k={k}: fitting refit_age_sex on "
                  f"{len(post_idx)} obs ...", flush=True)
            phm_k_sex = phm_k_sex.fit(ph_sex_tr_ld, ph_sex_va_ld, max_iters=400, tol=1e-3)
            print(f"[{datetime.datetime.now():%H:%M:%S}]   k={k}: refit_age_sex done. "
                  f"lam={float(phm_k_sex.lam.data):.3e}", flush=True)
            torch.save(phm_k_sex.state_dict(), ph_sex_path)
        phms_sex.append(phm_k_sex)
        del ph_sex_tr_ld, ph_sex_va_ld; gc.collect()

    # ── 4. Predictions on shared balanced test set ──────────────────────────
    test_ld  = _test_ld()
    train_ld = _train_ld(full_train_idx)

    def _coefs(phm):
        """Return (b_age, b_sex) — second is NaN for num_covariates=1 models."""
        w = phm.fz.weight.data.flatten().cpu().numpy()
        return float(w[0]), float(w[1]) if w.size > 1 else float("nan")

    def _eval_ensemble(phms, name_ens):
        """Evaluate a K-fold ensemble (len==1 degenerates to a single-phm eval via the same pipeline)."""
        # ŷ_marg(X*) = (1/N_train) Σ_j σ((1/K) Σ_k η_k(X*, z_j)); used for refit_nosamp_* and refit_split_*.
        # ── Recenter each phm on the full train set.
        for p in phms:
            _recenter_ensemble(p, full_train_idx)

        # ── Per-phm predictions.
        etas_test = [predict_eta(p, test_ld)  for p in phms]
        fxs_test  = [predict_fx (p, test_ld)  for p in phms]
        fzs_train = [predict_fz (p, train_ld) for p in phms]
        intercepts = [float(p.intercept.detach().cpu()) for p in phms]
        coefs = [_coefs(p) for p in phms]

        # ── Ensembles (mean across phms).
        eta_ens       = np.mean(etas_test, axis=0)
        fx_ens        = np.mean(fxs_test,  axis=0)
        fz_ens_train  = np.mean(fzs_train, axis=0)
        intercept_ens = float(np.mean(intercepts))
        eta_x_ens     = intercept_ens + fx_ens                     # eta without fz
        ba_ens  = float(np.mean([c[0] for c in coefs]))
        bs_vals = [c[1] for c in coefs]
        bs_ens  = float(np.mean(bs_vals)) if not any(np.isnan(b) for b in bs_vals) else float("nan")

        # ── Y-scale full-ensemble marg over training Z.
        eta_grid = (
            torch.tensor(eta_x_ens,    device=device)[:, None]
            + torch.tensor(fz_ens_train, device=device)[None, :]
        )                                                          # (n_test, N_train)
        y_ens_marg = torch.sigmoid(eta_grid).mean(dim=1).cpu().numpy()
        y_ens = 1.0 / (1.0 + np.exp(-eta_ens))

        m = _eval_metrics(y_ens, fx_hat=fx_ens, y_marg=y_ens_marg)
        m.update({"method": name_ens, "coef": COEF, "fold": fold,
                  "b_age": ba_ens, "b_sex": bs_ens, "lam": float("nan")})
        print(f"[{datetime.datetime.now():%H:%M:%S}]    {name_ens:>20}: "
              f"auc={m['auc']:.4f}  auc_marg={m['auc_marg']:.4f}  "
              f"bacc={m['bacc']:.4f}  corr(age)={m['corr_age']:+.3f}  "
              f"corr(sex)={m['corr_sex']:+.3f}  "
              f"b_age={ba_ens:+.4f}  b_sex={bs_ens:+.4f}", flush=True)
        with open(PROGRESS, "a", buffering=1) as pf:
            pf.write(json.dumps(m) + "\n")
        return m

    # ── 5. DNN baseline (no control) ────────────────────────────────────────
    y_hat_dnn, fx_hat_dnn = _predict_dnn(base_full, test_ld)
    m_dnn = _eval_metrics(y_hat_dnn, fx_hat=fx_hat_dnn, y_marg=None)
    m_dnn.update({"method": "dnn", "coef": COEF, "fold": fold,
                  "b_age": float("nan"), "b_sex": float("nan"), "lam": float("nan")})
    print(f"[{datetime.datetime.now():%H:%M:%S}]    {'dnn':>20}: "
          f"auc={m_dnn['auc']:.4f}  auc_marg=   nan  "
          f"bacc={m_dnn['bacc']:.4f}  corr(age)={m_dnn['corr_age']:+.3f}  "
          f"corr(sex)={m_dnn['corr_sex']:+.3f}  "
          f"b_age=    nan  b_sex=    nan", flush=True)
    with open(PROGRESS, "a", buffering=1) as pf:
        pf.write(json.dumps(m_dnn) + "\n")
    results_rows.append(m_dnn)

    # ── 6. No-samp refits (single-phm "ensemble" — backbone & head share data).
    results_rows.append(_eval_ensemble([phm_full_age], "refit_nosamp_age"))
    results_rows.append(_eval_ensemble([phm_full_sex], "refit_nosamp_age_sex"))

    # ── 7. Sample-split: k=0 sub-model of K=2 alone (backbone on h_1, head on h_0).
    results_rows.append(_eval_ensemble([phms_age[0]], "refit_split_age"))
    results_rows.append(_eval_ensemble([phms_sex[0]], "refit_split_age_sex"))

    # ── 8. Cross-fit ensembles (K-rotation mean).
    results_rows.append(_eval_ensemble(phms_age, f"crossfit_k{K}_age"))
    results_rows.append(_eval_ensemble(phms_sex, f"crossfit_k{K}_age_sex"))

    del phms_age, phms_sex, phm_full_age, phm_full_sex, base_full
    del test_ld, train_ld
    torch.cuda.empty_cache()
    gc.collect()


# ── Append to k3_crossfit_results.csv (preserve existing rows from prior runs) ──
df_new = pd.DataFrame(results_rows)
if os.path.exists(OUT_CSV):
    df_old = pd.read_csv(OUT_CSV)
    keep = ~(df_old.coef.eq(COEF) & df_old.fold.isin(FOLDS))
    df = pd.concat([df_old[keep], df_new], ignore_index=True)
else:
    df = df_new
df.to_csv(OUT_CSV, index=False)

print(f"\n[{datetime.datetime.now():%H:%M:%S}] Done. "
      f"Wrote {len(results_rows)} rows ({len(FOLDS)} fold(s) × 7 methods) "
      f"to {OUT_CSV}", flush=True)
