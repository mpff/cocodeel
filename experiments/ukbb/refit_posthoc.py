#!/usr/bin/env python
"""Refit only the PostHoc step using the existing base_half checkpoints.

Use case: cocodeel had a Z_val/Z_std bias bug (fixed in cocodeel commit 7d16b0a).
The base_full / base_half checkpoints are independent of that bug, so we keep
them as-is and rebuild only:

    posthoc_age      → PostHocCovarNetwork(base_half, num_covariates=1)
    posthoc_age_sex  → PostHocCovarNetwork(base_half, num_covariates=2)

For each (coef, fold):
  1. Load existing `base_half.pt`.
  2. Rebuild h2 loaders (same fold seed → identical train/val split).
  3. Refit phm_age and phm_sex.
  4. Overwrite checkpoints + replace posthoc rows in all rexports CSVs.
  5. Recompute test predictions, controlled predictions, summary metrics.

Hardcoded to RUN_DIR. Other methods (base_full / base_half) are left untouched.

Run:
    cd experiments/ukbb/
    conda run --no-capture-output -n dl-mri python refit_posthoc.py
"""
import os
import gc
import json
import argparse
import datetime

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
from cocodeel.posthoc_model import PostHocCovarNetwork

# ── Config (mirror run_ukbb_experiment.py) ───────────────────────────────────
COEFS = [0.0, 2.0]
N_SPLITS = 5
NTRAIN_PER_HALF = 2500
NTEST = 2500
DEFAULT_GPU = 1
BATCH_SIZE = 48
NUM_WORKERS = 16

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=DEFAULT_GPU)
args = parser.parse_args()
GPU = args.gpu

RUN_DIR = (
    "/home/RDC/pfeuffma/Research/proj-orthogonalisation/"
    "experiments/ukbb/runs/2026-04-14_17-26-52_final/"
)
REXPORTS = RUN_DIR + "rexports/"
PROGRESS = RUN_DIR + "progress_refit.log"

POSTHOC_METHODS = {"posthoc_age", "posthoc_age_sex"}


# ── Setup ─────────────────────────────────────────────────────────────────────
seed_everything(RANDOM_STATE)
device = torch.device(f"cuda:{GPU}")
print(f"[{datetime.datetime.now():%H:%M:%S}] device=cuda:{GPU}", flush=True)

print(f"[{datetime.datetime.now():%H:%M:%S}] Loading data ...", flush=True)
d = load_ukbb_data()
X, y, Z_full = d["X"], d["y"], d["Z_full"]
X_test, y_test, Z_full_test = d["X_test"], d["y_test"], d["Z_full_test"]

tform = default_transforms()
model_params = default_model_params()
trainer_params = default_trainer_params(GPU)

# Balanced test set — same call signature as run_ukbb_experiment.py.
idx_test = resample_synthetic(y_test, Z_full_test, NTEST, 0.0, RANDOM_STATE)
X_te = X_test[idx_test]
y_te = y_test[idx_test]
Z_te = Z_full_test[idx_test]


def _test_ld():
    ds = NumpyCovarDataset(X_te, y_te, Z_te, tform)
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


def _collect_preds(model, loader):
    model.eval()
    y_hats, fx_hats = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["X"].to(device, non_blocking=True)
            z = batch["Z"].to(device, non_blocking=True)
            z_model = z[:, :model.num_covariates]
            y_hats.append(model(x, z_model).cpu().numpy())
            fx_hats.append(model.predict_fx(x, z=None).cpu().numpy())
    return np.concatenate(y_hats).flatten(), np.concatenate(fx_hats).flatten()


def _collect_controlled_preds(model, loader, Z_train_raw):
    model.eval()
    y_ctrl = []
    with torch.no_grad():
        Z_tr_t = torch.tensor(Z_train_raw, dtype=torch.float32).to(device)
        fz_tr = model.predict_fz(Z_tr_t)
        for batch in loader:
            x = batch["X"].to(device, non_blocking=True)
            fx = model.predict_fx(x)
            fx_exp = fx.unsqueeze(1).expand(-1, Z_tr_t.shape[0], -1)
            eta = model.intercept + fx_exp + fz_tr
            y_ctrl.append(model.output_func(eta).mean(dim=1).cpu().numpy())
    return np.concatenate(y_ctrl).flatten()


def _eval_summary(name, y_hat, fx_hat, model):
    y_bin = (y_hat >= 0.5).astype(int)
    bacc = balanced_accuracy_score(y_te, y_bin)
    auc = roc_auc_score(y_te, y_hat)
    corr_age = float(np.corrcoef(fx_hat, Z_te[:, 0])[0, 1])
    corr_sex = float(np.corrcoef(fx_hat, Z_te[:, 1])[0, 1])
    w = model.fz.weight.data.flatten().cpu().numpy()
    b_age = float(w[0])
    b_sex = float(w[1]) if w.size > 1 else float("nan")
    lam = float(model.lam.data)
    return {"method": name, "bacc": bacc, "auc": auc,
            "corr_age": corr_age, "corr_sex": corr_sex,
            "b_age": b_age, "b_sex": b_sex, "lam": lam}


# ── Refit loop — accumulate new posthoc rows ─────────────────────────────────
preds_new = []        # testset_predictions.csv (posthoc rows only)
ctrl_new = []         # testset_predictions_controlled.csv (posthoc rows only)
coef_new = []         # fitted_coefs.csv (posthoc rows only)
lambda_new = []       # posthoc_lambda_paths.csv
results_new = []      # raw_results.csv (posthoc rows only)


for coef in COEFS:
    print(f"\n[{datetime.datetime.now():%H:%M:%S}] ══ coef={coef} ══", flush=True)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for fold, (train_ix, _unused) in enumerate(cv.split(np.zeros(len(y)), y)):
        fold_seed = RANDOM_STATE + fold
        ckpt_dir = RUN_DIR + f"coef={coef}/fold={fold}/"
        backbone_ckpt = ckpt_dir + "base_half.pt"
        assert os.path.exists(backbone_ckpt), f"missing {backbone_ckpt}"

        print(f"\n[{datetime.datetime.now():%H:%M:%S}] === coef={coef} fold={fold} ===",
              flush=True)

        # Reproduce the same pool / half_2 split as run_ukbb_experiment.py.
        pool_A, pool_B = train_test_split(
            train_ix, test_size=0.5, random_state=fold_seed, stratify=y[train_ix]
        )
        h1_local = resample_synthetic(
            y[pool_A], Z_full[pool_A], NTRAIN_PER_HALF, coef, fold_seed
        )
        h1_idx = pool_A[h1_local]

        h2_local = resample_synthetic(
            y[pool_B], Z_full[pool_B], NTRAIN_PER_HALF, coef, fold_seed + 100
        )
        h2_idx = pool_B[h2_local]

        full_idx = np.concatenate([h1_idx, h2_idx])

        # Identical inner split seed → identical train/val partition as the
        # original posthoc fits (only the cocodeel internals differ now).
        h2_age_tr_ld, h2_age_va_ld, _ = _make_loaders(h2_idx, fold_seed + 202, Z=Z_full[:, 0:1])
        h2_sex_tr_ld, h2_sex_va_ld, _ = _make_loaders(h2_idx, fold_seed + 202, Z=Z_full)

        # Load base_half backbone (frozen during posthoc fit by the model).
        base_half = BaseNetwork(**model_params).to(device)
        base_half.load_state_dict(torch.load(backbone_ckpt, map_location=device))
        base_half.eval()

        # Refit posthoc_age.
        print(f"[{datetime.datetime.now():%H:%M:%S}]  Fitting posthoc_age ...", flush=True)
        phm_age = PostHocCovarNetwork(base_half, num_covariates=1, orthogonalize=False).to(device)
        phm_age = phm_age.fit(h2_age_tr_ld, h2_age_va_ld, max_iters=400, tol=1e-3)
        print(f"[{datetime.datetime.now():%H:%M:%S}]  posthoc_age done. "
              f"lam={float(phm_age.lam.data):.3e}", flush=True)

        # Refit posthoc_age_sex.
        print(f"[{datetime.datetime.now():%H:%M:%S}]  Fitting posthoc_age_sex ...", flush=True)
        phm_sex = PostHocCovarNetwork(base_half, num_covariates=2, orthogonalize=False).to(device)
        phm_sex = phm_sex.fit(h2_sex_tr_ld, h2_sex_va_ld, max_iters=400, tol=1e-3)
        print(f"[{datetime.datetime.now():%H:%M:%S}]  posthoc_age_sex done. "
              f"lam={float(phm_sex.lam.data):.3e}", flush=True)

        # Predictions on the shared balanced test set.
        test_ld = _test_ld()
        y_hat_phage, fx_hat_phage = _collect_preds(phm_age, test_ld)
        y_hat_phsex, fx_hat_phsex = _collect_preds(phm_sex, test_ld)

        # Marginalize over the FULL training set (h1 + h2), not just h2.
        # Both halves are resampled with the same coef, so the confounding
        # level is consistent.  Using the full set is cleaner (larger sample
        # from the confounded training distribution).
        y_ctrl_phage = _collect_controlled_preds(phm_age, test_ld, Z_full[full_idx, 0:1])
        y_ctrl_phsex = _collect_controlled_preds(phm_sex, test_ld, Z_full[full_idx])

        # Summary metrics.
        for name, y_hat, fx_hat, mdl in [
            ("posthoc_age",     y_hat_phage, fx_hat_phage, phm_age),
            ("posthoc_age_sex", y_hat_phsex, fx_hat_phsex, phm_sex),
        ]:
            row = _eval_summary(name, y_hat, fx_hat, mdl)
            row.update({"coef": coef, "fold": fold})
            results_new.append(row)
            print(f"[{datetime.datetime.now():%H:%M:%S}]    {name}: "
                  f"bacc={row['bacc']:.4f}  auc={row['auc']:.4f}  "
                  f"corr(age)={row['corr_age']:+.3f}  corr(sex)={row['corr_sex']:+.3f}  "
                  f"b_age={row['b_age']:+.4f}  b_sex={row['b_sex']:+.4f}", flush=True)
            with open(PROGRESS, "a", buffering=1) as pf:
                pf.write(json.dumps(row) + "\n")

        # Per-obs prediction rows.
        for obs in range(len(y_hat_phage)):
            preds_new.append({"obs_id": obs, "method": "posthoc_age",
                              "coef": coef, "fold": fold,
                              "y": float(y_hat_phage[obs]), "fx": float(fx_hat_phage[obs])})
            preds_new.append({"obs_id": obs, "method": "posthoc_age_sex",
                              "coef": coef, "fold": fold,
                              "y": float(y_hat_phsex[obs]), "fx": float(fx_hat_phsex[obs])})
            ctrl_new.append({"obs_id": obs, "method": "posthoc_age",
                             "coef": coef, "fold": fold,
                             "y_controlled": float(y_ctrl_phage[obs])})
            ctrl_new.append({"obs_id": obs, "method": "posthoc_age_sex",
                             "coef": coef, "fold": fold,
                             "y_controlled": float(y_ctrl_phsex[obs])})

        # Fitted coefficients.
        w_age = phm_age.fz.weight.data.flatten().cpu().numpy()
        coef_new.append({"method": "posthoc_age", "coef": coef, "fold": fold,
                         "intercept": float(phm_age.intercept.data.cpu().numpy().squeeze()),
                         "age": float(w_age[0]), "sex": float("nan"),
                         "lam": float(phm_age.lam.data)})
        w_sex = phm_sex.fz.weight.data.flatten().cpu().numpy()
        coef_new.append({"method": "posthoc_age_sex", "coef": coef, "fold": fold,
                         "intercept": float(phm_sex.intercept.data.cpu().numpy().squeeze()),
                         "age": float(w_sex[0]), "sex": float(w_sex[1]),
                         "lam": float(phm_sex.lam.data)})

        # Lambda paths.
        for rec in phm_age.lambda_path_:
            lambda_new.append({"method": "posthoc_age", "coef": coef, "fold": fold,
                               "selected_lambda": float(phm_age.lam.data), **rec})
        for rec in phm_sex.lambda_path_:
            lambda_new.append({"method": "posthoc_age_sex", "coef": coef, "fold": fold,
                               "selected_lambda": float(phm_sex.lam.data), **rec})

        # Overwrite posthoc checkpoints (base_full/base_half left untouched).
        torch.save(phm_age.state_dict(), ckpt_dir + "posthoc_age.pt")
        torch.save(phm_sex.state_dict(), ckpt_dir + "posthoc_age_sex.pt")

        del base_half, phm_age, phm_sex, test_ld
        torch.cuda.empty_cache()
        gc.collect()


# ── Splice posthoc rows into existing CSVs ───────────────────────────────────
print(f"\n[{datetime.datetime.now():%H:%M:%S}] Splicing CSVs ...", flush=True)


def _replace_posthoc_rows(path, new_rows, key="method"):
    df_old = pd.read_csv(path)
    df_kept = df_old[~df_old[key].isin(POSTHOC_METHODS)]
    df_new = pd.DataFrame(new_rows, columns=df_old.columns)
    df = pd.concat([df_kept, df_new], ignore_index=True)
    df.to_csv(path, index=False)
    print(f"  → {path}  (kept {len(df_kept)} non-posthoc rows, "
          f"added {len(df_new)} posthoc rows)", flush=True)


_replace_posthoc_rows(RUN_DIR + "raw_results.csv",                     results_new)
_replace_posthoc_rows(REXPORTS + "testset_predictions.csv",            preds_new)
_replace_posthoc_rows(REXPORTS + "testset_predictions_controlled.csv", ctrl_new)
_replace_posthoc_rows(REXPORTS + "fitted_coefs.csv",                   coef_new)
_replace_posthoc_rows(REXPORTS + "posthoc_lambda_paths.csv",           lambda_new)

print(f"\n[{datetime.datetime.now():%H:%M:%S}] Done. "
      f"Refit {len(results_new)} posthoc fits.", flush=True)
