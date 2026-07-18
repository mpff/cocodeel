#!/usr/bin/env python
"""UKBB HighAlc sample-split experiment: base_full, base_half, refit_age, refit_age_sex over 5 folds × {coef=0, 2.0}."""
import os
import gc
import json
import datetime
import sklearn.utils.class_weight

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

# shared module — must be on sys.path already via ukbb_common's path setup
from ukbb_common import (
    proj_path, RANDOM_STATE,
    seed_everything, load_ukbb_data,
    resample_synthetic, NumpyCovarDataset,
    fast_loader,
    default_model_params, default_trainer_params, default_transforms,
    setup_run_dir, write_manifest,
)
from cocodeel.model import BaseNetwork
from cocodeel.refit_model import RefitCovarNetwork
from cocodeel.trainer import covar_trainer

# ── Config ────────────────────────────────────────────────────────────────────
COEFS = [0.0, 2.0]
N_SPLITS = 5
PILOT_FOLDS = 5           # full CV; set to 1 for a fold-0 pilot
NTRAIN_PER_HALF = 2500
NTEST = 2500
GPU = 0
BATCH_SIZE = 48
NUM_WORKERS = 16

# True DGP coefficients (for manifest)
TRUE_B_AGE_RAW = -0.298   # age coef in logit space (raw units)
TRUE_B_SEX_RAW = 2.0      # sex coef in logit space

# ── Setup ─────────────────────────────────────────────────────────────────────
seed_everything(RANDOM_STATE)
device = torch.device(f"cuda:{GPU}")

print(
    f"[{datetime.datetime.now():%H:%M:%S}] device=cuda:{GPU}  "
    f"mem_free={torch.cuda.mem_get_info(GPU)[0]/1024**3:.1f} GB",
    flush=True,
)

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"[{datetime.datetime.now():%H:%M:%S}] Loading data ...", flush=True)
d = load_ukbb_data()
X, y, Z_full = d["X"], d["y"], d["Z_full"]
X_test, y_test, Z_full_test = d["X_test"], d["y_test"], d["Z_full_test"]
print(
    f"[{datetime.datetime.now():%H:%M:%S}] Loaded. "
    f"X={X.shape}  X_test={X_test.shape}",
    flush=True,
)

# ── Shared config objects ─────────────────────────────────────────────────────
tform = default_transforms()
model_params = default_model_params()
trainer_params = default_trainer_params(GPU)
# Belt-and-braces for the `device or "cpu"` trap: force explicit string device
# so training always lands on GPU regardless of cocodeel version in scope.
trainer_params["device"] = f"cuda:{GPU}"

# ── Run infrastructure ────────────────────────────────────────────────────────
results_dir = setup_run_dir("final_v2")
write_manifest(results_dir, {
    "COEFS": COEFS,
    "N_SPLITS": N_SPLITS,
    "PILOT_FOLDS": PILOT_FOLDS,
    "NTRAIN_PER_HALF": NTRAIN_PER_HALF,
    "NTEST": NTEST,
    "GPU": GPU,
    "BATCH_SIZE": BATCH_SIZE,
    "methods": ["base_full", "base_half", "refit_age", "refit_age_sex"],
    "true_b_age_raw": TRUE_B_AGE_RAW,
    "true_b_sex_raw": TRUE_B_SEX_RAW,
})
progress_path = results_dir + "progress.log"
print(
    f"[{datetime.datetime.now():%H:%M:%S}] Run dir: {results_dir}  PID={os.getpid()}",
    flush=True,
)


# ── Balanced test set (coef=0 → unconfounded; same across all coef/fold) ─────
idx_test = resample_synthetic(y_test, Z_full_test, NTEST, 0.0, RANDOM_STATE)
X_te = X_test[idx_test]
y_te = y_test[idx_test]
Z_te = Z_full_test[idx_test]


def _test_ld():
    """Build test DataLoader with raw Z (RefitCovarNetwork standardizes internally)."""
    ds = NumpyCovarDataset(X_te, y_te, Z_te, tform)
    return fast_loader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


# ── Prediction helpers ─────────────────────────────────────────────────────────
def _collect_preds(model, loader):
    """Return (y_hat, fx_hat) arrays for every test observation.

    shuffle=False on loader is mandatory — order must match testset.csv.
    """
    model.eval()
    y_hats, fx_hats = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["X"].to(device, non_blocking=True)
            z = batch["Z"].to(device, non_blocking=True)
            if getattr(model, "num_covariates", 0) > 0:
                # Slice z to the model's expected number of covariates.
                # The test loader always carries full Z (age+sex); refit_age only needs age.
                z_model = z[:, :model.num_covariates]
                y_hats.append(model(x, z_model).cpu().numpy())
            else:
                y_hats.append(model(x).cpu().numpy())
            if hasattr(model, "predict_fx"):
                # BaseNetwork.predict_fx(x) — no z argument.
                # RefitCovarNetwork.predict_fx(x, z=None) — z unused when orthogonalize=False.
                if getattr(model, "num_covariates", 0) > 0:
                    fx_hats.append(model.predict_fx(x, z=None).cpu().numpy())
                else:
                    fx_hats.append(model.predict_fx(x).cpu().numpy())
    y_hat = np.concatenate(y_hats).flatten()
    fx_hat = np.concatenate(fx_hats).flatten() if fx_hats else np.full_like(y_hat, np.nan)
    return y_hat, fx_hat


def _collect_controlled_preds(model, loader, Z_train_raw):
    """Marginalize predictions over the refit training-Z distribution (half_2)."""
    # y_ctrl_i = mean_j[ sigmoid(intercept + fx_i + fz(z_j)) ], j over Z_train_raw.
    model.eval()
    y_ctrl = []
    with torch.no_grad():
        Z_tr_t = torch.tensor(Z_train_raw, dtype=torch.float32).to(device)
        fz_tr = model.predict_fz(Z_tr_t)   # (n_train, 1)
        for batch in loader:
            x = batch["X"].to(device, non_blocking=True)
            fx = model.predict_fx(x)                          # (batch, 1)
            fx_exp = fx.unsqueeze(1).expand(-1, Z_tr_t.shape[0], -1)
            eta = model.intercept + fx_exp + fz_tr            # (batch, n_train, 1)
            y_ctrl.append(model.output_func(eta).mean(dim=1).cpu().numpy())
    return np.concatenate(y_ctrl).flatten()


def _eval_summary(name, y_hat, fx_hat, model=None, y_ctrl=None):
    """Per-fold summary metrics; adds marginalized AUC/bacc when y_ctrl is given (refit models)."""
    y_bin = (y_hat >= 0.5).astype(int)
    bacc = balanced_accuracy_score(y_te, y_bin)
    auc = roc_auc_score(y_te, y_hat)
    corr_age = float(np.corrcoef(fx_hat, Z_te[:, 0])[0, 1])
    corr_sex = float(np.corrcoef(fx_hat, Z_te[:, 1])[0, 1])
    bacc_marg = float("nan")
    auc_marg = float("nan")
    if y_ctrl is not None:
        y_ctrl_bin = (y_ctrl >= 0.5).astype(int)
        bacc_marg = balanced_accuracy_score(y_te, y_ctrl_bin)
        auc_marg = roc_auc_score(y_te, y_ctrl)
    b_age = b_sex = float("nan")
    lam = float("nan")
    if model is not None and hasattr(model, "fz"):
        w = model.fz.weight.data.flatten().cpu().numpy()
        # fz.weight is de-standardized by RefitCovarNetwork._fit_effects, so w is
        # already in centered-Z units (= raw-Z units, since centering doesn't change slope).
        # Comparable directly to TRUE_B_AGE_RAW / TRUE_B_SEX_RAW.
        b_age = float(w[0])
        if len(w) > 1:
            b_sex = float(w[1])
        lam = float(model.lam.data)
    return {
        "method": name, "bacc": bacc, "auc": auc,
        "bacc_marg": bacc_marg, "auc_marg": auc_marg,
        "corr_age": corr_age, "corr_sex": corr_sex,
        "b_age": b_age, "b_sex": b_sex, "lam": lam,
    }


def _make_loaders(idx_all, fold_seed_inner, Z=None):
    """Stratified train/val DataLoaders over global indices; raw Z (default age+sex) is standardized inside the refit."""
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


# ── Prediction accumulators (for R exports) ───────────────────────────────────
# Rows: one per (method, coef, fold, observation)
preds_rows = []         # testset_predictions.csv
ctrl_rows = []          # testset_predictions_controlled.csv
coef_rows = []          # fitted_coefs.csv
lambda_rows = []        # refit_lambda_paths.csv
train_rows = []         # trainset_folds.csv
results_rows = []       # raw_results.csv


def _append_preds(method, coef, fold, y_hat, fx_hat):
    # obs_id is the row index into testset.csv (same test set for all coef/fold).
    # Explicit obs_id allows safe joins in R without relying on implicit row order.
    for obs in range(len(y_hat)):
        preds_rows.append({
            "obs_id": obs,
            "method": method, "coef": coef, "fold": fold,
            "y": float(y_hat[obs]), "fx": float(fx_hat[obs]),
        })


def _append_ctrl(method, coef, fold, y_ctrl):
    for obs in range(len(y_ctrl)):
        ctrl_rows.append({
            "obs_id": obs,
            "method": method, "coef": coef, "fold": fold,
            "y_controlled": float(y_ctrl[obs]),
        })


# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════
for coef in COEFS:
    print(f"\n[{datetime.datetime.now():%H:%M:%S}] ══ coef={coef} ══", flush=True)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for fold, (train_ix, _unused) in enumerate(cv.split(np.zeros(len(y)), y)):
        if fold >= PILOT_FOLDS:
            break
        fold_seed = RANDOM_STATE + fold
        print(
            f"\n[{datetime.datetime.now():%H:%M:%S}] === coef={coef}  fold={fold} ===",
            flush=True,
        )

        # ── 1. Split pool BEFORE resampling ──────────────────────────────
        pool_A, pool_B = train_test_split(
            train_ix, test_size=0.5, random_state=fold_seed, stratify=y[train_ix]
        )
        assert len(set(pool_A) & set(pool_B)) == 0, "pool_A and pool_B must be disjoint"

        # ── 2. Resample each half independently ──────────────────────────
        h1_local = resample_synthetic(y[pool_A], Z_full[pool_A], NTRAIN_PER_HALF, coef, fold_seed)
        h1_idx = pool_A[h1_local]

        h2_local = resample_synthetic(y[pool_B], Z_full[pool_B], NTRAIN_PER_HALF, coef, fold_seed + 100)
        h2_idx = pool_B[h2_local]

        # Disjointness holds by construction (pool_A ∩ pool_B = ∅ and resampling
        # stays within each pool), but assert explicitly on global indices.
        assert len(set(h1_idx) & set(h2_idx)) == 0, "h1_idx and h2_idx must be disjoint"

        full_idx = np.concatenate([h1_idx, h2_idx])

        # ── 3. Build DataLoaders ──────────────────────────────────────────
        # Distinct seed offsets so val splits are independent across groups.
        # +200 / +201 / +202 chosen to avoid overlap with resampling seeds (fold_seed, fold_seed+100).
        # h2 loaders are built twice (same seed → same train/val split) with different Z columns.
        full_tr_ld,     full_va_ld,     y_full_tr = _make_loaders(full_idx, fold_seed + 200)
        h1_tr_ld,       h1_va_ld,       y_h1_tr  = _make_loaders(h1_idx,   fold_seed + 201)
        h2_age_tr_ld,   h2_age_va_ld,   _        = _make_loaders(h2_idx,   fold_seed + 202, Z=Z_full[:, 0:1])
        h2_sex_tr_ld,   h2_sex_va_ld,   _        = _make_loaders(h2_idx,   fold_seed + 202, Z=Z_full)

        # ── 4. Train base_full ────────────────────────────────────────────
        print(f"[{datetime.datetime.now():%H:%M:%S}]  Training base_full ...", flush=True)
        cw_full = sklearn.utils.class_weight.compute_class_weight(
            "balanced", classes=np.unique(y_full_tr), y=y_full_tr
        )
        pw_full = torch.tensor(cw_full[1] / cw_full[0])
        tp_full = {
            **trainer_params,
            "loss_fn": nn.BCEWithLogitsLoss(pos_weight=pw_full.to(device)),
        }
        # Train with link="identity" so covar_trainer applies BCEWithLogitsLoss
        # directly to raw logit outputs (numerically stable).  After training,
        # we switch link to "logit" so model() outputs probabilities via sigmoid.
        base_full = covar_trainer(
            model=BaseNetwork,
            model_params={**model_params, "link": "identity"},
            train_loader=full_tr_ld, val_loader=full_va_ld,
            **tp_full,
        )
        base_full = base_full.center_effects(full_tr_ld)
        base_full.link = "logit"
        print(
            f"[{datetime.datetime.now():%H:%M:%S}]  base_full done. "
            f"best_epoch={base_full.best_epoch_}",
            flush=True,
        )

        # ── 5. Train base_half ────────────────────────────────────────────
        print(f"[{datetime.datetime.now():%H:%M:%S}]  Training base_half ...", flush=True)
        cw_h1 = sklearn.utils.class_weight.compute_class_weight(
            "balanced", classes=np.unique(y_h1_tr), y=y_h1_tr
        )
        pw_h1 = torch.tensor(cw_h1[1] / cw_h1[0])
        tp_h1 = {
            **trainer_params,
            "loss_fn": nn.BCEWithLogitsLoss(pos_weight=pw_h1.to(device)),
        }
        # Same identity→logit trick as base_full (see comment above).
        base_half = covar_trainer(
            model=BaseNetwork,
            model_params={**model_params, "link": "identity"},
            train_loader=h1_tr_ld, val_loader=h1_va_ld,
            **tp_h1,
        )
        base_half = base_half.center_effects(h1_tr_ld)
        base_half.link = "logit"
        print(
            f"[{datetime.datetime.now():%H:%M:%S}]  base_half done. "
            f"best_epoch={base_half.best_epoch_}",
            flush=True,
        )

        # ── 6. Refit on half_2 ────────────────────────────────────────────
        print(f"[{datetime.datetime.now():%H:%M:%S}]  Fitting refit_age ...", flush=True)
        phm_age = RefitCovarNetwork(base_half, num_covariates=1, orthogonalize=False).to(device)
        phm_age = phm_age.fit(h2_age_tr_ld, h2_age_va_ld, max_iters=400, tol=1e-3)
        print(
            f"[{datetime.datetime.now():%H:%M:%S}]  refit_age done. "
            f"lam={float(phm_age.lam.data):.3e}",
            flush=True,
        )

        print(f"[{datetime.datetime.now():%H:%M:%S}]  Fitting refit_age_sex ...", flush=True)
        phm_sex = RefitCovarNetwork(base_half, num_covariates=2, orthogonalize=False).to(device)
        phm_sex = phm_sex.fit(h2_sex_tr_ld, h2_sex_va_ld, max_iters=400, tol=1e-3)
        print(
            f"[{datetime.datetime.now():%H:%M:%S}]  refit_age_sex done. "
            f"lam={float(phm_sex.lam.data):.3e}",
            flush=True,
        )

        # ── 7. Collect predictions (shuffle=False is mandatory) ───────────
        test_ld = _test_ld()

        y_hat_full,    fx_hat_full    = _collect_preds(base_full, test_ld)
        y_hat_half,    fx_hat_half    = _collect_preds(base_half, test_ld)
        y_hat_phage,   fx_hat_phage   = _collect_preds(phm_age,   test_ld)
        y_hat_phsex,   fx_hat_phsex   = _collect_preds(phm_sex,   test_ld)

        # Controlled predictions: marginalize over the full training Z distribution
        # (h1 + h2).  Both halves share the same confounding level, so the full
        # set is a cleaner (larger) sample from the confounded distribution.
        # Pass raw Z — predict_fz applies center_z internally.
        y_ctrl_phage = _collect_controlled_preds(phm_age, test_ld, Z_full[full_idx, 0:1])
        y_ctrl_phsex = _collect_controlled_preds(phm_sex, test_ld, Z_full[full_idx])

        # ── 8. Summary metrics ────────────────────────────────────────────
        # y_ctrl is only available for refit models (base_* have no fz → no marginalization).
        for name, y_hat, fx_hat, model_obj, y_ctrl in [
            ("base_full",      y_hat_full,  fx_hat_full,  None,    None),
            ("base_half",      y_hat_half,  fx_hat_half,  None,    None),
            ("refit_age",    y_hat_phage, fx_hat_phage, phm_age, y_ctrl_phage),
            ("refit_age_sex",y_hat_phsex, fx_hat_phsex, phm_sex, y_ctrl_phsex),
        ]:
            row = _eval_summary(name, y_hat, fx_hat, model_obj, y_ctrl=y_ctrl)
            row.update({"coef": coef, "fold": fold})
            results_rows.append(row)
            print(
                f"[{datetime.datetime.now():%H:%M:%S}]    {name}: "
                f"auc={row['auc']:.4f}  auc_marg={row['auc_marg']:.4f}  "
                f"bacc={row['bacc']:.4f}  corr(age)={row['corr_age']:+.3f}  "
                f"corr(sex)={row['corr_sex']:+.3f}  "
                f"b_age={row['b_age']:+.4f}  b_sex={row['b_sex']:+.4f}",
                flush=True,
            )
            with open(progress_path, "a", buffering=1) as pf:
                pf.write(json.dumps(row) + "\n")

        # ── 9. Accumulate per-obs exports ────────────────────────────────
        _append_preds("base_full",      coef, fold, y_hat_full,  fx_hat_full)
        _append_preds("base_half",      coef, fold, y_hat_half,  fx_hat_half)
        _append_preds("refit_age",    coef, fold, y_hat_phage, fx_hat_phage)
        _append_preds("refit_age_sex",coef, fold, y_hat_phsex, fx_hat_phsex)
        _append_ctrl("refit_age",     coef, fold, y_ctrl_phage)
        _append_ctrl("refit_age_sex", coef, fold, y_ctrl_phsex)

        # Fitted coefficients — fz.weight de-standardized by _fit_effects, already raw-Z units.
        w_age = phm_age.fz.weight.data.flatten().cpu().numpy()
        coef_rows.append({
            "method": "refit_age", "coef": coef, "fold": fold,
            "intercept": float(phm_age.intercept.data.cpu().numpy().squeeze()),
            "age": float(w_age[0]), "sex": float("nan"),
            "lam": float(phm_age.lam.data),
        })
        w_sex = phm_sex.fz.weight.data.flatten().cpu().numpy()
        coef_rows.append({
            "method": "refit_age_sex", "coef": coef, "fold": fold,
            "intercept": float(phm_sex.intercept.data.cpu().numpy().squeeze()),
            "age": float(w_sex[0]), "sex": float(w_sex[1]),
            "lam": float(phm_sex.lam.data),
        })
        for base_name, base_obj in [("base_full", base_full), ("base_half", base_half)]:
            coef_rows.append({
                "method": base_name, "coef": coef, "fold": fold,
                "intercept": float(base_obj.intercept.data.cpu().numpy().squeeze()),
                "age": float("nan"), "sex": float("nan"), "lam": float("nan"),
            })

        # Lambda paths for both refit models
        for rec in phm_age.lambda_path_:
            lambda_rows.append({"method": "refit_age", "coef": coef, "fold": fold,
                                 "selected_lambda": float(phm_age.lam.data), **rec})
        for rec in phm_sex.lambda_path_:
            lambda_rows.append({"method": "refit_age_sex", "coef": coef, "fold": fold,
                                 "selected_lambda": float(phm_sex.lam.data), **rec})

        # Training fold Z distributions (for R)
        for obs_idx in h2_idx:  # half_2 = refit training set
            train_rows.append({
                "coef": coef, "fold": fold, "half": 2,
                "y": int(y[obs_idx]), "age": float(Z_full[obs_idx, 0]),
                "sex": float(Z_full[obs_idx, 1]),
            })

        # ── 10. Save checkpoints ─────────────────────────────────────────
        ckpt_dir = results_dir + f"coef={coef}/fold={fold}/"
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(base_full.state_dict(),  ckpt_dir + "base_full.pt")
        torch.save(base_half.state_dict(),  ckpt_dir + "base_half.pt")
        torch.save(phm_age.state_dict(),    ckpt_dir + "refit_age.pt")
        torch.save(phm_sex.state_dict(),    ckpt_dir + "refit_age_sex.pt")

        del base_full, base_half, phm_age, phm_sex
        torch.cuda.empty_cache()
        gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# Write R exports and summary CSV
# ══════════════════════════════════════════════════════════════════════════════
import csv as csv_mod
import pandas as pd

rexports_dir = results_dir + "rexports/"
os.makedirs(rexports_dir, exist_ok=True)


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[{datetime.datetime.now():%H:%M:%S}]  → {path}", flush=True)


# raw_results.csv — summary metrics
_write_csv(
    results_dir + "raw_results.csv",
    results_rows,
    ["method", "coef", "fold", "bacc", "auc", "bacc_marg", "auc_marg",
     "corr_age", "corr_sex", "b_age", "b_sex", "lam"],
)

# testset_predictions.csv
_write_csv(
    rexports_dir + "testset_predictions.csv",
    preds_rows,
    ["obs_id", "method", "coef", "fold", "y", "fx"],
)

# testset_predictions_controlled.csv
_write_csv(
    rexports_dir + "testset_predictions_controlled.csv",
    ctrl_rows,
    ["obs_id", "method", "coef", "fold", "y_controlled"],
)

# testset.csv — obs_id matches obs_id in testset_predictions.csv
pd.DataFrame({
    "obs_id": np.arange(len(y_te)),
    "y": y_te, "age": Z_te[:, 0], "sex": Z_te[:, 1],
}).to_csv(rexports_dir + "testset.csv", index=False)
print(f"[{datetime.datetime.now():%H:%M:%S}]  → {rexports_dir}testset.csv", flush=True)

# trainset_folds.csv
_write_csv(
    rexports_dir + "trainset_folds.csv",
    train_rows,
    ["coef", "fold", "half", "y", "age", "sex"],
)

# fitted_coefs.csv
_write_csv(
    rexports_dir + "fitted_coefs.csv",
    coef_rows,
    ["method", "coef", "fold", "intercept", "age", "sex", "lam"],
)

# refit_lambda_paths.csv — one row per (method, coef, fold, lambda)
if lambda_rows:
    lambda_df = pd.DataFrame(lambda_rows)
    lambda_df.to_csv(rexports_dir + "refit_lambda_paths.csv", index=False)
    print(
        f"[{datetime.datetime.now():%H:%M:%S}]  → {rexports_dir}refit_lambda_paths.csv",
        flush=True,
    )

print(
    f"\n[{datetime.datetime.now():%H:%M:%S}] Done. "
    f"{len(results_rows)} fits. Results: {results_dir}",
    flush=True,
)
