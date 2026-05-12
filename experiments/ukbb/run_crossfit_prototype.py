#!/usr/bin/env python
"""Cross-fit post-hoc prototype on UKBB.

Augments the existing 2026-04-26_..._final_v2 run with B-side artifacts:

  base_half_B            — BaseNetwork trained on h2_idx (mirror of base_half)
  posthoc_age_BA         — PostHocCovarNetwork(base_half_B, num_covariates=1) fit on h1_idx
  posthoc_age_sex_BA     — PostHocCovarNetwork(base_half_B, num_covariates=2) fit on h1_idx

The A-side (base_half + posthoc_age + posthoc_age_sex) is reused from final_v2
as-is — same seeds, same data partitions, identical fit. Only B-side is new.

Per (coef, fold), six methods are reported:
  posthoc_age, posthoc_age_BA, crossfit_age,
  posthoc_age_sex, posthoc_age_sex_BA, crossfit_age_sex.

The crossfit_* methods average η = intercept + fx + fz across A and B before
applying sigmoid.

Skip-if-exists: base_half_B.pt, posthoc_age_BA.pt, posthoc_age_sex_BA.pt are
loaded from disk if present, only refit when missing.

Usage:
    cd experiments/ukbb/
    conda run --no-capture-output -n dl-mri \
        python run_crossfit_prototype.py --gpu 0 --coef 0.0 --folds 0
    # or:
    python run_crossfit_prototype.py --gpu 0 --coef 0.0           # all 5 folds
    python run_crossfit_prototype.py --gpu 0 --coef 2.0           # all 5 folds
"""
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
from cocodeel.posthoc_model import PostHocCovarNetwork
from cocodeel.trainer import covar_trainer

# ── Config (mirror run_ukbb_experiment.py) ───────────────────────────────────
N_SPLITS = 5
NTRAIN_PER_HALF = 2500
NTEST = 2500
BATCH_SIZE = 48
NUM_WORKERS = 16

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--coef", type=float, required=True, choices=[0.0, 2.0])
parser.add_argument("--folds", type=str, default="0,1,2,3,4",
                    help="comma-separated fold indices (default: all 5)")
args = parser.parse_args()
GPU = args.gpu
COEF = args.coef
FOLDS = [int(f) for f in args.folds.split(",")]

RUN_DIR = (
    "/home/RDC/pfeuffma/Research/proj-orthogonalisation/"
    "experiments/ukbb/runs/2026-04-26_13-16-37_final_v2/"
)
OUT_CSV = RUN_DIR + "crossfit_results.csv"
PROGRESS = RUN_DIR + "progress_crossfit.log"


# ── Setup ─────────────────────────────────────────────────────────────────────
seed_everything(RANDOM_STATE)
device = torch.device(f"cuda:{GPU}")
print(f"[{datetime.datetime.now():%H:%M:%S}] device=cuda:{GPU}  "
      f"coef={COEF}  folds={FOLDS}", flush=True)

print(f"[{datetime.datetime.now():%H:%M:%S}] Loading data ...", flush=True)
d = load_ukbb_data()
X, y, Z_full = d["X"], d["y"], d["Z_full"]
X_test, y_test, Z_full_test = d["X_test"], d["y_test"], d["Z_full_test"]

tform = default_transforms()
model_params = default_model_params()
trainer_params = default_trainer_params(GPU)
trainer_params["device"] = f"cuda:{GPU}"

idx_test = resample_synthetic(y_test, Z_full_test, NTEST, 0.0, RANDOM_STATE)
X_te, y_te, Z_te = X_test[idx_test], y_test[idx_test], Z_full_test[idx_test]


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


# ── Prediction helpers (logit-space, for ensemble) ───────────────────────────
def _collect_eta(model, loader):
    """Return raw η = intercept + predict_fx(x) + predict_fz(z) on test set."""
    model.eval()
    etas = []
    with torch.no_grad():
        for batch in loader:
            x = batch["X"].to(device, non_blocking=True)
            z = batch["Z"].to(device, non_blocking=True)
            z_model = z[:, :model.num_covariates]
            fx = model.predict_fx(x, z=None)        # (batch, 1)
            fz = model.predict_fz(z_model)          # (batch, 1)
            eta = model.intercept + fx + fz         # (batch, 1)
            etas.append(eta.cpu().numpy())
    return np.concatenate(etas).flatten()


def _collect_eta_marg(model, loader, Z_train_raw):
    """Marginalize η over training Z distribution.

    eta_marg_i = mean_j[ intercept + fx(x_i) + fz(z_j) ]
                = intercept + fx(x_i) + mean_j[ fz(z_j) ]
    (linearity makes this exact regardless of N_train).

    Returns shape (N_test,).
    """
    model.eval()
    etas = []
    with torch.no_grad():
        Z_tr_t = torch.tensor(Z_train_raw, dtype=torch.float32).to(device)
        fz_mean = model.predict_fz(Z_tr_t).mean(dim=0)  # (1,)
        for batch in loader:
            x = batch["X"].to(device, non_blocking=True)
            fx = model.predict_fx(x)                    # (batch, 1)
            eta = model.intercept + fx + fz_mean        # (batch, 1)
            etas.append(eta.cpu().numpy())
    return np.concatenate(etas).flatten()


def _collect_fx_fz_on_train(phm, full_train_idx):
    """Per-obs fx and fz from phm on the full training set h1∪h2.

    Used to compute the post-ensemble centring constants c_X, c_Z that enforce
    n^{-1} Σ_i f̂_X^cf(X_i) = 0 on the full training sample (see app:crossfit-proof).
    Each phm is centred on its own training half by Center modules, so the
    ensemble loses centring on h1∪h2; this restores it.
    """
    train_ds = NumpyCovarDataset(
        X[full_train_idx], y[full_train_idx], Z_full[full_train_idx], tform
    )
    train_ld = fast_loader(
        train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    phm.eval()
    fxs, fzs = [], []
    with torch.no_grad():
        for batch in train_ld:
            x = batch["X"].to(device, non_blocking=True)
            z = batch["Z"].to(device, non_blocking=True)
            z_model = z[:, : phm.num_covariates]
            fxs.append(phm.predict_fx(x).cpu().numpy())
            fzs.append(phm.predict_fz(z_model).cpu().numpy())
    return np.concatenate(fxs).flatten(), np.concatenate(fzs).flatten()


def _eval_metrics(y_hat, fx_hat=None, y_marg=None):
    """Compute auc/bacc + marginalized variants + corr."""
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


# ── Main loop ─────────────────────────────────────────────────────────────────
results_rows = []

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
all_train_ix = list(cv.split(np.zeros(len(y)), y))

for fold in FOLDS:
    train_ix = all_train_ix[fold][0]
    fold_seed = RANDOM_STATE + fold
    ckpt_dir = RUN_DIR + f"coef={COEF}/fold={fold}/"
    assert os.path.exists(ckpt_dir + "base_half.pt"),     f"missing {ckpt_dir}base_half.pt"
    assert os.path.exists(ckpt_dir + "posthoc_age.pt"),   f"missing {ckpt_dir}posthoc_age.pt"
    assert os.path.exists(ckpt_dir + "posthoc_age_sex.pt"), f"missing {ckpt_dir}posthoc_age_sex.pt"

    print(f"\n[{datetime.datetime.now():%H:%M:%S}] ══ coef={COEF} fold={fold} ══", flush=True)

    # Reproduce same data partition as run_ukbb_experiment.py.
    pool_A, pool_B = train_test_split(
        train_ix, test_size=0.5, random_state=fold_seed, stratify=y[train_ix]
    )
    h1_local = resample_synthetic(y[pool_A], Z_full[pool_A], NTRAIN_PER_HALF, COEF, fold_seed)
    h1_idx = pool_A[h1_local]
    h2_local = resample_synthetic(y[pool_B], Z_full[pool_B], NTRAIN_PER_HALF, COEF, fold_seed + 100)
    h2_idx = pool_B[h2_local]

    # ── Loaders (B-side: backbone trained on h2, posthoc fit on h1) ──────────
    h2_tr_ld_B,    h2_va_ld_B,    y_h2_tr = _make_loaders(h2_idx, fold_seed + 201)
    h1_age_tr_ld,  h1_age_va_ld,  _ = _make_loaders(h1_idx, fold_seed + 203, Z=Z_full[:, 0:1])
    h1_sex_tr_ld,  h1_sex_va_ld,  _ = _make_loaders(h1_idx, fold_seed + 203, Z=Z_full)

    # ── A-side: load existing checkpoints ─────────────────────────────────────
    print(f"[{datetime.datetime.now():%H:%M:%S}] Loading A-side checkpoints ...", flush=True)
    base_half = BaseNetwork(**model_params).to(device)
    base_half.load_state_dict(torch.load(ckpt_dir + "base_half.pt", map_location=device))
    base_half.link = "logit"
    base_half.eval()

    phm_A_age = PostHocCovarNetwork(base_half, num_covariates=1, orthogonalize=False).to(device)
    phm_A_age.load_state_dict(torch.load(ckpt_dir + "posthoc_age.pt", map_location=device))
    phm_A_age.eval()

    phm_A_sex = PostHocCovarNetwork(base_half, num_covariates=2, orthogonalize=False).to(device)
    phm_A_sex.load_state_dict(torch.load(ckpt_dir + "posthoc_age_sex.pt", map_location=device))
    phm_A_sex.eval()

    # ── B-side: train base_half_B (skip if checkpoint exists) ─────────────────
    bhB_path = ckpt_dir + "base_half_B.pt"
    if os.path.exists(bhB_path):
        print(f"[{datetime.datetime.now():%H:%M:%S}] base_half_B exists — loading.", flush=True)
        base_half_B = BaseNetwork(**model_params).to(device)
        base_half_B.load_state_dict(torch.load(bhB_path, map_location=device))
        base_half_B = base_half_B.center_effects(h2_tr_ld_B)
        base_half_B.link = "logit"
        base_half_B.eval()
    else:
        print(f"[{datetime.datetime.now():%H:%M:%S}] Training base_half_B on h2 ...", flush=True)
        cw_h2 = sklearn.utils.class_weight.compute_class_weight(
            "balanced", classes=np.unique(y_h2_tr), y=y_h2_tr
        )
        pw_h2 = torch.tensor(cw_h2[1] / cw_h2[0])
        tp_h2 = {
            **trainer_params,
            "loss_fn": nn.BCEWithLogitsLoss(pos_weight=pw_h2.to(device)),
        }
        base_half_B = covar_trainer(
            model=BaseNetwork,
            model_params={**model_params, "link": "identity"},
            train_loader=h2_tr_ld_B, val_loader=h2_va_ld_B,
            **tp_h2,
        )
        base_half_B = base_half_B.center_effects(h2_tr_ld_B)
        base_half_B.link = "logit"
        print(f"[{datetime.datetime.now():%H:%M:%S}] base_half_B done. "
              f"best_epoch={base_half_B.best_epoch_}", flush=True)
        torch.save(base_half_B.state_dict(), bhB_path)

    # ── posthoc_age_BA (skip if exists) ───────────────────────────────────────
    phm_B_age_path = ckpt_dir + "posthoc_age_BA.pt"
    phm_B_age = PostHocCovarNetwork(base_half_B, num_covariates=1, orthogonalize=False).to(device)
    if os.path.exists(phm_B_age_path):
        print(f"[{datetime.datetime.now():%H:%M:%S}] posthoc_age_BA exists — loading.", flush=True)
        phm_B_age.load_state_dict(torch.load(phm_B_age_path, map_location=device))
        phm_B_age.eval()
    else:
        print(f"[{datetime.datetime.now():%H:%M:%S}] Fitting posthoc_age_BA on h1 ...", flush=True)
        phm_B_age = phm_B_age.fit(h1_age_tr_ld, h1_age_va_ld, max_iters=400, tol=1e-3)
        print(f"[{datetime.datetime.now():%H:%M:%S}] posthoc_age_BA done. "
              f"lam={float(phm_B_age.lam.data):.3e}", flush=True)
        torch.save(phm_B_age.state_dict(), phm_B_age_path)

    # ── posthoc_age_sex_BA (skip if exists) ──────────────────────────────────
    phm_B_sex_path = ckpt_dir + "posthoc_age_sex_BA.pt"
    phm_B_sex = PostHocCovarNetwork(base_half_B, num_covariates=2, orthogonalize=False).to(device)
    if os.path.exists(phm_B_sex_path):
        print(f"[{datetime.datetime.now():%H:%M:%S}] posthoc_age_sex_BA exists — loading.", flush=True)
        phm_B_sex.load_state_dict(torch.load(phm_B_sex_path, map_location=device))
        phm_B_sex.eval()
    else:
        print(f"[{datetime.datetime.now():%H:%M:%S}] Fitting posthoc_age_sex_BA on h1 ...", flush=True)
        phm_B_sex = phm_B_sex.fit(h1_sex_tr_ld, h1_sex_va_ld, max_iters=400, tol=1e-3)
        print(f"[{datetime.datetime.now():%H:%M:%S}] posthoc_age_sex_BA done. "
              f"lam={float(phm_B_sex.lam.data):.3e}", flush=True)
        torch.save(phm_B_sex.state_dict(), phm_B_sex_path)

    # ── Predictions on shared balanced test set ──────────────────────────────
    test_ld = _test_ld()

    def _collect_fx(model):
        model.eval()
        fxs = []
        with torch.no_grad():
            for batch in test_ld:
                x = batch["X"].to(device, non_blocking=True)
                fxs.append(model.predict_fx(x, z=None).cpu().numpy())
        return np.concatenate(fxs).flatten()

    def _coefs(phm):
        """Return (b_age, b_sex) — second is NaN for num_covariates=1 models."""
        w = phm.fz.weight.data.flatten().cpu().numpy()
        return float(w[0]), float(w[1]) if w.size > 1 else float("nan")

    def _eval_pair(phm_A, phm_B, name_A, name_B, name_ens,
                   z_train_A_cols, z_train_B_cols):
        """Run prediction + ensemble + metrics for one (A, B) post-hoc pair."""
        eta_A = _collect_eta(phm_A, test_ld)
        eta_B = _collect_eta(phm_B, test_ld)
        eta_ens = 0.5 * (eta_A + eta_B)
        y_A   = 1.0 / (1.0 + np.exp(-eta_A))
        y_B   = 1.0 / (1.0 + np.exp(-eta_B))
        y_ens = 1.0 / (1.0 + np.exp(-eta_ens))

        eta_A_marg = _collect_eta_marg(phm_A, test_ld, z_train_A_cols)
        eta_B_marg = _collect_eta_marg(phm_B, test_ld, z_train_B_cols)
        eta_ens_marg = 0.5 * (eta_A_marg + eta_B_marg)
        y_A_marg   = 1.0 / (1.0 + np.exp(-eta_A_marg))
        y_B_marg   = 1.0 / (1.0 + np.exp(-eta_B_marg))

        fx_A = _collect_fx(phm_A)
        fx_B = _collect_fx(phm_B)
        fx_ens = 0.5 * (fx_A + fx_B)

        # Post-ensemble centring on full training sample (h1 ∪ h2).
        # Each phm is centred on its own training half; the ensemble loses
        # that constraint on the union. We compute c_X, c_Z on h1∪h2 and
        # apply the location reparametrisation from app:crossfit-proof:
        # intercept ← intercept + c_X + c_Z, fx ← fx − c_X, fz ← fz − c_Z.
        # The joint η is invariant; only the marginal-over-training-Z
        # prediction shifts by +c_Z (correctly marginalising over h1∪h2).
        full_train_idx = np.concatenate([h1_idx, h2_idx])
        fx_A_tr, fz_A_tr = _collect_fx_fz_on_train(phm_A, full_train_idx)
        fx_B_tr, fz_B_tr = _collect_fx_fz_on_train(phm_B, full_train_idx)
        fx_ens_tr = 0.5 * (fx_A_tr + fx_B_tr)
        fz_ens_tr = 0.5 * (fz_A_tr + fz_B_tr)
        c_X = float(fx_ens_tr.mean())
        c_Z = float(fz_ens_tr.mean())
        eta_ens_marg_centred = eta_ens_marg + c_Z
        y_ens_marg = 1.0 / (1.0 + np.exp(-eta_ens_marg_centred))

        ba_A, bs_A = _coefs(phm_A)
        ba_B, bs_B = _coefs(phm_B)
        ba_ens = 0.5 * (ba_A + ba_B)
        bs_ens = 0.5 * (bs_A + bs_B) if not (np.isnan(bs_A) or np.isnan(bs_B)) else float("nan")
        lam_A = float(phm_A.lam.data)
        lam_B = float(phm_B.lam.data)

        local_rows = []
        nan = float("nan")
        for nm, y_hat, fx_hat, y_marg, b_age, b_sex, lam, cx_row, cz_row in [
            (name_A,   y_A,   fx_A,   y_A_marg,   ba_A,   bs_A,   lam_A, nan, nan),
            (name_B,   y_B,   fx_B,   y_B_marg,   ba_B,   bs_B,   lam_B, nan, nan),
            (name_ens, y_ens, fx_ens, y_ens_marg, ba_ens, bs_ens, nan,   c_X, c_Z),
        ]:
            m = _eval_metrics(y_hat, fx_hat=fx_hat, y_marg=y_marg)
            m.update({"method": nm, "coef": COEF, "fold": fold,
                      "b_age": b_age, "b_sex": b_sex, "lam": lam,
                      "c_X": cx_row, "c_Z": cz_row})
            local_rows.append(m)
            extra = f"  c_X={cx_row:+.3f}  c_Z={cz_row:+.3f}" if not np.isnan(cx_row) else ""
            print(f"[{datetime.datetime.now():%H:%M:%S}]    {nm:>20}: "
                  f"auc={m['auc']:.4f}  auc_marg={m['auc_marg']:.4f}  "
                  f"bacc={m['bacc']:.4f}  corr(age)={m['corr_age']:+.3f}  "
                  f"corr(sex)={m['corr_sex']:+.3f}  "
                  f"b_age={b_age:+.4f}  b_sex={b_sex:+.4f}  lam={lam:.3e}{extra}", flush=True)
            with open(PROGRESS, "a", buffering=1) as pf:
                pf.write(json.dumps(m) + "\n")

        delta = eta_A - eta_B
        print(f"[{datetime.datetime.now():%H:%M:%S}]    diag {name_ens}: η_A − η_B  "
              f"mean={delta.mean():+.3f}  std={delta.std():.3f}", flush=True)
        return local_rows

    # Age-only pair (Z = age)
    rows_age = _eval_pair(
        phm_A_age, phm_B_age,
        "posthoc_age", "posthoc_age_BA", "crossfit_age",
        Z_full[h2_idx, 0:1], Z_full[h1_idx, 0:1],
    )
    # Age+sex pair (Z = age + sex)
    rows_sex = _eval_pair(
        phm_A_sex, phm_B_sex,
        "posthoc_age_sex", "posthoc_age_sex_BA", "crossfit_age_sex",
        Z_full[h2_idx, :], Z_full[h1_idx, :],
    )

    results_rows.extend(rows_age)
    results_rows.extend(rows_sex)

    del base_half, base_half_B, phm_A_age, phm_B_age, phm_A_sex, phm_B_sex, test_ld
    torch.cuda.empty_cache()
    gc.collect()


# ── Append to crossfit_results.csv (preserve existing rows from prior runs) ──
df_new = pd.DataFrame(results_rows)
if os.path.exists(OUT_CSV):
    df_old = pd.read_csv(OUT_CSV)
    # Drop any rows for this (coef, folds) combination — rerunning overwrites
    keep = ~(df_old.coef.eq(COEF) & df_old.fold.isin(FOLDS))
    df = pd.concat([df_old[keep], df_new], ignore_index=True)
else:
    df = df_new
df.to_csv(OUT_CSV, index=False)

print(f"\n[{datetime.datetime.now():%H:%M:%S}] Done. "
      f"Wrote {len(results_rows)} rows ({len(FOLDS)} fold(s) × 6 methods) "
      f"to {OUT_CSV}", flush=True)
