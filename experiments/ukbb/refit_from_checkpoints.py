#!/usr/bin/env python
"""Sample-split Panel A/B: refit the age and age+sex heads from released base checkpoints.

For each (coef, fold) this loads the base_full and base_half backbones (and base_half_B for
the cross-fit) from a run's checkpoints, refits the linear f_Z heads without retraining any
CNN, and evaluates on the shared balanced test set. It reports the uncontrolled DNN, the
single-split refit (head on h2, backbone base_half trained on h1), and the 2-fold cross-fit
(the A/B ensemble of the h2-head over base_half and the h1-head over base_half_B, via
CrossFitEnsemble). The partition mirrors train_backbones.py, so each refit lands on the same
half as the released run. aggregate.py reduces the per-fold records to the R-ready rexports
and crossfit_results.csv.

Usage:  python experiments/ukbb/refit_from_checkpoints.py --coefs 0.0,2.0 --folds 0,1,2,3,4
"""
import os
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
COEFS_DEFAULT = [0.0, 2.0]
N_SPLITS = 5
NTRAIN_PER_HALF = 2500
NTEST = 2500
BATCH_SIZE = 48
NUM_WORKERS = 16
REFIT_MAX_ITERS = 400          # IRLS backfitting controls (match the released run)
REFIT_TOL = 1e-3
TRUE_B_AGE_RAW = -0.298        # DGP age coef in logit/raw units (for reference)
TRUE_B_SEX_RAW = 2.0
SRC_RUN = ROOT / "experiments/ukbb/runs/2026-04-26_13-16-37_final_v2"
OUT_RUN = ROOT / "experiments/ukbb/runs/final_v2_refit"
MODEL_PARAMS = dict(backbone=ResNet50, backbone_params={"pretrained_model": ""},
                    num_covariates=0, link="logit")

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--coefs", type=str, default="0.0,2.0")
parser.add_argument("--folds", type=str, default="0,1,2,3,4")
parser.add_argument("--int-coef", type=float, default=0.0)
parser.add_argument("--rho-coef", type=float, default=0.0)
parser.add_argument("--src-run", type=str, default=None)
parser.add_argument("--out-run", type=str, default=None)
args = parser.parse_args()
COEFS = [float(c) for c in args.coefs.split(",")]
FOLDS = [int(f) for f in args.folds.split(",")]
DGP = dict(int_coef=args.int_coef, rho_coef=args.rho_coef)

# non-additive DGP variants read and write their own run dirs
if any(DGP.values()):
    TAG = f"nonadd_int{args.int_coef:g}_rho{args.rho_coef:g}"
    SRC_RUN = ROOT / "experiments/ukbb/runs" / TAG
    OUT_RUN = ROOT / "experiments/ukbb/runs" / f"{TAG}_refit"
if args.src_run:
    SRC_RUN = Path(args.src_run)
if args.out_run:
    OUT_RUN = Path(args.out_run)

# ── setup ─────────────────────────────────────────────────────────────────────
seed_everything(RANDOM_STATE)
device = torch.device(f"cuda:{args.gpu}")
tform = default_transforms()
print(f"[{datetime.datetime.now():%H:%M:%S}] device=cuda:{args.gpu}  src={SRC_RUN.name}  out={OUT_RUN.name}", flush=True)

# ── data ──────────────────────────────────────────────────────────────────────
d = load_ukbb_data()
X, y, Z = d["X"], d["y"], d["Z"]
X_test, y_test, Z_test = d["X_test"], d["y_test"], d["Z_test"]

# balanced (coef=0) test set, shared across every coef/fold; shuffle=False so obs_id is stable
idx_test = resample_synthetic(y_test, Z_test, NTEST, 0.0, RANDOM_STATE)
X_te, y_te, Z_te = X_test[idx_test], y_test[idx_test], Z_test[idx_test]


def test_loader():
    ds = NumpyCovarDataset(X_te, y_te, Z_te, tform)
    return fast_loader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


def refit_loaders(idx_all, seed, z_cols):
    """Stratified train/val loaders over global indices, carrying Z columns z_cols."""
    tr_loc, va_loc = train_test_split(np.arange(len(idx_all)), test_size=0.2,
                                      random_state=seed, stratify=y[idx_all])
    idx_tr, idx_va = idx_all[tr_loc], idx_all[va_loc]
    tr = fast_loader(NumpyCovarDataset(X[idx_tr], y[idx_tr], Z[idx_tr][:, z_cols], tform),
                     batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    va = fast_loader(NumpyCovarDataset(X[idx_va], y[idx_va], Z[idx_va][:, z_cols], tform),
                     batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return tr, va


# ── prediction ────────────────────────────────────────────────────────────────
@torch.no_grad()
def collect_preds(model, loader):
    """Per-obs (y_hat, fx_hat); refit models slice Z to their covariate count."""
    model.eval()
    ys, fxs = [], []
    for batch in loader:
        x = batch["X"].to(device, non_blocking=True)
        if getattr(model, "num_covariates", 0) > 0:
            z = batch["Z"].to(device, non_blocking=True)[:, :model.num_covariates]
            ys.append(model(x, z).cpu().numpy())
            fxs.append(model.predict_fx(x, z=None).cpu().numpy())
        else:
            ys.append(model(x).cpu().numpy())
            fxs.append(model.predict_fx(x).cpu().numpy())
    return np.concatenate(ys).flatten(), np.concatenate(fxs).flatten()


@torch.no_grad()
def collect_controlled(model, loader, Z_train_raw):
    """Marginalize sigmoid(intercept + fx_i + fz(z_j)) over the training-Z distribution."""
    model.eval()
    out = []
    Z_tr = torch.tensor(Z_train_raw, dtype=torch.float32).to(device)
    fz_tr = model.predict_fz(Z_tr)
    for batch in loader:
        x = batch["X"].to(device, non_blocking=True)
        fx = model.predict_fx(x)
        eta = model.intercept + fx.unsqueeze(1).expand(-1, Z_tr.shape[0], -1) + fz_tr
        out.append(model.output_func(eta).mean(dim=1).cpu().numpy())
    return np.concatenate(out).flatten()


def eval_summary(name, coef, fold, y_hat, fx_hat, model=None, y_ctrl=None):
    """Per-fold metrics; adds marginalized auc/bacc and fitted coefs for refit models."""
    row = dict(method=name, coef=coef, fold=fold,
               bacc=balanced_accuracy_score(y_te, (y_hat >= 0.5).astype(int)),
               auc=roc_auc_score(y_te, y_hat),
               bacc_marg=float("nan"), auc_marg=float("nan"),
               corr_age=float(np.corrcoef(fx_hat, Z_te[:, 0])[0, 1]),
               corr_sex=float(np.corrcoef(fx_hat, Z_te[:, 1])[0, 1]),
               b_age=float("nan"), b_sex=float("nan"), lam=float("nan"))
    if y_ctrl is not None:
        row["bacc_marg"] = balanced_accuracy_score(y_te, (y_ctrl >= 0.5).astype(int))
        row["auc_marg"] = roc_auc_score(y_te, y_ctrl)
    if model is not None and hasattr(model, "fz"):
        w = model.fz.weight.data.flatten().cpu().numpy()
        row["b_age"] = float(w[0])
        row["b_sex"] = float(w[1]) if w.size > 1 else float("nan")
        row["lam"] = float(model.lam.data)
    return row


@torch.no_grad()
def crossfit_summary(name, coef, fold, ens, ncov, loader):
    """Cross-fit ensemble metrics: regular AUC + logit-space marginal AUC.

    After .recenter(full), f_Z has zero mean on the training sample, so the marginal
    prediction over the training-Z distribution is sigmoid(intercept + f_X) — obtained by
    dropping f_Z from eta. This matches the released crossfit_results.csv convention.
    """
    ens.eval()
    out = ens.models[0].output_func
    y_reg, y_marg, fx = [], [], []
    for batch in loader:
        x = batch["X"].to(device, non_blocking=True)
        z = batch["Z"].to(device, non_blocking=True)[:, :ncov]
        eta = ens.predict_eta(x, z)
        y_reg.append(out(eta).cpu().numpy())
        y_marg.append(out(eta - ens.predict_fz(z)).cpu().numpy())
        fx.append(ens.predict_fx(x).cpu().numpy())
    y_reg = np.concatenate(y_reg).flatten()
    y_marg = np.concatenate(y_marg).flatten()
    fx = np.concatenate(fx).flatten()
    w = np.mean([m.fz.weight.data.flatten().cpu().numpy() for m in ens.models], axis=0)
    return dict(method=name, coef=coef, fold=fold,
                auc=float(roc_auc_score(y_te, y_reg)),
                bacc=float(balanced_accuracy_score(y_te, (y_reg >= 0.5).astype(int))),
                auc_marg=float(roc_auc_score(y_te, y_marg)),
                bacc_marg=float(balanced_accuracy_score(y_te, (y_marg >= 0.5).astype(int))),
                corr_age=float(np.corrcoef(fx, Z_te[:, 0])[0, 1]),
                corr_sex=float(np.corrcoef(fx, Z_te[:, 1])[0, 1]),
                b_age=float(w[0]), b_sex=float(w[1]) if w.size > 1 else float("nan"),
                lam=float("nan"))


# ── refit one (coef, fold) ────────────────────────────────────────────────────
def run_fold(coef, fold):
    fold_seed = RANDOM_STATE + fold
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    train_ix = list(cv.split(np.zeros(len(y)), y))[fold][0]

    # partition (mirror train_backbones.py: split pool before resampling)
    pool_A, pool_B = train_test_split(train_ix, test_size=0.5, random_state=fold_seed, stratify=y[train_ix])
    h1_idx = pool_A[resample_synthetic(y[pool_A], Z[pool_A], NTRAIN_PER_HALF, coef, fold_seed, **DGP)]
    h2_idx = pool_B[resample_synthetic(y[pool_B], Z[pool_B], NTRAIN_PER_HALF, coef, fold_seed + 100, **DGP)]
    full_idx = np.concatenate([h1_idx, h2_idx])

    # backbones (load released state dicts; already centered, link=logit)
    ckpt_dir = SRC_RUN / f"coef={coef}" / f"fold={fold}"
    base_full = BaseNetwork(**MODEL_PARAMS).to(device)
    base_full.load_state_dict(torch.load(ckpt_dir / "base_full.pt", map_location=device))
    base_full.eval()
    base_half = BaseNetwork(**MODEL_PARAMS).to(device)
    base_half.load_state_dict(torch.load(ckpt_dir / "base_half.pt", map_location=device))
    base_half.eval()

    # refit (age, then age+sex) on h2; same inner-split seed as the released run
    # age and age+sex share the inner split seed (+202) so both refits validate on the
    # same held-out 20% of h2 — a paired comparison; the B-side mirrors this with +203.
    age_tr, age_va = refit_loaders(h2_idx, fold_seed + 202, z_cols=[0])
    refit_age = RefitCovarNetwork(base_half, num_covariates=1, orthogonalize=False).to(device)
    refit_age = refit_age.fit(age_tr, age_va, max_iters=REFIT_MAX_ITERS, tol=REFIT_TOL)
    sex_tr, sex_va = refit_loaders(h2_idx, fold_seed + 202, z_cols=[0, 1])
    refit_sex = RefitCovarNetwork(base_half, num_covariates=2, orthogonalize=False).to(device)
    refit_sex = refit_sex.fit(sex_tr, sex_va, max_iters=REFIT_MAX_ITERS, tol=REFIT_TOL)

    # test evaluation (single-split A-side)
    te = test_loader()
    y_full, fx_full = collect_preds(base_full, te)
    y_half, fx_half = collect_preds(base_half, te)
    y_age, fx_age = collect_preds(refit_age, te)
    y_sex, fx_sex = collect_preds(refit_sex, te)
    yc_age = collect_controlled(refit_age, te, Z[full_idx][:, [0]])
    yc_sex = collect_controlled(refit_sex, te, Z[full_idx])

    # save A-side checkpoints before the cross-fit recenter mutates their centering
    out_dir = OUT_RUN / f"coef={coef}" / f"fold={fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(refit_age.state_dict(), out_dir / "refit_age.pt")
    torch.save(refit_sex.state_dict(), out_dir / "refit_age_sex.pt")

    # cross-fit (2-fold A/B ensemble): head B on h1 over base_half_B (backbone trained on h2)
    base_half_B = BaseNetwork(**MODEL_PARAMS).to(device)
    base_half_B.load_state_dict(torch.load(ckpt_dir / "base_half_B.pt", map_location=device))
    base_half_B.eval()
    ageB_tr, ageB_va = refit_loaders(h1_idx, fold_seed + 203, z_cols=[0])
    refit_ageB = RefitCovarNetwork(base_half_B, num_covariates=1, orthogonalize=False).to(device)
    refit_ageB = refit_ageB.fit(ageB_tr, ageB_va, max_iters=REFIT_MAX_ITERS, tol=REFIT_TOL)
    sexB_tr, sexB_va = refit_loaders(h1_idx, fold_seed + 203, z_cols=[0, 1])
    refit_sexB = RefitCovarNetwork(base_half_B, num_covariates=2, orthogonalize=False).to(device)
    refit_sexB = refit_sexB.fit(sexB_tr, sexB_va, max_iters=REFIT_MAX_ITERS, tol=REFIT_TOL)
    # recenter grid must carry Z with exactly the model's covariate count
    full_age = fast_loader(NumpyCovarDataset(X[full_idx], y[full_idx], Z[full_idx][:, [0]], tform),
                           batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    full_sex = fast_loader(NumpyCovarDataset(X[full_idx], y[full_idx], Z[full_idx], tform),
                           batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    # sample-split rows: A-side single refit at its own (h2) centering, i.e. marginalized over
    # h2; computed before the ensemble recenter mutates the A-side centering. These feed the
    # figure's "Sample split" estimator under the same logit-space marginalization as the cross-fit.
    crossfit = [crossfit_summary("refit_age", coef, fold, CrossFitEnsemble([refit_age]), 1, te),
                crossfit_summary("refit_age_sex", coef, fold, CrossFitEnsemble([refit_sex]), 2, te)]
    # cross-fit rows: A/B ensemble recentered on the full training sample
    cf_age = CrossFitEnsemble([refit_age, refit_ageB]).recenter(full_age)
    cf_sex = CrossFitEnsemble([refit_sex, refit_sexB]).recenter(full_sex)
    crossfit += [crossfit_summary("crossfit_age", coef, fold, cf_age, 1, te),
                 crossfit_summary("crossfit_age_sex", coef, fold, cf_sex, 2, te)]
    for r in crossfit:
        print(f"[{datetime.datetime.now():%H:%M:%S}]    {r['method']:>14}: auc={r['auc']:.4f} "
              f"auc_marg={r['auc_marg']:.4f} b_age={r['b_age']:+.4f} b_sex={r['b_sex']:+.4f}", flush=True)

    # records
    summary = [
        eval_summary("base_full", coef, fold, y_full, fx_full),
        eval_summary("base_half", coef, fold, y_half, fx_half),
        eval_summary("refit_age", coef, fold, y_age, fx_age, refit_age, yc_age),
        eval_summary("refit_age_sex", coef, fold, y_sex, fx_sex, refit_sex, yc_sex),
    ]
    for r in summary:
        print(f"[{datetime.datetime.now():%H:%M:%S}]    {r['method']:>14}: auc={r['auc']:.4f} "
              f"auc_marg={r['auc_marg']:.4f} b_age={r['b_age']:+.4f} b_sex={r['b_sex']:+.4f}", flush=True)
    coefs = []
    for name, m in [("refit_age", refit_age), ("refit_age_sex", refit_sex)]:
        w = m.fz.weight.data.flatten().cpu().numpy()
        coefs.append(dict(method=name, coef=coef, fold=fold,
                          intercept=float(m.intercept.data.cpu().numpy().squeeze()),
                          age=float(w[0]), sex=float(w[1]) if w.size > 1 else float("nan"),
                          lam=float(m.lam.data)))
    for name, m in [("base_full", base_full), ("base_half", base_half)]:
        coefs.append(dict(method=name, coef=coef, fold=fold,
                          intercept=float(m.intercept.data.cpu().numpy().squeeze()),
                          age=float("nan"), sex=float("nan"), lam=float("nan")))
    lambdas = ([dict(method="refit_age", coef=coef, fold=fold,
                     selected_lambda=float(refit_age.lam.data), **rec) for rec in refit_age.lambda_path_]
               + [dict(method="refit_age_sex", coef=coef, fold=fold,
                       selected_lambda=float(refit_sex.lam.data), **rec) for rec in refit_sex.lambda_path_])

    np.savez_compressed(
        out_dir / "record.npz",
        coef=coef, fold=fold,
        test_y=y_te, test_age=Z_te[:, 0], test_sex=Z_te[:, 1],
        train_y=y[h2_idx], train_age=Z[h2_idx, 0], train_sex=Z[h2_idx, 1],
        y__base_full=y_full, fx__base_full=fx_full,
        y__base_half=y_half, fx__base_half=fx_half,
        y__refit_age=y_age, fx__refit_age=fx_age,
        y__refit_age_sex=y_sex, fx__refit_age_sex=fx_sex,
        yctrl__refit_age=yc_age, yctrl__refit_age_sex=yc_sex,
        summary=np.array(json.dumps(summary)),
        coefs=np.array(json.dumps(coefs)),
        lambdas=np.array(json.dumps(lambdas)),
        crossfit=np.array(json.dumps(crossfit)),
    )
    del base_full, base_half, base_half_B, refit_age, refit_sex, refit_ageB, refit_sexB
    del cf_age, cf_sex, te, full_age, full_sex
    torch.cuda.empty_cache()
    gc.collect()


# ── main ──────────────────────────────────────────────────────────────────────
for coef in COEFS:
    for fold in FOLDS:
        rec = OUT_RUN / f"coef={coef}" / f"fold={fold}" / "record.npz"
        if rec.exists():
            print(f"[{datetime.datetime.now():%H:%M:%S}] coef={coef} fold={fold} cached — skip.", flush=True)
            continue
        print(f"\n[{datetime.datetime.now():%H:%M:%S}] ══ coef={coef} fold={fold} ══", flush=True)
        run_fold(coef, fold)
print(f"\n[{datetime.datetime.now():%H:%M:%S}] Done. Records in {OUT_RUN}", flush=True)
