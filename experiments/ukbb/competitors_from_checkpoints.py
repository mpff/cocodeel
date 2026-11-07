#!/usr/bin/env python
"""Post-hoc competitors on the released backbones: orthogonalisation, feature erasure, last-layer retraining.

For each (coef, fold) this loads base_full (the DNN trained on the full confounded sample, no
control), extracts its features phi on that sample and on the balanced test set, and fits three
post-hoc corrections for Z = age, all on the full training sample:

  posthoc_web           eta = beta0 + f_X(x) - (z - zbar)^T b,  b = argmin ||f_X - Z_c b||
                        (base_full's head is kept; the linear Z-component of f_X is removed)
  posthoc_web_features  eta = beta0 + (phi_c - (z - zbar)^T B)^T w      (Weber et al.)
  leace                 eta = beta0 + (phi_c - M phi_c)^T w             (Belrose et al. 2023)

The erasers are FeatureErasureNetwork; their head w is an L2-penalised logistic regression on the
standardised erased features, with the penalty chosen by validation log-loss on a stratified 80/20
split of the training sample and then refit on all of it. Where eta depends on z, the reported
marginal prediction averages over the training-Z distribution,
    y_marg(x_i) = mean_j sigmoid(eta(x_i, z_j)),
as for the refit and the joint model.

Deep feature reweighting (DFR, Kirichenko et al. 2023) retrains the last layer as an l1-penalised
logistic regression on group-balanced subsets (all of the smallest group, an equal draw from every
other) and averages the weights over the subsets. Groups are g = (y, age quintile of h1). Both
variants follow the released dfr_evaluate_spurious.py at its defaults, with base_half's training
half h1 as "train" and the held-out half h2 as "val":

  dfr_val_age  DFR^Val_Tr: C maximises the worst-group accuracy on one random half of h2 after
               fitting on a balanced subset of the other; the head is averaged over 20 balanced
               subsets of h2, on features standardised with the h1 statistics
  dfr_tr_age   DFR^Tr_Tr:  C maximises the worst-group accuracy on h2 after fitting on one
               balanced subset of h1; the head is averaged over 10 balanced subsets of h1

DFR needs no Z at prediction time, so its marginal prediction is its prediction.

The DNN with Controls without a sample split (nosplit_age) refits the last layer of base_full with
Z = age by RefitCovarNetwork on the same full training sample, with the controls of
refit_from_checkpoints.py; its marginal prediction drops f_Z, centred on that sample, from eta,
as for the sample-split and cross-fit rows.

Features are cached per fold in features.npz (base_full) and features_base_half.npz. Writes one row
per (method, coef, fold) to competitor_results.csv in final_v2_refit/.

Usage:  python experiments/ukbb/competitors_from_checkpoints.py --gpu 0 --coefs 0.0,2.0 --folds 0,1,2,3,4
"""
import sys
import csv
import argparse
import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, log_loss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.ukbb.common.backbone import ResNet50
from experiments.ukbb.common.data import (
    RANDOM_STATE, seed_everything, load_ukbb_data, resample_synthetic,
    NumpyCovarDataset, fast_loader, default_transforms,
)
from cocodeel.model import BaseNetwork
from cocodeel.benchmarking.posthoc_model import FeatureErasureNetwork
from cocodeel.refit_model import RefitCovarNetwork
from cocodeel.crossfit import CrossFitEnsemble

# ── run config ────────────────────────────────────────────────────────────────
N_SPLITS = 5
NTRAIN_PER_HALF = 2500
NTEST = 2500
BATCH_SIZE = 48
NUM_WORKERS = 4
C_GRID = np.logspace(-5, 1, 13)    # inverse L2 penalty of the logistic head
DFR_C_GRID = [1.0, 0.7, 0.3, 0.1, 0.07, 0.03, 0.01]   # released dfr_evaluate_spurious.py
N_AGE_BINS = 5
REFIT_MAX_ITERS = 400   # IRLS backfitting controls, as in refit_from_checkpoints.py
REFIT_TOL = 1e-3
Z_COLS = [0]   # Z = age
SRC_RUN = ROOT / "experiments/ukbb/runs/2026-04-26_13-16-37_final_v2"
OUT_RUN = ROOT / "experiments/ukbb/runs/final_v2_refit"
MODEL_PARAMS = dict(backbone=ResNet50, backbone_params={"pretrained_model": ""},
                    num_covariates=0, link="logit")
FIELDS = ["method", "coef", "fold", "auc", "bacc", "auc_marg", "bacc_marg", "corr_age", "corr_sex", "C"]

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
print(f"[{datetime.datetime.now():%H:%M:%S}] device=cuda:{args.gpu}  src={SRC_RUN.name}  out={OUT_RUN.name}", flush=True)


@torch.no_grad()
def features(model, X_, y_, Z_):
    """phi = backbone(x) and base_full's f_X(x) = fx(center_x(phi)) over a sample."""
    ld = fast_loader(NumpyCovarDataset(X_, y_, Z_, tform), batch_size=BATCH_SIZE,
                     shuffle=False, num_workers=NUM_WORKERS)
    phis, fxs = [], []
    for b in ld:
        phi = model.backbone(b["X"].to(device, non_blocking=True))
        phis.append(phi.cpu().numpy())
        fxs.append(model.fx(model.center_x(phi)).cpu().numpy())
    return np.concatenate(phis), np.concatenate(fxs).flatten()


def cached_features(path, ckpt, idx):
    """phi and f_X of the backbone at ckpt on the sample idx and on the balanced test set, cached at path."""
    if path.exists():
        f = np.load(path)
        return f["phi_tr"], f["fx_tr"], f["phi_te"], f["fx_te"], float(f["intercept"])
    model = BaseNetwork(**MODEL_PARAMS).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    phi_tr, fx_tr = features(model, X[idx], y[idx], Z[idx])
    phi_te, fx_te = features(model, X_te, y_te, Z_te)
    intercept = float(model.intercept.item())
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, phi_tr=phi_tr, fx_tr=fx_tr, phi_te=phi_te, fx_te=fx_te, intercept=intercept)
    del model
    torch.cuda.empty_cache()
    return phi_tr, fx_tr, phi_te, fx_te, intercept


def sigmoid(eta):
    return 1 / (1 + np.exp(-eta))


def summary(method, coef, fold, eta, eta_marg, fx, C=float("nan")):
    """AUC/bacc with the test set's own z and marginalised over training Z; corr of f_X with Z."""
    y_reg, y_marg = sigmoid(eta), sigmoid(eta_marg)
    return dict(method=method, coef=coef, fold=fold,
                auc=roc_auc_score(y_te, y_reg), bacc=balanced_accuracy_score(y_te, y_reg >= 0.5),
                auc_marg=roc_auc_score(y_te, y_marg), bacc_marg=balanced_accuracy_score(y_te, y_marg >= 0.5),
                corr_age=np.corrcoef(fx, Z_te[:, 0])[0, 1], corr_sex=np.corrcoef(fx, Z_te[:, 1])[0, 1], C=C)


def marginal_eta(a, c, Zc_tr):
    """eta_marg(x_i) = logit mean_j sigmoid(a_i - zc_j^T c) for eta = a(x) - (z - zbar)^T c."""
    p = sigmoid(a[:, None] - (Zc_tr @ c)[None, :]).mean(1)
    return np.log(p) - np.log1p(-p)


def logistic_head(R, y_tr, seed):
    """L2-penalised logistic regression on standardised R; penalty by validation log-loss, refit on all."""
    tr, va = train_test_split(np.arange(len(y_tr)), test_size=0.2, random_state=seed, stratify=y_tr)
    scaler = StandardScaler().fit(R[tr])
    losses = [log_loss(y_tr[va], LogisticRegression(C=C, max_iter=5000).fit(scaler.transform(R[tr]), y_tr[tr])
                       .predict_proba(scaler.transform(R[va]))[:, 1]) for C in C_GRID]
    C = C_GRID[int(np.argmin(losses))]
    scaler = StandardScaler().fit(R)
    clf = LogisticRegression(C=C, max_iter=5000).fit(scaler.transform(R), y_tr)
    # back to the scale of R: eta = beta0 + R^T w
    w = clf.coef_.flatten() / scaler.scale_
    return clf.intercept_[0] - scaler.mean_ @ w, w, C


def balanced_subset(groups, rng):
    """All of the smallest group and an equally large random draw from every other group."""
    ids, counts = np.unique(groups, return_counts=True)
    return np.concatenate([rng.choice(np.flatnonzero(groups == g), counts.min(), replace=False) for g in ids])


def worst_group_acc(y_true, y_pred, groups):
    return min(np.mean(y_pred[groups == g] == y_true[groups == g]) for g in np.unique(groups))


def dfr_logreg(x, y_, C, max_iter=100):
    return LogisticRegression(penalty="l1", C=C, solver="liblinear", max_iter=max_iter).fit(x, y_)


def averaged_head(fits, scaler):
    """Average the fitted weights, then back to the scale of phi: eta = beta0 + phi^T w."""
    w = np.mean([f.coef_.ravel() for f in fits], axis=0) / scaler.scale_
    return np.mean([f.intercept_[0] for f in fits]) - scaler.mean_ @ w, w


def dfr_val(phi_tr, phi_va, y_va, g_va, rng):
    """DFR^Val_Tr: dfr_on_validation_tune and dfr_on_validation_eval of the released code."""
    perm = rng.permutation(len(y_va))
    tune, fit = perm[:len(y_va) // 2], perm[len(y_va) // 2:]
    fit = fit[balanced_subset(g_va[fit], rng)]
    scaler = StandardScaler().fit(phi_va[fit])
    wga = {C: worst_group_acc(y_va[tune], dfr_logreg(scaler.transform(phi_va[fit]), y_va[fit], C)
                              .predict(scaler.transform(phi_va[tune])), g_va[tune]) for C in DFR_C_GRID}
    C = max(wga, key=wga.get)
    scaler = StandardScaler().fit(phi_tr)
    subsets = [balanced_subset(g_va, rng) for _ in range(20)]
    return (*averaged_head([dfr_logreg(scaler.transform(phi_va[b]), y_va[b], C) for b in subsets], scaler), C)


def dfr_train(phi_tr, y_tr, g_tr, phi_va, y_va, g_va, rng):
    """DFR^Tr_Tr: dfr_train_subset_tune and dfr_train_subset_eval of the released code."""
    scaler = StandardScaler().fit(phi_tr)
    b = balanced_subset(g_tr, rng)
    wga = {C: worst_group_acc(y_va, dfr_logreg(scaler.transform(phi_tr[b]), y_tr[b], C, max_iter=20)
                              .predict(scaler.transform(phi_va)), g_va) for C in DFR_C_GRID}
    C = max(wga, key=wga.get)
    subsets = [balanced_subset(g_tr, rng) for _ in range(10)]
    return (*averaged_head([dfr_logreg(scaler.transform(phi_tr[b]), y_tr[b], C) for b in subsets], scaler), C)


def run_fold(coef, fold, done):
    fold_seed = RANDOM_STATE + fold
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    train_ix = list(cv.split(np.zeros(len(y)), y))[fold][0]
    # partition (mirror train_backbones.py: split pool before resampling)
    pool_A, pool_B = train_test_split(train_ix, test_size=0.5, random_state=fold_seed, stratify=y[train_ix])
    h1_idx = pool_A[resample_synthetic(y[pool_A], Z[pool_A], NTRAIN_PER_HALF, coef, fold_seed)]
    h2_idx = pool_B[resample_synthetic(y[pool_B], Z[pool_B], NTRAIN_PER_HALF, coef, fold_seed + 100)]
    full_idx = np.concatenate([h1_idx, h2_idx])
    y_tr, Z_tr = y[full_idx].astype(int), Z[full_idx].astype(np.float64)

    out_dir = OUT_RUN / f"coef={coef}" / f"fold={fold}"
    src_dir = SRC_RUN / f"coef={coef}" / f"fold={fold}"
    phi_tr, fx_tr, phi_te, fx_te, intercept = cached_features(out_dir / "features.npz",
                                                              src_dir / "base_full.pt", full_idx)

    rows = []
    zbar = Z_tr[:, Z_COLS].mean(0)
    Zc_tr, Zc_te = Z_tr[:, Z_COLS] - zbar, Z_te[:, Z_COLS].astype(np.float64) - zbar

    # posthoc_web: remove the linear Z-component of base_full's f_X
    m = "posthoc_web_age"
    if (m, coef, fold) not in done:
        b = np.linalg.lstsq(Zc_tr, fx_tr.astype(np.float64), rcond=None)[0]
        rows.append(summary(m, coef, fold, intercept + fx_te - Zc_te @ b,
                            marginal_eta(intercept + fx_te, b, Zc_tr), fx_te - Zc_te @ b))

    # feature erasure + tuned logistic head
    for eraser, m in [("residualize", "posthoc_web_features_age"), ("leace", "leace_age")]:
        if (m, coef, fold) in done:
            continue
        net = FeatureErasureNetwork(BaseNetwork(**MODEL_PARAMS), len(Z_COLS), eraser)
        R = net.fit_eraser(torch.tensor(phi_tr), torch.tensor(Z_tr[:, Z_COLS], dtype=torch.float32)).numpy()
        beta0, w, C = logistic_head(R, y_tr, fold_seed + 300)
        phi_te_c = phi_te.astype(np.float64) - net.center_x.mean.double().numpy()
        if eraser == "residualize":
            # f_X = (phi_c - zc^T B)^T w = phi_c^T w - zc^T (B w)
            c = net.B.double().numpy() @ w
            a = phi_te_c @ w
            fx, eta_marg = a - Zc_te @ c, marginal_eta(beta0 + a, c, Zc_tr)
        else:
            fx = (phi_te_c - phi_te_c @ net.M.double().numpy().T) @ w
            eta_marg = beta0 + fx
        rows.append(summary(m, coef, fold, beta0 + fx, eta_marg, fx, C))

    # DFR on base_half, with its training half h1 as "train" and the held-out half h2 as "val"
    if not {("dfr_val_age", coef, fold), ("dfr_tr_age", coef, fold)} <= done:
        phi, _, phi_te_h, _, _ = cached_features(out_dir / "features_base_half.npz",
                                                 src_dir / "base_half.pt", full_idx)
        n1 = len(h1_idx)
        y_h, age_h = y[full_idx].astype(int), Z[full_idx, 0].astype(np.float64)
        g_h = 2 * np.digitize(age_h, np.quantile(age_h[:n1], np.linspace(0, 1, N_AGE_BINS + 1)[1:-1])) + y_h
        rng = np.random.default_rng(fold_seed + 400)
        heads = {"dfr_val_age": dfr_val(phi[:n1], phi[n1:], y_h[n1:], g_h[n1:], rng),
                 "dfr_tr_age": dfr_train(phi[:n1], y_h[:n1], g_h[:n1], phi[n1:], y_h[n1:], g_h[n1:], rng)}
        for m, (beta0, w, C) in heads.items():
            if (m, coef, fold) not in done:
                fx = phi_te_h.astype(np.float64) @ w
                rows.append(summary(m, coef, fold, beta0 + fx, beta0 + fx, fx, C))

    # DNN with Controls, no sample split: base_full's last layer refit with age on full_idx,
    # inner 80/20 split seeded as the sample-split refit in refit_from_checkpoints.py
    m = "nosplit_age"
    if (m, coef, fold) not in done:
        base_full = BaseNetwork(**MODEL_PARAMS).to(device)
        base_full.load_state_dict(torch.load(src_dir / "base_full.pt", map_location=device))
        base_full.eval()
        tr_loc, va_loc = train_test_split(np.arange(len(full_idx)), test_size=0.2,
                                          random_state=fold_seed + 202, stratify=y[full_idx])
        tr, va = [fast_loader(NumpyCovarDataset(X[full_idx[i]], y[full_idx[i]], Z[full_idx[i]][:, Z_COLS], tform),
                              batch_size=BATCH_SIZE, shuffle=sh, num_workers=NUM_WORKERS)
                  for i, sh in [(tr_loc, True), (va_loc, False)]]
        refit = RefitCovarNetwork(base_full, num_covariates=1, orthogonalize=False).to(device)
        ens = CrossFitEnsemble([refit.fit(tr, va, max_iters=REFIT_MAX_ITERS, tol=REFIT_TOL)]).eval()
        te = fast_loader(NumpyCovarDataset(X_te, y_te, Z_te[:, Z_COLS], tform),
                         batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        eta, fz, fx = [], [], []
        with torch.no_grad():
            for b in te:
                x, z = b["X"].to(device, non_blocking=True), b["Z"].to(device, non_blocking=True)
                eta.append(ens.predict_eta(x, z).cpu().numpy())
                fz.append(ens.predict_fz(z).cpu().numpy())
                fx.append(ens.predict_fx(x).cpu().numpy())
        eta, fz, fx = [np.concatenate(a).flatten().astype(np.float64) for a in (eta, fz, fx)]
        rows.append(summary(m, coef, fold, eta, eta - fz, fx))
        del base_full, refit, ens
        torch.cuda.empty_cache()
    return rows


# ── main (append to competitor_results.csv, resumable per method) ─────────────
OUT_CSV = OUT_RUN / "competitor_results.csv"
done = set()
if OUT_CSV.exists():
    done = {(r["method"], float(r["coef"]), int(r["fold"])) for r in csv.DictReader(open(OUT_CSV))}
else:
    with open(OUT_CSV, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()
for coef in COEFS:
    for fold in FOLDS:
        print(f"[{datetime.datetime.now():%H:%M:%S}] ══ coef={coef} fold={fold} ══", flush=True)
        rows = run_fold(coef, fold, done)
        with open(OUT_CSV, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerows(rows)
        for r in rows:
            print(f"    {r['method']:>28}: auc={r['auc']:.4f} auc_marg={r['auc_marg']:.4f} "
                  f"corr_age={r['corr_age']:+.3f} C={r['C']:.1e}", flush=True)
print(f"[{datetime.datetime.now():%H:%M:%S}] Done. {OUT_CSV}", flush=True)
