#!/usr/bin/env python
"""K=3 cross-fit on ADNI with 5-fold outer CV (empirical, no synthetic confounding).

Outer loop is a subject-grouped, AD-stratified 5-fold CV over the entire
1674-sample dataset (no separate train/test split). Each outer fold's 1/5
hold-out is its test set; the other 4/5 is the training pool. Nothing in
the training pipeline touches the outer fold's hold-out, so using it as
the test set is leakage-free.

Per outer fold:
  baseline  = single DNN trained on the 4/5 outer pool, no covariates.
  crossfit  = K=3 cross-fit on the same 4/5 pool:
                StratifiedGroupKFold(3) on outer_pool → h_0, h_1, h_2
                per rotation k:
                  backbone_k ← covar_trainer on h_{(k+1)%3} ∪ h_{(k+2)%3}
                  λ_z_k     ← group-aware LogisticRegressionCV on B(Z[h_k])
                  phm_k     ← PostHocCovarNetwork(backbone_k).fit on h_k
                              with penalty_z = spline.make_penalty(λ_z_k)
                  SURGERY    : recentre center_x, center_z on the 4/5
                              outer pool; intercept absorbs the shift.

Predictions on outer fold's hold-out:
  baseline:        σ(intercept_bl + fx_bl(x))
  crossfit:        σ((1/3) Σ_k η_k(x, z))
  crossfit_marg:   (1/3) Σ_k mean_{j ∈ outer_pool} σ(η_k(x, z_j))

Aggregation:
  metrics_per_fold.csv — long-form (5 outer × 3 methods).
  metrics_summary.csv  — mean ± std across the 5 folds.
  metrics_pooled.csv   — pooled out-of-sample prediction (each subject once),
                          metric computed on the pooled vector.
  fz_age.pdf           — 5 thin gray ensemble fz(age) curves + thick black mean.
  attention_map.pdf    — LRP relevance maps, 3 cross-sections × {baseline,
                          crossfit}, averaged across all 5 outer folds.

Skip-if-exists for backbones, posthocs, baselines, and per-fold relevance
maps. Resume by setting ADNI_K3_RESUME_DIR=<run_dir>.

Usage:
    cd experiments/adni/
    conda run --no-capture-output -n dl-mri python run_adni_k3_crossfit.py
"""
import os
import sys
import gc
import copy
import types
import random
import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

import sklearn
import sklearn.utils.class_weight
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, accuracy_score, log_loss,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ── Project paths ────────────────────────────────────────────────────────────
PROJ = Path("~/Research/proj-orthogonalisation/").expanduser()
sys.path.insert(0, str(PROJ / "submodules/cocodeel"))
sys.path.insert(1, str(PROJ / "submodules/nitorch"))
sys.path.insert(2, str(PROJ / "experiments"))

from nitorch.transforms import IntensityRescale, ToTensor
from cocodeel.model import BaseNetwork
from cocodeel.posthoc_model import PostHocCovarNetwork
from cocodeel.trainer import covar_trainer

from zennit.composites import EpsilonPlus
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.attribution import Gradient


# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_STATE = 45
GPU = 0
K = 3
N_OUTER = 5

DATA_H5 = Path("~/Research/Datasets/proj-orthogonalisation/"
               "adni-screen-SUBJ-SEX-AGE-AD-n1682.h5").expanduser()

_resume = os.environ.get("ADNI_K3_RESUME_DIR")
RUN_DIR = (Path(_resume) if _resume else
           PROJ / "experiments/adni/runs"
                / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_outer5"))

BATCH_SIZE = 64
NUM_WORKERS = 16

trainer_params = {
    "device": f"cuda:{GPU}",
    "loss_fn": nn.BCEWithLogitsLoss(),       # pos_weight set per fold below
    "epochs": 128,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "patience": 32,
    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "scheduler_kwargs": {"patience": 5, "factor": 0.8},
}


# ── Inline helpers (verbatim from notebooks/paper/application) ───────────────
class ADNISixtyFourBackbone(nn.Module):
    """3D-CNN trunk for ADNI sMRI; input (1, 182, 218, 182) or (1, 96, 114, 96)."""
    def __init__(self, out_features=32, drp_rate=0.3, downsampled=False):
        super().__init__()
        self.out_features = out_features
        self.drp_rate = drp_rate
        self.downsampled = downsampled
        if downsampled:
            self.conv_1 = nn.Sequential(
                nn.Dropout3d(p=drp_rate),
                nn.Conv3d(1, 16, kernel_size=5, stride=1, padding=0),
                nn.BatchNorm3d(16), nn.ELU(),
                nn.MaxPool3d(kernel_size=3, stride=3, padding=0))
        else:
            self.conv_1 = nn.Sequential(
                nn.Dropout3d(p=drp_rate),
                nn.Conv3d(1, 16, kernel_size=10, stride=2, padding=0),
                nn.BatchNorm3d(16), nn.ELU(),
                nn.MaxPool3d(kernel_size=3, stride=3, padding=0))
        self.conv_2 = nn.Sequential(
            nn.Dropout3d(p=drp_rate),
            nn.Conv3d(16, 32, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm3d(32), nn.ELU(),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=0))
        self.conv_3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(64), nn.ELU())
        self.conv_4 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(128), nn.ELU())
        self.conv_5 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(64), nn.ELU())
        self.conv_6 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(64), nn.ELU(),
            nn.MaxPool3d(kernel_size=4, stride=2, padding=0))
        self.fc = nn.Sequential(nn.Linear(128, out_features), nn.ELU())

    def forward(self, x):
        for layer in (self.conv_1, self.conv_2, self.conv_3,
                      self.conv_4, self.conv_5, self.conv_6):
            x = layer(x)
        return self.fc(torch.flatten(x, 1))


class numpyCovarDataset(Dataset):
    def __init__(self, X, y, Z, transform=None, covar_transform=None):
        self.X = X
        self.y = y[:, None] if y.ndim == 1 else y
        self.Z = Z[:, None] if Z.ndim == 1 else Z
        self.transform = transform
        self.covar_transform = covar_transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        image = self.X[idx]
        covar = self.Z[idx]
        if self.transform:
            image = self.transform(image)
        if self.covar_transform:
            covar = self.covar_transform(covar)
        if not torch.is_tensor(image):
            image = torch.tensor(self.X[idx], dtype=torch.float32)
        if not torch.is_tensor(covar):
            covar = torch.tensor(covar, dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        return {"X": image, "y": label, "Z": covar}


class ToSplineDesign:
    """Replace covariate `pos` with a B-spline design; keep other covariates."""
    def __init__(self, knots, n_covariates, degree=3, pos=0):
        self.pos = pos
        self.degree = degree
        self.knot_vector = np.concatenate([
            np.repeat(knots[0], degree + 1),
            knots[1:-1],
            np.repeat(knots[-1], degree + 1),
        ])
        self.n_basis = len(self.knot_vector) - degree - 1
        self.n_out = self.n_basis + (n_covariates - 1)

    def __call__(self, covar):
        covar = np.asarray(covar)
        single = covar.ndim == 1
        covar = np.atleast_2d(covar)
        x = covar[:, self.pos]
        # Clip x to the basis support [knot_vector[degree], knot_vector[-degree-1]]
        # so out-of-range points get the boundary basis value, not a silent
        # zero (the previous nan_to_num pattern silently set fz=0 outside
        # support, biasing the marginalisation).
        lo = self.knot_vector[self.degree]
        hi = self.knot_vector[-self.degree - 1]
        x_clipped = np.clip(x, lo, hi)
        B = np.column_stack([
            BSpline.basis_element(
                self.knot_vector[i:i + self.degree + 2], extrapolate=False
            )(x_clipped)
            for i in range(self.n_basis)
        ])
        # Any residual NaN at exact-boundary float-comparison edge cases.
        B = np.nan_to_num(B, nan=0.0)
        remaining = np.delete(covar, self.pos, axis=1)
        result = np.concatenate((B, remaining), axis=1)
        return result.squeeze(0) if single else result

    def make_penalty(self, lam_z):
        """Second-order P-spline penalty on the spline columns; padded with
        zeros for the unpenalised covariate column(s) (sex).

        No Tikhonov regularisation. If the IRLS FWL solve becomes singular
        under saturated weights, that's a real symptom — perfect separation
        in a sub-pool, or a degenerate basis direction — and should not be
        masked numerically.
        """
        D = np.diff(np.eye(self.n_basis), n=2, axis=0)
        P = lam_z * D.T @ D
        n_rest = self.n_out - self.n_basis
        P_full = np.block([
            [P,                                np.zeros((self.n_basis, n_rest))],
            [np.zeros((n_rest, self.n_basis)), np.zeros((n_rest, n_rest))],
        ])
        return torch.tensor(P_full, dtype=torch.float32)


# ── Setup ─────────────────────────────────────────────────────────────────────
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

device = torch.device(f"cuda:{GPU}")
RUN_DIR.mkdir(parents=True, exist_ok=True)
print(f"[setup] run_dir={RUN_DIR}", flush=True)

# Data — whole sample, no train/test split.
data = h5py.File(DATA_H5, "r")
X = data["X"][:]
y = np.float32(data["AD"][:])
Z = np.column_stack((data["AGE"][:], data["SEX"][:])).astype(np.float32)
i_subj = data["i"][:]
print(f"[data] X={X.shape} y={y.shape} Z={Z.shape} "
      f"n_subj={len(np.unique(i_subj))}", flush=True)

# Spline basis is built PER FOLD inside the outer loop, using the quantiles of
# that fold's training pool age distribution. With whole-sample knots, a fold's
# training pool would not necessarily span the boundary regions, and the
# corresponding basis functions would have zero data leverage in the IRLS solve
# (held in place only by the P-spline penalty). Per-fold knots eliminate this.
# Spline.n_out depends only on the (constant) degree and number of knots, so
# the post-hoc model architecture (num_covariates=spline.n_out) is identical
# across folds even though the basis itself is fold-specific.
zmin, zmax = Z[:, 0].min(), Z[:, 0].max()        # for the common figure xgrid

img_transforms = [IntensityRescale(masked=True), ToTensor()]

model_params = {
    "backbone": ADNISixtyFourBackbone,
    "backbone_params": {"out_features": 256, "drp_rate": 0.3, "downsampled": False},
    "num_covariates": 0,
    "link": "logit",
}


# ── Loaders ───────────────────────────────────────────────────────────────────
def make_train_val_loaders(idx_pool, covar_tforms, fold_seed):
    """Group-aware, AD-stratified 80/20 train/val split inside a sub-pool —
    used for backbone early-stopping and posthoc IRLS validation.

    `covar_tforms` is a list of covariate transforms (e.g. [spline_k]) or None
    for backbones (no covariates).
    """
    sgkf_inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=fold_seed)
    tr_loc, va_loc = next(sgkf_inner.split(
        np.zeros(len(idx_pool)), y[idx_pool], groups=i_subj[idx_pool],
    ))
    idx_tr, idx_va = idx_pool[tr_loc], idx_pool[va_loc]
    ct = transforms.Compose(covar_tforms) if covar_tforms else None
    tr = numpyCovarDataset(X[idx_tr], y[idx_tr], Z[idx_tr],
                           transform=transforms.Compose(img_transforms),
                           covar_transform=ct)
    va = numpyCovarDataset(X[idx_va], y[idx_va], Z[idx_va],
                           transform=transforms.Compose(img_transforms),
                           covar_transform=ct)
    return (
        DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True,
                   num_workers=NUM_WORKERS, persistent_workers=True),
        DataLoader(va, batch_size=BATCH_SIZE, shuffle=False,
                   num_workers=NUM_WORKERS, persistent_workers=True),
        y[idx_tr],
    )


def make_full_loader(idx_pool, covar_tforms):
    """shuffle=False loader over the full pool. No persistent workers — these
    loaders are iterated only a handful of times (centering, prediction, LRP),
    and we don't want to keep workers alive for them."""
    ct = transforms.Compose(covar_tforms) if covar_tforms else None
    ds = numpyCovarDataset(X[idx_pool], y[idx_pool], Z[idx_pool],
                           transform=transforms.Compose(img_transforms),
                           covar_transform=ct)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS)


# ── Surgery ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def surgery_recentre(phm, pool_loader_lin, spline_k, Z_pool_raw):
    """Recentre phm.center_x and phm.center_z on the union pool (full outer
    training pool), using rotation-specific spline `spline_k`.

    η(x, z) = b_0 + W_x·(H(x) - μ_h) + W_z·(Z - μ_z)
    Shift to (μ_h', μ_z'):
        b_0' = b_0 + W_x·(μ_h' - μ_h) + W_z·(μ_z' - μ_z)

    After surgery: fx_k and fz_k mean-zero on the full outer pool (under
    spline_k's basis).
    """
    phm.eval()
    Hs = []
    for batch in pool_loader_lin:
        Hs.append(phm.backbone(batch["X"].to(device, non_blocking=True)))
    H_all = torch.cat(Hs, dim=0)
    # Apply rotation's spline to raw outer-pool Z once (cheap; numpy on (n_pool, 2)).
    Z_pool_spl = spline_k(Z_pool_raw)
    Z_all = torch.tensor(Z_pool_spl, dtype=torch.float32, device=device)

    mu_h_new = H_all.mean(dim=0)
    mu_z_new = Z_all.mean(dim=0)
    delta_h = mu_h_new - phm.center_x.mean.detach().clone()
    delta_z = mu_z_new - phm.center_z.mean.detach().clone()

    fx_shift = (phm.fx.weight.squeeze(0) * delta_h).sum()
    fz_shift = (phm.fz.weight.squeeze(0) * delta_z).sum()
    phm.intercept.data += (fx_shift + fz_shift)
    phm.center_x.mean.data.copy_(mu_h_new)
    phm.center_z.mean.data.copy_(mu_z_new)
    return phm


# ── Prediction helpers ────────────────────────────────────────────────────────
@torch.no_grad()
def collect_eta(phm, loader_lin, spline_k):
    """η_i = intercept + fx(x_i) + fz(spline_k(z_i)) on a no-spline loader.

    The loader returns raw Z (age, sex); we apply the rotation's spline_k
    inline so we don't have to spin up a per-rotation spline-aware DataLoader.
    """
    phm.eval()
    out = []
    for batch in loader_lin:
        x = batch["X"].to(device, non_blocking=True)
        z_raw = batch["Z"].numpy()
        z = torch.tensor(spline_k(z_raw), dtype=torch.float32, device=device)
        eta = phm.intercept + phm.predict_fx(x, z=None) + phm.predict_fz(z)
        out.append(eta.cpu().numpy())
    return np.concatenate(out).flatten()


@torch.no_grad()
def collect_p_marginal(phm, loader_lin, spline_k, Z_pool_raw):
    """Probability-scale marginalisation over the *training* Z distribution:

        p_marg_i = (1/n_pool) Σ_j σ( intercept + fx(x_i) + fz(spline_k(z_j)) )

    Z_pool_raw is the raw (age, sex) pool used to define the marginalisation
    distribution; we apply spline_k once to get the spline-design matrix.
    """
    phm.eval()
    Z_pool_spl = spline_k(Z_pool_raw)
    Z_t = torch.tensor(Z_pool_spl, dtype=torch.float32, device=device)
    fz_pool = phm.predict_fz(Z_t).squeeze(-1)            # (n_pool,)
    out = []
    for batch in loader_lin:
        x = batch["X"].to(device, non_blocking=True)
        fx = phm.predict_fx(x, z=None).squeeze(-1)        # (b,)
        eta = phm.intercept + fx[:, None] + fz_pool[None, :]   # (b, n_pool)
        p = torch.sigmoid(eta).mean(dim=1)                 # (b,)
        out.append(p.cpu().numpy())
    return np.concatenate(out).flatten()


@torch.no_grad()
def collect_eta_base(model, loader):
    """For BaseNetwork (num_covariates=0): η = intercept + predict_fx(x)."""
    model.eval()
    out = []
    for batch in loader:
        x = batch["X"].to(device, non_blocking=True)
        eta = model.intercept + model.predict_fx(x)
        out.append(eta.cpu().numpy())
    return np.concatenate(out).flatten()


def deviance(y_true, p_pred):
    return -2.0 * (log_loss(y_true, y_true) - log_loss(y_true, p_pred))


def deviance_explained(y_true, p_pred):
    p_null = np.full_like(y_true, np.mean(y_true))
    return 1.0 - deviance(y_true, p_pred) / deviance(y_true, p_null)


def metric_dict(y_true, p_pred):
    p_clip = np.clip(p_pred, 1e-7, 1 - 1e-7)
    return {
        "auc":           roc_auc_score(y_true, p_clip),
        "acc":           accuracy_score(y_true, (p_clip >= 0.5).astype(int)),
        "bacc":          balanced_accuracy_score(y_true, (p_clip >= 0.5).astype(int)),
        "dev_explained": deviance_explained(y_true, p_clip),
    }


# ── LRP setup ─────────────────────────────────────────────────────────────────
canonizers = [SequentialMergeBatchNorm()]
composite = EpsilonPlus(canonizers=canonizers)


def lrp_forward_with_z(self, x):
    return self.predict_fx(x, z=self.Z)


def lrp_forward_no_z(self, x):
    return self.predict_fx(x)


def lrp_relevance(model, test_ds_lin, spline_k=None):
    """Mean |relevance| over AD-positive test subjects. Returns (D, H, W).

    test_ds_lin returns raw Z (age, sex). For posthoc models, pass a
    per-rotation `spline_k` and we apply it inline before setting `model.Z`.
    For the no-covariate baseline, pass `spline_k=None`.
    """
    rel_sum = torch.zeros_like(test_ds_lin[0]["X"]).squeeze()
    n = 0
    for j in range(len(test_ds_lin)):
        if test_ds_lin[j]["y"].item() == 0:
            continue
        Xin = test_ds_lin[j]["X"].unsqueeze(0).to(device)
        if spline_k is not None:
            z_raw = test_ds_lin[j]["Z"].numpy()
            Zin = torch.tensor(spline_k(z_raw),
                               dtype=torch.float32, device=device).unsqueeze(0)
            model.Z = Zin
            model.Z.requires_grad = True
        with Gradient(model=model, composite=composite) as attributor:
            _, rel = attributor(Xin)
        rel_sum += rel.abs().cpu().squeeze()
        n += 1
    return (rel_sum / max(n, 1)).numpy()


# ── Outer 5-fold CV ───────────────────────────────────────────────────────────
outer_cv = StratifiedGroupKFold(n_splits=N_OUTER, shuffle=True, random_state=RANDOM_STATE)
outer_splits = list(outer_cv.split(np.zeros(len(y)), y, groups=i_subj))

fold_results = []          # long-form per-fold metric rows
pred_pooled = {            # for pooled out-of-sample metric
    "baseline":         {"y_true": [], "y_pred": []},
    "crossfit_k3":      {"y_true": [], "y_pred": []},
    "crossfit_k3_marg": {"y_true": [], "y_pred": []},
}
fz_curves_per_fold = []
lrp_baseline_per_fold = []
lrp_crossfit_per_fold = []

xgrid = np.linspace(zmin, zmax, 300)


for f in range(N_OUTER):
    pool_idx, test_idx = outer_splits[f]
    fold_dir = RUN_DIR / f"outer{f}"
    k3_dir = fold_dir / "k3"
    fold_dir.mkdir(parents=True, exist_ok=True)
    k3_dir.mkdir(parents=True, exist_ok=True)
    fold_seed = RANDOM_STATE + 1000 * f

    print(f"\n══ outer fold {f}: pool n={len(pool_idx)} "
          f"({len(np.unique(i_subj[pool_idx]))} subj)  "
          f"test n={len(test_idx)} ({len(np.unique(i_subj[test_idx]))} subj) ══",
          flush=True)

    # Per-fold spline: knots from the OUTER POOL quantiles. All K=3 rotations
    # within this outer fold share the same basis. (Per-rotation knots were
    # tried and rejected — they break the cross-fit interpretation by
    # estimating different sieve projections per rotation, and they introduce
    # silent NaN-zeroing when the marginalisation evaluates spline_k on points
    # outside h_k's range.)
    knots_f = np.quantile(Z[pool_idx, 0], np.linspace(0, 1, 5))
    spline_f = ToSplineDesign(knots=knots_f, n_covariates=Z.shape[1], pos=0)
    covar_tforms_f = [spline_f]
    print(f"[outer{f}] outer-pool knots = "
          f"{np.array2string(knots_f, precision=2)}", flush=True)

    # Loaders for this fold (linear / no-covar-transform only — spline is
    # applied inline by the prediction helpers, not via the DataLoader).
    pool_loader_lin = make_full_loader(pool_idx, covar_tforms=None)
    test_loader_lin = make_full_loader(test_idx, covar_tforms=None)
    test_ds_lin = numpyCovarDataset(X[test_idx], y[test_idx], Z[test_idx],
                                    transform=transforms.Compose(img_transforms))

    # ── Baseline DNN on the full outer pool ──────────────────────────────────
    bl_path = fold_dir / "baseline.pt"
    bl_tr, bl_va, y_bl_tr = make_train_val_loaders(
        pool_idx, covar_tforms=None, fold_seed=fold_seed + 1,
    )
    if bl_path.exists():
        print(f"[outer{f}] baseline exists — loading.", flush=True)
        baseline = BaseNetwork(**model_params).to(device)
        baseline.load_state_dict(torch.load(bl_path, map_location=device))
        baseline = baseline.center_effects(bl_tr)
        baseline.link = "logit"
        baseline.eval()
    else:
        print(f"[outer{f}] training baseline on n={len(pool_idx)} ...", flush=True)
        cw = sklearn.utils.class_weight.compute_class_weight(
            "balanced", classes=np.unique(y_bl_tr), y=y_bl_tr)
        pw = torch.tensor(cw[1] / cw[0]).to(device)
        tp = {**trainer_params, "loss_fn": nn.BCEWithLogitsLoss(pos_weight=pw)}
        baseline = covar_trainer(
            model=BaseNetwork,
            model_params={**model_params, "link": "identity"},
            train_loader=bl_tr, val_loader=bl_va, **tp,
        ).center_effects(bl_tr)
        baseline.link = "logit"
        torch.save(baseline.state_dict(), bl_path)
        print(f"[outer{f}] baseline saved → {bl_path}", flush=True)
    del bl_tr, bl_va; gc.collect()

    # ── K=3 cross-fit on the outer pool ──────────────────────────────────────
    inner_cv = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=fold_seed)
    sub_pools = [pool_idx[idx] for _, idx in inner_cv.split(
        np.zeros(len(pool_idx)), y[pool_idx], groups=i_subj[pool_idx],
    )]
    print(f"[outer{f}] sub-pool sizes: {[len(p) for p in sub_pools]}", flush=True)

    phms = []
    lam_zs = []
    for k in range(K):
        dnn_idx = np.concatenate([sub_pools[(k + 1) % K], sub_pools[(k + 2) % K]])
        post_idx = sub_pools[k]

        # Backbone on dnn_idx.
        bb_path = k3_dir / f"backbone_k{k}.pt"
        bb_tr, bb_va, y_bb_tr = make_train_val_loaders(
            dnn_idx, covar_tforms=None, fold_seed=fold_seed + 1000 + k,
        )
        if bb_path.exists():
            print(f"[outer{f}/k={k}] backbone exists — loading.", flush=True)
            backbone = BaseNetwork(**model_params).to(device)
            backbone.load_state_dict(torch.load(bb_path, map_location=device))
            backbone = backbone.center_effects(bb_tr)
            backbone.link = "logit"
            backbone.eval()
        else:
            print(f"[outer{f}/k={k}] training backbone on n={len(dnn_idx)} ...",
                  flush=True)
            cw = sklearn.utils.class_weight.compute_class_weight(
                "balanced", classes=np.unique(y_bb_tr), y=y_bb_tr)
            pw = torch.tensor(cw[1] / cw[0]).to(device)
            tp = {**trainer_params, "loss_fn": nn.BCEWithLogitsLoss(pos_weight=pw)}
            backbone = covar_trainer(
                model=BaseNetwork,
                model_params={**model_params, "link": "identity"},
                train_loader=bb_tr, val_loader=bb_va, **tp,
            ).center_effects(bb_tr)
            backbone.link = "logit"
            torch.save(backbone.state_dict(), bb_path)
            print(f"[outer{f}/k={k}] backbone saved → {bb_path}", flush=True)
        del bb_tr, bb_va; gc.collect()

        # λ_z via group-aware CV on the (outer-fold) spline design.
        B_post = spline_f(Z[post_idx])
        gkf = GroupKFold(n_splits=5)
        gkf_splits = list(gkf.split(B_post, y[post_idx], groups=i_subj[post_idx]))
        lr_cv = LogisticRegressionCV(
            Cs=np.logspace(-5, 3, 50), fit_intercept=True,
            penalty="l2", cv=gkf_splits,
        )
        lr_cv.fit(B_post, y[post_idx])
        lam_z = 1.0 / lr_cv.C_[0]
        lam_zs.append(lam_z)
        P_z = spline_f.make_penalty(lam_z).to(device)
        print(f"[outer{f}/k={k}] lam_z = {lam_z:.4e}", flush=True)

        # Posthoc on post_idx.
        ph_path = k3_dir / f"posthoc_k{k}.pt"
        ph_tr, ph_va, _ = make_train_val_loaders(
            post_idx, covar_tforms=covar_tforms_f, fold_seed=fold_seed + 2000 + k,
        )
        phm = PostHocCovarNetwork(backbone, num_covariates=spline_f.n_out,
                                   orthogonalize=False).to(device)
        if ph_path.exists():
            print(f"[outer{f}/k={k}] posthoc exists — loading.", flush=True)
            phm.load_state_dict(torch.load(ph_path, map_location=device))
            phm.eval()
        else:
            print(f"[outer{f}/k={k}] fitting posthoc on n={len(post_idx)} ...",
                  flush=True)
            phm = phm.fit(ph_tr, ph_va, penalty_z=P_z, max_iters=400)
            torch.save(phm.state_dict(), ph_path)
            print(f"[outer{f}/k={k}] posthoc saved → {ph_path}", flush=True)

        # Surgery: recentre on the full outer pool.
        phm = surgery_recentre(phm, pool_loader_lin, spline_f, Z[pool_idx])
        phms.append(phm)
        del ph_tr, ph_va; gc.collect()

    # Surgery sanity check on the outer pool. Each rotation has its own spline,
    # so apply spline_k to raw Z[pool_idx] for the fz mean.
    @torch.no_grad()
    def _mean_components(phm, pool_loader_lin, spline_k, Z_pool_raw):
        fxs = []
        for batch in pool_loader_lin:
            xb = batch["X"].to(device, non_blocking=True)
            fxs.append(phm.predict_fx(xb, z=None).cpu().numpy())
        Z_pool_spl = torch.tensor(spline_k(Z_pool_raw),
                                  dtype=torch.float32, device=device)
        fzs = phm.predict_fz(Z_pool_spl).cpu().numpy()
        return float(np.concatenate(fxs).mean()), float(fzs.mean())

    print(f"[outer{f}] surgery-check (mean fx, mean fz on outer pool):",
          flush=True)
    for k, p in enumerate(phms):
        mfx, mfz = _mean_components(p, pool_loader_lin, spline_f, Z[pool_idx])
        print(f"  k={k}: mean(fx)={mfx:+.3e}  mean(fz)={mfz:+.3e}", flush=True)

    # ── Predictions on outer hold-out ────────────────────────────────────────
    y_te = y[test_idx]

    eta_base = collect_eta_base(baseline, test_loader_lin)
    p_base = 1.0 / (1.0 + np.exp(-eta_base))

    etas_cf = [collect_eta(phms[k], test_loader_lin, spline_f) for k in range(K)]
    eta_cf = np.mean(etas_cf, axis=0)
    p_cf = 1.0 / (1.0 + np.exp(-eta_cf))

    # Each rotation marginalises over the full outer pool, under the
    # (shared) outer-fold spline.
    ps_marg = [collect_p_marginal(phms[k], test_loader_lin, spline_f, Z[pool_idx])
               for k in range(K)]
    p_cf_marg = np.mean(ps_marg, axis=0)

    np.savez(fold_dir / "predictions.npz",
             test_idx=test_idx, y_true=y_te,
             p_base=p_base, p_cf=p_cf, p_cf_marg=p_cf_marg)

    for name, p_pred in [("baseline", p_base),
                         ("crossfit_k3", p_cf),
                         ("crossfit_k3_marg", p_cf_marg)]:
        m = metric_dict(y_te, p_pred)
        m.update({"outer_fold": f, "method": name})
        fold_results.append(m)
        pred_pooled[name]["y_true"].append(y_te)
        pred_pooled[name]["y_pred"].append(np.clip(p_pred, 1e-7, 1 - 1e-7))
        print(f"[outer{f}] {name:>20}: auc={m['auc']:.4f}  bacc={m['bacc']:.4f}  "
              f"acc={m['acc']:.4f}  dev={m['dev_explained']:.4f}", flush=True)

    # ── fz(age) ensemble curve for this fold ─────────────────────────────────
    # All rotations share spline_f. Evaluate per rotation on the common
    # (whole-sample) xgrid, mean across rotations, mask outside the outer
    # fold's basis support [knots_f[0], knots_f[-1]].
    sex_mean = float(phms[0].center_z.mean[-1].item())   # identical across k post-surgery
    Zg = spline_f(np.column_stack([xgrid, np.full_like(xgrid, sex_mean)]))
    Zg_t = torch.tensor(Zg, dtype=torch.float32, device=device)
    with torch.no_grad():
        fz_per_k = np.stack([
            p.predict_fz(Zg_t).cpu().numpy().flatten() for p in phms
        ])
    fz_ens = fz_per_k.mean(axis=0)
    in_support = (xgrid >= knots_f[0]) & (xgrid <= knots_f[-1])
    fz_ens = np.where(in_support, fz_ens, np.nan)
    fz_curves_per_fold.append(fz_ens)
    np.save(fold_dir / "fz_ens.npy", fz_ens)

    # ── LRP for this fold ────────────────────────────────────────────────────
    rel_bl_path = fold_dir / "lrp_baseline.npy"
    if rel_bl_path.exists():
        print(f"[outer{f}] LRP baseline cached.", flush=True)
        rel_bl = np.load(rel_bl_path)
    else:
        print(f"[outer{f}] computing LRP baseline ...", flush=True)
        m = copy.deepcopy(baseline)
        m.forward = types.MethodType(lrp_forward_no_z, m)
        m = m.to(device).eval()
        rel_bl = lrp_relevance(m, test_ds_lin, spline_k=None)
        del m
        torch.cuda.empty_cache(); gc.collect()
        np.save(rel_bl_path, rel_bl)
    lrp_baseline_per_fold.append(rel_bl)

    rel_cf_path = fold_dir / "lrp_crossfit.npy"
    if rel_cf_path.exists():
        print(f"[outer{f}] LRP crossfit cached.", flush=True)
        rel_cf = np.load(rel_cf_path)
    else:
        print(f"[outer{f}] computing LRP crossfit (3 rotations) ...", flush=True)
        rel_per_k = []
        for k, phm in enumerate(phms):
            m = copy.deepcopy(phm)
            m.forward = types.MethodType(lrp_forward_with_z, m)
            m = m.to(device).eval()
            rel_per_k.append(lrp_relevance(m, test_ds_lin, spline_k=spline_f))
            del m
            torch.cuda.empty_cache(); gc.collect()
        rel_cf = np.mean(np.stack(rel_per_k), axis=0)
        np.save(rel_cf_path, rel_cf)
    lrp_crossfit_per_fold.append(rel_cf)

    del baseline, phms, etas_cf, ps_marg
    torch.cuda.empty_cache(); gc.collect()


# ── Aggregation ───────────────────────────────────────────────────────────────
df_fold = pd.DataFrame(fold_results)
df_fold = df_fold[["outer_fold", "method", "auc", "acc", "bacc", "dev_explained"]]
df_fold.to_csv(RUN_DIR / "metrics_per_fold.csv", index=False)

df_summary = df_fold.groupby("method").agg(
    auc_mean=("auc", "mean"), auc_std=("auc", "std"),
    acc_mean=("acc", "mean"), acc_std=("acc", "std"),
    bacc_mean=("bacc", "mean"), bacc_std=("bacc", "std"),
    dev_mean=("dev_explained", "mean"), dev_std=("dev_explained", "std"),
).reset_index()
df_summary.to_csv(RUN_DIR / "metrics_summary.csv", index=False)


def pooled_row(method):
    yt = np.concatenate(pred_pooled[method]["y_true"])
    yp = np.concatenate(pred_pooled[method]["y_pred"])
    m = metric_dict(yt, yp)
    m["method"] = method
    return m

df_pooled = pd.DataFrame([pooled_row(m) for m in pred_pooled])
df_pooled = df_pooled[["method", "auc", "acc", "bacc", "dev_explained"]]
df_pooled.to_csv(RUN_DIR / "metrics_pooled.csv", index=False)

print("\n[metrics per fold]")
print(df_fold.to_string(index=False))
print("\n[metrics summary (mean ± std across 5 outer folds)]")
print(df_summary.to_string(index=False))
print("\n[metrics pooled (each subject once)]")
print(df_pooled.to_string(index=False))


# ── fz figure ─────────────────────────────────────────────────────────────────
fz_curves_per_fold = np.stack(fz_curves_per_fold)   # (N_OUTER, 300), with NaNs outside fold support
mean_curve = np.nanmean(fz_curves_per_fold, axis=0)
# Mean is undefined where fewer than 2 folds contribute — drop to NaN there.
support_count = np.sum(~np.isnan(fz_curves_per_fold), axis=0)
mean_curve = np.where(support_count >= 2, mean_curve, np.nan)

fig, ax = plt.subplots(figsize=(6, 4))
for f in range(N_OUTER):
    ax.plot(xgrid, fz_curves_per_fold[f], color="gray", alpha=0.55, linewidth=1.0,
            label="outer-fold ensembles" if f == 0 else None)
ax.plot(xgrid, mean_curve, color="black", linewidth=2.4, label="CV mean")
ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax.set_xlabel("Age")
ax.set_ylabel(r"$f_z$(age)")
ax.set_title("ADNI K=3 cross-fit, 5-fold outer CV: spline age effect")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(RUN_DIR / "fz_age.pdf")
plt.close(fig)
print(f"[figure] wrote {RUN_DIR / 'fz_age.pdf'}", flush=True)


# ── Attention map figure ─────────────────────────────────────────────────────
rel_bl_avg = np.mean(np.stack(lrp_baseline_per_fold), axis=0)
rel_cf_avg = np.mean(np.stack(lrp_crossfit_per_fold), axis=0)


def _normalise(arr):
    return arr / arr.sum()


rel_bl_n = _normalise(rel_bl_avg)
rel_cf_n = _normalise(rel_cf_avg)
vmax = max(rel_bl_n.max(), rel_cf_n.max())
rel_bl_n = rel_bl_n / vmax
rel_cf_n = rel_cf_n / vmax

cuts = (rel_bl_n.shape[0] // 2,
        rel_bl_n.shape[1] // 2,
        rel_bl_n.shape[2] // 2)
panels = [("baseline DNN  (5-fold mean)", rel_bl_n),
          ("crossfit K=3  (5-fold × 3-rot mean)", rel_cf_n)]

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8, 12))
for col, (title, arr) in enumerate(panels):
    axes[0, col].imshow(np.rot90(arr[cuts[0], :, :]), cmap="coolwarm",
                        vmin=0, vmax=1, aspect="equal")
    axes[1, col].imshow(np.rot90(arr[:, cuts[1], :]), cmap="coolwarm",
                        vmin=0, vmax=1, aspect="equal")
    axes[2, col].imshow(np.rot90(arr[:, :, cuts[2]]), cmap="coolwarm",
                        vmin=0, vmax=1, aspect="equal")
    axes[0, col].set_title(title)
for r, lbl in enumerate(["sagittal", "coronal", "axial"]):
    axes[r, 0].set_ylabel(lbl)
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("ADNI fx attention (LRP EpsilonPlus, |rel|, AD-positive subjects)",
             fontsize=11)
fig.tight_layout()
fig.savefig(RUN_DIR / "attention_map.pdf")
plt.close(fig)
print(f"[figure] wrote {RUN_DIR / 'attention_map.pdf'}", flush=True)


print(f"\n[done] all outputs under {RUN_DIR}", flush=True)
