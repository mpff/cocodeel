#!/usr/bin/env python
"""LRP-EpsilonPlus cohort-mean attribution maps for UKBB models.

Reproduces the recipe used in the archived notebook
`experiments/_archive/notebooks/UKKBB_HighalcAgeSex_Synthetic.ipynb` (cells
99-106), updated for the current 4-model panel:

    base_bal, base_conf, refit (sample-split), crossfit (cross-fit ensemble).

Pipeline (per model m):
    1. Per fold f:
        - Compute LRP-EpsilonPlus relevance r_{f,i} for each cohort subject
          i ∈ S (the n=100 subjects with sex==1 ∧ y==1).
        - Mean across subjects, signed:  m_{f} = (1/|S|) Σ_i r_{f,i}.
    2. Per voxel:
          R_v = mean_f |m_{f,v}|
       i.e. take |.| of each fold-mean map, then average across folds.

Cross-fit handling: LRP isn't path-linear in arbitrary additive output
combinations, so the 0.5(fx_A + fx_B) wrapper used for IG isn't safe here.
Instead we run LRP separately on the A-side and B-side post-hoc models and
average their per-subject relevance maps before fold aggregation:
    r_{f,i}^{(crossfit)} = 0.5 (r_{f,i}^{A} + r_{f,i}^{B})

Composite + canonizer follow the old notebook:
    - base models      : EpsilonPlus(canonizers=[ResNetCanonizer()])
    - post-hoc / cross : EpsilonPlus(canonizers=[SequentialMergeBatchNorm()])
      (post-hoc forward is `predict_fx(x, z=None)` — a bare backbone +
      learned head, so the ResNet residual structure has been collapsed by
      the post-hoc head; ResNetCanonizer would re-canonicalise and
      typically agrees here, but SequentialMergeBatchNorm matches the old
      notebook for the post-hoc case.)

Outputs into <run_dir>/lrp/:
    <method>_lrp.nii.gz                — R_v   (final |.|-then-mean map)
    <method>_lrp_{sagittal,coronal,axial}.csv  — central slices

Cost: ~30 min for 4 methods × 5 folds × 100 subjects on a single A100.
"""
import os
import sys
import gc
import argparse
import numpy as np
import torch
import nibabel as nib
from zennit.attribution import Gradient
from zennit.composites import EpsilonPlus
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.torchvision import ResNetCanonizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ukbb_common import (
    seed_everything, RANDOM_STATE,
    load_ukbb_data, resample_synthetic, NumpyCovarDataset,
    default_model_params, default_transforms,
)
from cocodeel.model import BaseNetwork
from cocodeel.posthoc_model import PostHocCovarNetwork

# ── Config ────────────────────────────────────────────────────────────────────
N_SUBJECTS = 100
NTEST      = 2500
GPU        = 1
N_FOLDS    = 5
BATCH_SIZE = 4
RUN_DIR = (
    "/home/RDC/pfeuffma/Research/proj-orthogonalisation/"
    "experiments/ukbb/runs/2026-04-26_13-16-37_final_v2/"
)


class PostHocImageOnly(torch.nn.Module):
    def __init__(self, posthoc_model):
        super().__init__()
        self.model = posthoc_model

    def forward(self, x):
        return self.model.predict_fx(x, z=None)


def load_base(coef, fold, model_params, device):
    ckpt = RUN_DIR + f"coef={coef}/fold={fold}/base_full.pt"
    net = BaseNetwork(**model_params).to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()
    return net


def load_posthoc_age(coef, fold, model_params, device):
    backbone_ckpt = RUN_DIR + f"coef={coef}/fold={fold}/base_half.pt"
    posthoc_ckpt  = RUN_DIR + f"coef={coef}/fold={fold}/posthoc_age.pt"
    base_half = BaseNetwork(**model_params).to(device)
    base_half.load_state_dict(torch.load(backbone_ckpt, map_location=device))
    phm = PostHocCovarNetwork(base_half, num_covariates=1, orthogonalize=False).to(device)
    phm.load_state_dict(torch.load(posthoc_ckpt, map_location=device))
    phm.eval()
    return phm


def load_crossfit_pair(coef, fold, model_params, device):
    base_A = BaseNetwork(**model_params).to(device)
    base_A.load_state_dict(torch.load(
        RUN_DIR + f"coef={coef}/fold={fold}/base_half.pt", map_location=device))
    phm_A = PostHocCovarNetwork(base_A, num_covariates=1, orthogonalize=False).to(device)
    phm_A.load_state_dict(torch.load(
        RUN_DIR + f"coef={coef}/fold={fold}/posthoc_age.pt", map_location=device))
    phm_A.eval()

    base_B = BaseNetwork(**model_params).to(device)
    base_B.load_state_dict(torch.load(
        RUN_DIR + f"coef={coef}/fold={fold}/base_half_B.pt", map_location=device))
    phm_B = PostHocCovarNetwork(base_B, num_covariates=1, orthogonalize=False).to(device)
    phm_B.load_state_dict(torch.load(
        RUN_DIR + f"coef={coef}/fold={fold}/posthoc_age_BA.pt", map_location=device))
    phm_B.eval()
    return phm_A, phm_B


def lrp_subject_mean(network, dataset, filt_ix, composite, device):
    """Run LRP-EpsilonPlus over the cohort and return the SIGNED mean
    relevance across subjects.  Shape: (D, H, W)."""
    subset = torch.utils.data.Subset(dataset, filt_ix)
    loader = torch.utils.data.DataLoader(
        subset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    accum = None
    n_seen = 0
    for batch in loader:
        x = batch["X"].to(device, non_blocking=True)
        x.requires_grad = True
        with Gradient(model=network, composite=composite) as attributor:
            _, rel = attributor(x)
        rel_np = rel.detach().cpu().numpy().squeeze(1)   # (B, D, H, W)
        if accum is None:
            accum = np.zeros(rel_np.shape[1:], dtype=np.float64)
        accum += rel_np.sum(axis=0)
        n_seen += rel_np.shape[0]
    return (accum / max(n_seen, 1)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=GPU)
    parser.add_argument("--folds", type=int, nargs="+",
                        default=list(range(N_FOLDS)))
    parser.add_argument("--age-stratum", choices=["all", "young", "old"],
                        default="all")
    parser.add_argument("--out-suffix", default="",
                        help="Append to lrp output dirname, e.g. '_young'.")
    parser.add_argument("--models", nargs="+",
                        choices=["base_bal", "base_conf", "refit", "crossfit"],
                        default=["base_bal", "base_conf", "refit", "crossfit"])
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    seed_everything(RANDOM_STATE)

    print("Loading data ...", flush=True)
    d = load_ukbb_data()
    X_test, y_test, Z_test = d["X_test"], d["y_test"], d["Z_full_test"]
    idx = resample_synthetic(y_test, Z_test, NTEST, 0.0, RANDOM_STATE)
    X_te, y_te, Z_te = X_test[idx], y_test[idx], Z_test[idx]

    tform = default_transforms()
    test_ds = NumpyCovarDataset(X_te, y_te, Z_te, tform)
    sex_col = Z_te[:, 1]
    age_col = Z_te[:, 0]
    base_filt = [
        i for i in range(len(test_ds))
        if int(sex_col[i]) == 1 and int(y_te[i]) == 1
    ][:N_SUBJECTS]
    if args.age_stratum == "all":
        filt_ix = base_filt
    else:
        ages = age_col[base_filt]
        median_age = float(np.median(ages))
        if args.age_stratum == "young":
            filt_ix = [base_filt[i] for i in range(len(base_filt))
                       if ages[i] < median_age]
        else:
            filt_ix = [base_filt[i] for i in range(len(base_filt))
                       if ages[i] >= median_age]
        print(f"age median split @ {median_age:.1f}: stratum={args.age_stratum}",
              flush=True)
    print(f"cohort: {len(filt_ix)} subjects", flush=True)

    # Composites: ResNetCanonizer for the base DNN, BatchNorm-merge for the
    # post-hoc forward (mirrors the old notebook).
    comp_base    = EpsilonPlus(canonizers=[ResNetCanonizer()])
    comp_posthoc = EpsilonPlus(canonizers=[SequentialMergeBatchNorm()])

    model_params = default_model_params()
    fold_means = {m: [] for m in args.models}   # list of per-fold (D,H,W) maps

    for fold in args.folds:
        print(f"\n=== fold {fold} ===", flush=True)
        for mname in args.models:
            print(f":: {mname} fold {fold}", flush=True)
            if mname == "base_bal":
                net = load_base(0.0, fold, model_params, device)
                m_f = lrp_subject_mean(net, test_ds, filt_ix, comp_base, device)
            elif mname == "base_conf":
                net = load_base(2.0, fold, model_params, device)
                m_f = lrp_subject_mean(net, test_ds, filt_ix, comp_base, device)
            elif mname == "refit":
                phm = load_posthoc_age(2.0, fold, model_params, device)
                net = PostHocImageOnly(phm)
                m_f = lrp_subject_mean(net, test_ds, filt_ix,
                                       comp_posthoc, device)
            elif mname == "crossfit":
                phm_A, phm_B = load_crossfit_pair(2.0, fold, model_params, device)
                net_A = PostHocImageOnly(phm_A)
                net_B = PostHocImageOnly(phm_B)
                # LRP separately on A and B; average the signed per-fold mean.
                m_A = lrp_subject_mean(net_A, test_ds, filt_ix,
                                       comp_posthoc, device)
                m_B = lrp_subject_mean(net_B, test_ds, filt_ix,
                                       comp_posthoc, device)
                m_f = 0.5 * (m_A + m_B)
                # Tag for the cleanup below.
                net = (net_A, net_B)
            print(f":: :: shape={m_f.shape} "
                  f"min={m_f.min():.3e} max={m_f.max():.3e}", flush=True)
            fold_means[mname].append(m_f)
            del net
            torch.cuda.empty_cache()
            gc.collect()

    # Fold aggregation: |.|-then-mean over folds (matches old notebook).
    print(f"\nAggregating |.|-then-mean across {len(args.folds)} folds ...",
          flush=True)
    out_dir = RUN_DIR + f"lrp{args.out_suffix}/"
    os.makedirs(out_dir, exist_ok=True)
    print(f"output dir: {out_dir}", flush=True)
    affine = np.eye(4)
    for mname, maps in fold_means.items():
        stacked = np.stack(maps, axis=0)                # (F, D, H, W)
        R = np.abs(stacked).mean(axis=0).astype(np.float32)
        print(f"[{mname}] R_v: min={R.min():.3e} max={R.max():.3e} "
              f"mean={R.mean():.3e}", flush=True)
        nib.save(nib.Nifti1Image(R, affine),
                 out_dir + f"{mname}_lrp.nii.gz")
        mx, my, mz = R.shape[0]//2, R.shape[1]//2, R.shape[2]//2
        for orient, sl in [
            ("sagittal", np.rot90(R[mx, :, :])),
            ("coronal",  np.rot90(R[:, my, :])),
            ("axial",    np.rot90(R[:, :, mz])),
        ]:
            np.savetxt(out_dir + f"{mname}_lrp_{orient}.csv",
                       sl, delimiter=",", fmt="%.6e")
        print(f"  wrote {mname}_lrp.nii.gz + 3 slice CSVs", flush=True)


if __name__ == "__main__":
    main()
