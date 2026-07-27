#!/usr/bin/env python
"""Per-voxel χ² statistic and (FDR-adjusted) p-value maps from fold-averaged IG attributions for base_bal, base_conf, refit, and crossfit."""
# crossfit uses CrossFitRefitImageOnly = ½(predict_fx_A + predict_fx_B); IG is
# path-linear, so the IG of the average equals the average of the per-side IGs.
import os
import sys
import gc
import argparse
import numpy as np
import torch
import nibabel as nib
from scipy.stats import chi2
from scipy.ndimage import gaussian_filter
from zennit.attribution import IntegratedGradients

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from experiments.ukbb.common.data import (
    seed_everything, RANDOM_STATE,
    load_ukbb_data, resample_synthetic, NumpyCovarDataset,
    default_transforms,
)
from experiments.ukbb.common.backbone import default_model_params
from cocodeel.model import BaseNetwork
from cocodeel.refit_model import RefitCovarNetwork

# ── Config ────────────────────────────────────────────────────────────────────
N_SUBJECTS    = 100
NTEST         = 2500
GPU           = 1
N_FOLDS       = 5
BATCH_SIZE    = 4
IG_N_ITER     = 20
SMOOTH_SIGMA  = 1.0    # voxels; set to 0 to disable per-subject smoothing
MAD_TO_SIGMA  = 1.4826
EPS_SIGMA     = 1e-12  # guard against degenerate σ̂
RUN_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "2026-04-26_13-16-37_final_v2", "")


class RefitImageOnly(torch.nn.Module):
    def __init__(self, refit_model):
        super().__init__()
        self.model = refit_model

    def forward(self, x):
        return self.model.predict_fx(x, z=None)


class CrossFitRefitImageOnly(torch.nn.Module):
    """Cross-fit ensemble: ½(predict_fx_A + predict_fx_B).

    IG of an average equals the average of IGs (IG is path-linear in the
    network's output), so this wrapper produces principled ensemble attributions.
    """
    def __init__(self, refit_A, refit_B):
        super().__init__()
        self.A = refit_A
        self.B = refit_B

    def forward(self, x):
        return 0.5 * (self.A.predict_fx(x, z=None) + self.B.predict_fx(x, z=None))


def load_base(coef, fold, model_params, device):
    ckpt = RUN_DIR + f"coef={coef}/fold={fold}/base_full.pt"
    net = BaseNetwork(**model_params).to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()
    return net


def load_refit_age(coef, fold, model_params, device):
    backbone_ckpt = RUN_DIR + f"coef={coef}/fold={fold}/base_half.pt"
    refit_ckpt  = RUN_DIR + f"coef={coef}/fold={fold}/refit_age.pt"
    base_half = BaseNetwork(**model_params).to(device)
    base_half.load_state_dict(torch.load(backbone_ckpt, map_location=device))
    phm = RefitCovarNetwork(base_half, num_covariates=1, orthogonalize=False).to(device)
    phm.load_state_dict(torch.load(refit_ckpt, map_location=device))
    phm.eval()
    return phm


def load_crossfit_age(coef, fold, model_params, device):
    """Load the A-side (base_half + refit_age) and B-side (base_half_B + refit_age_BA) refit models for ensembling."""
    base_A = BaseNetwork(**model_params).to(device)
    base_A.load_state_dict(torch.load(
        RUN_DIR + f"coef={coef}/fold={fold}/base_half.pt", map_location=device))
    phm_A = RefitCovarNetwork(base_A, num_covariates=1, orthogonalize=False).to(device)
    phm_A.load_state_dict(torch.load(
        RUN_DIR + f"coef={coef}/fold={fold}/refit_age.pt", map_location=device))
    phm_A.eval()

    base_B = BaseNetwork(**model_params).to(device)
    base_B.load_state_dict(torch.load(
        RUN_DIR + f"coef={coef}/fold={fold}/base_half_B.pt", map_location=device))
    phm_B = RefitCovarNetwork(base_B, num_covariates=1, orthogonalize=False).to(device)
    phm_B.load_state_dict(torch.load(
        RUN_DIR + f"coef={coef}/fold={fold}/refit_age_BA.pt", map_location=device))
    phm_B.eval()
    return phm_A, phm_B


def per_subject_smoothed_ig(network, dataset, filt_ix, device, smooth_sigma):
    """Return (n, D, H, W) array of per-subject IG.

    If smooth_sigma > 0 each per-subject volume is Gaussian-smoothed; if 0,
    raw IG is returned (relevant when downstream aggregation already provides
    sufficient smoothing via averaging across folds and subjects).
    """
    subset = torch.utils.data.Subset(dataset, filt_ix)
    loader = torch.utils.data.DataLoader(
        subset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    out = []
    for batch in loader:
        x = batch["X"].to(device, non_blocking=True)
        x.requires_grad = True
        with IntegratedGradients(model=network, n_iter=IG_N_ITER) as attributor:
            _, rel = attributor(x)
        rel_np = rel.detach().cpu().numpy().squeeze(1)   # (B, D, H, W)
        for vol in rel_np:
            out.append(
                gaussian_filter(vol, sigma=smooth_sigma) if smooth_sigma > 0
                else vol
            )
    return np.stack(out, axis=0)


def benjamini_hochberg(p_flat):
    """BH-FDR adjustment. Input/output 1D."""
    n = p_flat.size
    order = np.argsort(p_flat)
    ranked = p_flat[order]
    p_adj = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity
    p_adj = np.minimum.accumulate(p_adj[::-1])[::-1]
    p_adj = np.clip(p_adj, 0.0, 1.0)
    out = np.empty_like(p_flat)
    out[order] = p_adj
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=GPU)
    parser.add_argument("--folds", type=int, nargs="+",
                        default=list(range(N_FOLDS)))
    parser.add_argument("--smooth-sigma", type=float, default=SMOOTH_SIGMA,
                        help="Per-subject Gaussian σ in voxels; 0 = no smoothing.")
    parser.add_argument("--age-stratum", choices=["all", "young", "old"],
                        default="all",
                        help="Restrict cohort to age < median ('young') or "
                             "≥ median ('old') of the n=100 sex==1, y==1 subjects.")
    parser.add_argument("--out-suffix", default="",
                        help="Append to chi2 output dirname, e.g. '_young'.")
    parser.add_argument("--models", nargs="+",
                        choices=["base_bal", "base_conf", "refit", "crossfit"],
                        default=["base_bal", "base_conf", "refit", "crossfit"],
                        help="Subset of models to (re)compute. Output files for "
                             "other models in the same dir are left untouched.")
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
    age_col = Z_te[:, 0]   # raw age, col 0 (cf. run_ukbb_experiment.py:175-176)
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
        else:  # "old"
            filt_ix = [base_filt[i] for i in range(len(base_filt))
                       if ages[i] >= median_age]
        print(f"age median split @ {median_age:.1f}: stratum={args.age_stratum}",
              flush=True)
    print(f"cohort: {len(filt_ix)} subjects", flush=True)

    # Brain mask — read from existing template (cohort-matched).
    template = nib.load(RUN_DIR + "lrp_maps/template_brain.nii.gz").get_fdata()
    brain_mask = template > 1e-3
    print(f"brain mask: {brain_mask.sum()} voxels "
          f"({100*brain_mask.mean():.1f}% of volume)", flush=True)

    model_params = default_model_params()
    all_loaders = {
        "base_bal":  lambda f: load_base(0.0, f, model_params, device),
        "base_conf": lambda f: load_base(2.0, f, model_params, device),
        "refit":     lambda f: RefitImageOnly(
                                  load_refit_age(2.0, f, model_params, device)),
        "crossfit":  lambda f: CrossFitRefitImageOnly(
                                  *load_crossfit_age(2.0, f, model_params, device)),
    }
    model_loaders = {m: all_loaders[m] for m in args.models}
    print(f"models to compute: {list(model_loaders)}", flush=True)

    # Accumulator: per-subject sum of per-fold smoothed IG per model.
    # Shape: (n_subjects, D, H, W).
    n = len(filt_ix)
    D, H, W = template.shape
    accum = {m: np.zeros((n, D, H, W), dtype=np.float32) for m in model_loaders}
    n_folds_used = 0

    for fold in args.folds:
        print(f"\n=== fold {fold} ===", flush=True)
        for mname, loader in model_loaders.items():
            print(f":: {mname} fold {fold}", flush=True)
            net = loader(fold)
            r = per_subject_smoothed_ig(net, test_ds, filt_ix, device,
                                        args.smooth_sigma)
            print(f":: :: shape={r.shape} "
                  f"min={r.min():.3e} max={r.max():.3e}", flush=True)
            accum[mname] += r
            del net, r
            torch.cuda.empty_cache()
            gc.collect()
        n_folds_used += 1

    # Per-subject fold mean.
    print(f"\nAveraging across {n_folds_used} folds ...", flush=True)
    fold_mean = {m: accum[m] / n_folds_used for m in accum}
    del accum
    gc.collect()

    out_dir = RUN_DIR + f"chi2{args.out_suffix}/"
    os.makedirs(out_dir, exist_ok=True)
    print(f"output dir: {out_dir}", flush=True)
    affine = np.eye(4)

    for mname, rbar in fold_mean.items():
        # Cohort-mean signed attribution: r̄_v = (1/n) Σ_i r̄_{i,v}.
        # Used for divergent (coolwarm) overlay panels — preserves sign /
        # direction information that the chi² T_v throws away.
        cohort_mean = rbar.mean(axis=0)  # shape (D, H, W)
        print(f"\n[{mname}] cohort-mean attribution: "
              f"min={cohort_mean.min():.3e}  max={cohort_mean.max():.3e}  "
              f"|max|={np.abs(cohort_mean).max():.3e}", flush=True)
        nib.save(nib.Nifti1Image(cohort_mean.astype(np.float32), affine),
                 out_dir + f"{mname}_mean.nii.gz")
        mx, my, mz = (cohort_mean.shape[0]//2,
                      cohort_mean.shape[1]//2,
                      cohort_mean.shape[2]//2)
        for orient, sl in [
            ("sagittal", np.rot90(cohort_mean[mx, :, :])),
            ("coronal",  np.rot90(cohort_mean[:, my, :])),
            ("axial",    np.rot90(cohort_mean[:, :, mz])),
        ]:
            np.savetxt(out_dir + f"{mname}_mean_{orient}.csv",
                       sl, delimiter=",", fmt="%.6e")

        print(f"[{mname}] computing T_v ...", flush=True)
        # Per-subject σ̂ via MAD inside brain.
        sigma_i = np.empty(n, dtype=np.float32)
        for i in range(n):
            in_brain = rbar[i][brain_mask]
            mad = np.median(np.abs(in_brain - np.median(in_brain)))
            sigma_i[i] = MAD_TO_SIGMA * mad + EPS_SIGMA
        print(f"  σ̂ summary: min={sigma_i.min():.3e}  "
              f"median={np.median(sigma_i):.3e}  max={sigma_i.max():.3e}",
              flush=True)
        # T_v = Σ_i r̄²_{i,v} / σ̂_i²    shape (D, H, W)
        T = np.einsum("ijkl,i->jkl",
                      rbar ** 2,
                      1.0 / (sigma_i ** 2))
        # χ²_n p-value, only inside brain to save compute on huge volumes.
        p = np.ones_like(T)
        p[brain_mask] = chi2.sf(T[brain_mask], df=n)
        # FDR across in-brain voxels.
        p_in = p[brain_mask]
        p_fdr_in = benjamini_hochberg(p_in)
        p_fdr = np.ones_like(p)
        p_fdr[brain_mask] = p_fdr_in

        # Cap −log10 p at 50 (= p < 1e-50). Some voxels have astronomically
        # small p-values that underflow chi2.sf to 0 → log10 → −inf; cap is
        # cosmetic but avoids inf in downstream plotting.
        NLP_CAP = 50.0
        nlp     = np.minimum(-np.log10(np.clip(p,     1e-300, 1.0)), NLP_CAP)
        nlp_fdr = np.minimum(-np.log10(np.clip(p_fdr, 1e-300, 1.0)), NLP_CAP)

        n_sig    = int((p[brain_mask]    < 0.05).sum())
        n_sig_fdr = int((p_fdr[brain_mask] < 0.05).sum())
        print(f"  in-brain voxels < 0.05 (uncorrected): {n_sig} / {brain_mask.sum()}",
              flush=True)
        print(f"  in-brain voxels < 0.05 (FDR):         {n_sig_fdr} / {brain_mask.sum()}",
              flush=True)

        nib.save(nib.Nifti1Image(T.astype(np.float32), affine),
                 out_dir + f"{mname}_T.nii.gz")
        nib.save(nib.Nifti1Image(nlp.astype(np.float32), affine),
                 out_dir + f"{mname}_neg_log10_p.nii.gz")
        nib.save(nib.Nifti1Image(nlp_fdr.astype(np.float32), affine),
                 out_dir + f"{mname}_neg_log10_pfdr.nii.gz")

        # Also dump central slices as CSV for the R plot.
        mx, my, mz = nlp_fdr.shape[0]//2, nlp_fdr.shape[1]//2, nlp_fdr.shape[2]//2
        for orient, sl in [
            ("sagittal", np.rot90(nlp_fdr[mx, :, :])),
            ("coronal",  np.rot90(nlp_fdr[:, my, :])),
            ("axial",    np.rot90(nlp_fdr[:, :, mz])),
        ]:
            np.savetxt(out_dir + f"{mname}_{orient}.csv",
                       sl, delimiter=",", fmt="%.6e")
        print(f"  wrote T, p, p_fdr (+ slices) to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
