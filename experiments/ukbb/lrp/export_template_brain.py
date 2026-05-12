#!/usr/bin/env python
"""Compute a mean template brain over the (balanced) test set and a per-method
noise threshold for the LRP maps.

Outputs into <run_dir>/lrp_maps/:
  template_brain.nii.gz       — voxel-wise mean of all test images
  noise_thresholds.csv        — per-method `sigma_noise` and threshold = 3·σ
                                where σ is the std of LRP values OUTSIDE the
                                brain mask (i.e. pure-noise voxels).

Run:
    cd experiments/ukbb/lrp/
    conda run -n dl-mri python export_template_brain.py
"""
import os
import sys
import csv
import glob
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ukbb_common import (
    seed_everything, RANDOM_STATE, load_ukbb_data,
    resample_synthetic, NumpyCovarDataset, default_transforms,
)

NTEST = 2500
RUN_DIR = (
    "/home/RDC/pfeuffma/Research/proj-orthogonalisation/"
    "experiments/ukbb/runs/2026-04-14_17-26-52_final/"
)
LRP_DIR = RUN_DIR + "lrp_maps/"
# Pre-smooth LRP volumes with a 3D Gaussian to consolidate the salt-and-pepper
# signal into contiguous regions before thresholding. σ=2 voxels ~= 4 mm fwhm.
LRP_SMOOTH_SIGMA = 1.0
# Noise band defined as a quantile of |rel| INSIDE the brain (after smoothing).
# Using the in-brain distribution because non-brain inputs are zeroed by
# `IntensityRescale(masked=True)` so produce no useful noise estimate.
NOISE_QUANTILE = 0.90    # transparent if |smoothed rel| < q(0.90) in brain


def main():
    seed_everything(RANDOM_STATE)
    print("Loading data ...", flush=True)
    d = load_ukbb_data()
    X_test, y_test, Z_test = d["X_test"], d["y_test"], d["Z_full_test"]
    idx = resample_synthetic(y_test, Z_test, NTEST, 0.0, RANDOM_STATE)
    X_te, y_te, Z_te = X_test[idx], y_test[idx], Z_test[idx]

    tform = default_transforms()
    test_ds = NumpyCovarDataset(X_te, y_te, Z_te, tform)

    # Filter to the same cohort the LRP maps were computed on:
    # sex==1 AND y==1 (male high-alc cases). Z column 1 = sex, see
    # run_ukbb_experiment.py:175-176.
    filt_ix = [
        i for i in range(len(test_ds))
        if int(Z_te[i, 1]) == 1 and int(y_te[i]) == 1
    ]
    print(f"template cohort: {len(filt_ix)} subjects (sex==1, y==1)", flush=True)

    # Mean template brain over the cohort.
    print("Computing mean template ...", flush=True)
    accum = None
    for i in filt_ix:
        x = test_ds[i]["X"].numpy().squeeze()   # shape (D, H, W)
        accum = x.astype(np.float64) if accum is None else accum + x
    template = (accum / len(filt_ix)).astype(np.float32)
    print(f"template shape={template.shape}  range=[{template.min():.3f}, {template.max():.3f}]")

    out_template = LRP_DIR + "template_brain.nii.gz"
    nib.save(nib.Nifti1Image(template, np.eye(4)), out_template)
    print(f"wrote {out_template}", flush=True)

    # Brain mask from template (foreground voxels).
    eps = 1e-3
    brain_mask = template > eps
    print(f"brain mask: {brain_mask.sum()} / {brain_mask.size} voxels "
          f"({100*brain_mask.mean():.1f}%)", flush=True)

    # Per-method noise band: in-brain |rel| quantile of the SMOOTHED volume.
    # Smoothed volumes are also written back so extract_lrp_slices.py and the
    # R plot work from the same data the threshold was computed on.
    thresholds = []
    for f in sorted(glob.glob(LRP_DIR + "*_fold0_n*.nii.gz")):
        if "template" in f or "_smooth" in f:
            continue
        name = os.path.basename(f).replace(".nii.gz", "")
        method = name.split("_fold")[0]
        rel = nib.load(f).get_fdata()
        rel_s = gaussian_filter(rel, sigma=LRP_SMOOTH_SIGMA)
        # Persist smoothed volume for downstream use.
        out_smooth = LRP_DIR + name + "_smooth.nii.gz"
        nib.save(nib.Nifti1Image(rel_s.astype(np.float32), np.eye(4)),
                 out_smooth)
        in_brain_abs = np.abs(rel_s[brain_mask])
        threshold = float(np.quantile(in_brain_abs, NOISE_QUANTILE))
        thresholds.append({
            "method":      method,
            "threshold":   threshold,
            "quantile":    NOISE_QUANTILE,
            "smooth_sigma": LRP_SMOOTH_SIGMA,
            "n_brain_voxels": int(brain_mask.sum()),
        })
        print(f"{method:10s}  σ={LRP_SMOOTH_SIGMA}  "
              f"threshold (q={NOISE_QUANTILE:.2f}) = {threshold:.4e}",
              flush=True)

    out_csv = LRP_DIR + "noise_thresholds.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=thresholds[0].keys())
        w.writeheader()
        w.writerows(thresholds)
    print(f"wrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
