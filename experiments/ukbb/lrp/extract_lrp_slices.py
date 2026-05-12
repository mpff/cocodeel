#!/usr/bin/env python
"""Extract center sagittal/coronal/axial slices from LRP NIfTIs as CSVs.

The R plotting script has no NIfTI reader (RNifti / oro.nifti unavailable
on this machine). This helper reads the volumes written by
`export_lrp_maps.py` and writes one CSV per (model, orientation) for
geom_raster plotting in R.

Convention: slices through the center voxel of each axis. Orientation
naming mirrors the legacy notebook (`UKKBB_HighalcAgeSex_Synthetic.py:2154-2158`):
    sagittal = vol[mid_x, :, :]
    coronal  = vol[:, mid_y, :]
    axial    = vol[:, :, mid_z]
All slices are `np.rot90`-ed to display upright, matching the legacy plots.
"""
import os
import glob
import numpy as np
import nibabel as nib

RUN_DIR = (
    "/home/RDC/pfeuffma/Research/proj-orthogonalisation/"
    "experiments/ukbb/runs/2026-04-14_17-26-52_final/"
)
LRP_DIR = RUN_DIR + "lrp_maps/"


def main():
    # Read the SMOOTHED LRP volumes (written by export_template_brain.py).
    # Falling back to raw is intentionally avoided — if smoothing has not been
    # done, the threshold computed downstream won't match the slices.
    files = sorted(glob.glob(LRP_DIR + "*_fold0_n*_smooth.nii.gz"))
    if not files:
        raise FileNotFoundError(
            "No *_smooth.nii.gz volumes found. Run export_template_brain.py "
            "first to produce smoothed volumes + thresholds."
        )
    # Per-method vmax (volume-wide max of each model's 3D volume).
    # Written to vmax_per_method.csv for the R plot — each column is normalised
    # to its own max, so this file gives R both the raw max (for labelling) and
    # the per-method scale.
    per_method_vmax = {}
    for f in files:
        # Strip "_smooth" so output filenames stay backward-compatible.
        name = os.path.basename(f).replace("_smooth.nii.gz", "")
        method = name.split("_fold")[0]
        vol = nib.load(f).get_fdata()
        # Symmetric vmax: max absolute deviation from 0 (signed data → diverging scale).
        per_method_vmax[method] = max(per_method_vmax.get(method, 0.0),
                                       float(np.abs(vol).max()))
    print(f"per-method vmax: {per_method_vmax}", flush=True)
    with open(LRP_DIR + "vmax_per_method.csv", "w") as fh:
        fh.write("method,vmax\n")
        for method, v in per_method_vmax.items():
            fh.write(f"{method},{v:.6e}\n")

    for f in files:
        name = os.path.basename(f).replace("_smooth.nii.gz", "")
        vol = nib.load(f).get_fdata()
        mx, my, mz = vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2
        slices = {
            "sagittal": np.rot90(vol[mx, :, :]),
            "coronal":  np.rot90(vol[:, my, :]),
            "axial":    np.rot90(vol[:, :, mz]),
        }
        for orient, s in slices.items():
            out = LRP_DIR + f"{name}_{orient}.csv"
            np.savetxt(out, s, delimiter=",", fmt="%.6e")
            print(f"{out}  shape={s.shape}  max={s.max():.3e}", flush=True)

    # Template brain slices (background for the R plot).
    tpath = LRP_DIR + "template_brain.nii.gz"
    if os.path.exists(tpath):
        vol = nib.load(tpath).get_fdata()
        mx, my, mz = vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2
        slices = {
            "sagittal": np.rot90(vol[mx, :, :]),
            "coronal":  np.rot90(vol[:, my, :]),
            "axial":    np.rot90(vol[:, :, mz]),
        }
        for orient, s in slices.items():
            out = LRP_DIR + f"template_{orient}.csv"
            np.savetxt(out, s, delimiter=",", fmt="%.6e")
            print(f"{out}  shape={s.shape}  max={s.max():.3e}", flush=True)
    else:
        print(f"WARN: {tpath} missing — run export_template_brain.py first.",
              flush=True)


if __name__ == "__main__":
    main()
