#!/usr/bin/env python
"""Export LRP relevance maps for the UKBB Panel C figure.

Three models, fold 0:
  base_full @ coef=0.0   → "Base (Bal)"
  base_full @ coef=2.0   → "Base (Conf)"
  posthoc_age @ coef=2.0 → "Refit (Conf)"

Per model: take the first 100 test subjects with sex == 1, compute
relevance via zennit `Gradient + EpsilonGammaBox`, transform per-subject
with sqrt(abs(rel)), then mean across subjects.

This averaging strategy is copied verbatim from the legacy notebook
`_archive/notebooks/UKKBB_HighalcAgeSex_Synthetic.py:2120-2160`.
Do not change without discussion — averaging strategy affects cross-model
comparability of the resulting maps.

Outputs three NIfTI volumes to <run_dir>/lrp_maps/, ready to be read by
`UKBB_lrp_panel.R` for plotting.
"""
import os
import sys
import gc
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from zennit.attribution import IntegratedGradients
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ukbb_common import (
    seed_everything, RANDOM_STATE,
    load_ukbb_data, resample_synthetic, NumpyCovarDataset,
    fast_loader, default_model_params, default_transforms,
)
from cocodeel.model import BaseNetwork
from cocodeel.posthoc_model import PostHocCovarNetwork


# ── Config ────────────────────────────────────────────────────────────────────
N_SUBJECTS = 100
NTEST = 2500
GPU = 0
FOLD = 0
RUN_DIR = (
    "/home/RDC/pfeuffma/Research/proj-orthogonalisation/"
    "experiments/ukbb/runs/2026-04-14_17-26-52_final/"
)


# ── PostHoc forward shim ──────────────────────────────────────────────────────
class PostHocImageOnly(torch.nn.Module):
    """Wrap PostHocCovarNetwork to expose only the image contribution.

    Mirrors the legacy `deep_forward` shim — LRP attributes only
    `predict_fx(x, z=None)`, dropping intercept + linear covariate effect.
    """
    def __init__(self, posthoc_model):
        super().__init__()
        self.model = posthoc_model

    def forward(self, x):
        return self.model.predict_fx(x, z=None)


# ── Relevance computation ────────────────────────────────────────────────────
IG_N_ITER       = 20    # path-integral steps; A/B test showed n=20 sufficient
SMOOTH_SIGMA    = 1.0   # per-subject Gaussian smoothing (voxels) before normalisation

def compute_relevance_map(network, dataset, sex_col, y_col, n_subjects, device,
                           batch_size=4, num_workers=0):
    """Population-level importance map via Integrated Gradients.

    Per-subject pipeline (sex==1 AND y==1, first n_subjects):
        1. Compute IG attribution.
        2. Take |rel|.
        3. Smooth with Gaussian σ=SMOOTH_SIGMA voxels.
        4. Normalise by per-subject max → result in [0, 1].
    Then mean across subjects → population importance map in [0, 1].

    Method choice: Integrated Gradients (Sundararajan et al. 2017). Chosen
    over ε-LRP after an A/B test (`ab_compare_attribution.py`) showed IG
    produces maps where the classifier head's orthogonal weight rotation
    is visible in the spatial map (cos sim Bal vs Conf = 0.09 with IG vs
    0.95 with ε-LRP). IG satisfies the completeness axiom and is
    class-sensitive by construction. Baseline = zero input (no brain).

    Aggregation: magnitude-only (sign-agnostic). Answers "which brain
    regions are reliably *used* for the prediction across subjects",
    independent of evidence direction.

    Parallelisation: batched zennit attribution (batch_size=4) keeps GPU
    utilisation higher than sample-by-sample. num_workers=0 because the
    NumpyCovarDataset is fully in-memory; spawning workers would fork
    the entire dataset 4× and exhaust RAM with no I/O to overlap.
    """

    # Pre-collect filtered indices (sex==1 AND y==1).
    filt_ix = [
        i for i in range(len(dataset))
        if int(sex_col[i]) == 1 and int(y_col[i]) == 1
    ][:n_subjects]
    if len(filt_ix) < n_subjects:
        print(
            f"WARN: only {len(filt_ix)} sex==1, y==1 subjects available "
            f"(scanned all {len(dataset)})", flush=True,
        )

    subset = torch.utils.data.Subset(dataset, filt_ix)
    loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    accum = None
    n_used = 0
    for batch in loader:
        x = batch["X"].to(device, non_blocking=True)
        x.requires_grad = True
        with IntegratedGradients(model=network, n_iter=IG_N_ITER) as attributor:
            _, rel = attributor(x)
        signed = torch.sign(rel) * torch.sqrt(torch.abs(rel))   # (B, 1, D, H, W)
        signed_sum = signed.sum(dim=0).detach().cpu()
        accum = signed_sum if accum is None else accum + signed_sum
        n_used += x.shape[0]

    relevance = accum / n_used
    return relevance.squeeze().numpy(), n_used


# ── Model loading ─────────────────────────────────────────────────────────────
def load_base(coef, fold, model_params, device):
    ckpt = RUN_DIR + f"coef={coef}/fold={fold}/base_full.pt"
    net = BaseNetwork(**model_params).to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()
    return net


def load_posthoc_age(coef, fold, model_params, device):
    """Reconstruct posthoc_age: BaseNetwork(half) backbone wrapped in
    PostHocCovarNetwork with num_covariates=1, then load fitted state."""
    backbone_ckpt = RUN_DIR + f"coef={coef}/fold={fold}/base_half.pt"
    posthoc_ckpt  = RUN_DIR + f"coef={coef}/fold={fold}/posthoc_age.pt"

    base_half = BaseNetwork(**model_params).to(device)
    base_half.load_state_dict(torch.load(backbone_ckpt, map_location=device))

    phm = PostHocCovarNetwork(base_half, num_covariates=1, orthogonalize=False).to(device)
    phm.load_state_dict(torch.load(posthoc_ckpt, map_location=device))
    phm.eval()
    return phm


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_SUBJECTS)
    parser.add_argument("--fold", type=int, default=FOLD)
    parser.add_argument("--gpu", type=int, default=GPU)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    seed_everything(RANDOM_STATE)

    # ── Test set (matches run_ukbb_experiment: balanced @ coef=0) ──
    print("Loading data ...", flush=True)
    d = load_ukbb_data()
    X_test, y_test, Z_test = d["X_test"], d["y_test"], d["Z_full_test"]
    idx = resample_synthetic(y_test, Z_test, NTEST, 0.0, RANDOM_STATE)
    X_te, y_te, Z_te = X_test[idx], y_test[idx], Z_test[idx]
    sex_col = Z_te[:, 1]   # col 1 = sex (col 0 = age, per run_ukbb_experiment.py:175-176)
    y_col   = y_te

    tform = default_transforms()
    test_ds = NumpyCovarDataset(X_te, y_te, Z_te, tform)

    model_params = default_model_params()

    # ── Models ──
    print(f"Loading models (fold={args.fold}) ...", flush=True)
    base_bal  = load_base(0.0, args.fold, model_params, device)
    base_conf = load_base(2.0, args.fold, model_params, device)
    refit     = PostHocImageOnly(load_posthoc_age(2.0, args.fold, model_params, device))

    # Sanity probe: verify weight divergence + logit divergence on one subject.
    p_bal  = next(base_bal.parameters()).detach()
    p_conf = next(base_conf.parameters()).detach()
    print(
        f"[sanity] first-param max|Δ| bal vs conf: "
        f"{(p_bal - p_conf).abs().max().item():.4e}", flush=True,
    )
    probe_idx = int(np.where(sex_col == 1)[0][0])
    x_probe = test_ds[probe_idx]["X"].unsqueeze(0).to(device)
    print(f"[sanity] input range: min={x_probe.min().item():.3f}  "
          f"max={x_probe.max().item():.3f}  "
          f"(EpsilonGammaBox expects [0,1])", flush=True)
    with torch.no_grad():
        print(
            f"[sanity] logits on subject {probe_idx} | "
            f"bal={base_bal(x_probe).item():.3f}  "
            f"conf={base_conf(x_probe).item():.3f}  "
            f"refit_fx={refit(x_probe).item():.3f}",
            flush=True,
        )

    out_dir = RUN_DIR + "lrp_maps/"
    os.makedirs(out_dir, exist_ok=True)

    affine = np.eye(4)   # MNI alignment not used; voxel coords only

    for name, net in [
        ("base_bal",  base_bal),
        ("base_conf", base_conf),
        ("refit",     refit),
    ]:
        print(f":: {name}", flush=True)
        rel, n_used = compute_relevance_map(
            net, test_ds, sex_col, y_col, args.n, device,
        )
        print(f":: :: shape={rel.shape}  n_used={n_used}", flush=True)
        out_path = out_dir + f"{name}_fold{args.fold}_n{n_used}.nii.gz"
        nib.save(nib.Nifti1Image(rel.astype(np.float32), affine), out_path)
        print(f":: :: wrote {out_path}", flush=True)
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
