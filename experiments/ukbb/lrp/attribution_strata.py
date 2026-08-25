#!/usr/bin/env python
"""Integrated-Gradients maps per (sex, y) stratum for the unconfounded DNN, the confounded DNN, and the refit.

The three arms are the models a practitioner would hold: a DNN trained without
confounding (the target), a DNN trained under confounding (carrying the omitted
variable bias), and the sample-split refit of the confounded backbone. All are
evaluated on the same unconfounded test cohort, so any difference between the maps
is a difference in what the fitted image effect attends to.

Attribution is of f_X alone: for the refit that is predict_fx(x, z=None), which
excludes the covariate effect by construction.
"""
import os
import sys
import gc
import argparse
from pathlib import Path

# shared machine: cap the BLAS/OpenMP pools before numpy and torch read them
N_THREADS = int(os.environ.get("ATTRIB_THREADS", "8"))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(N_THREADS)

import numpy as np
import torch

torch.set_num_threads(N_THREADS)
import nibabel as nib
from scipy.stats import t as student_t
from scipy.ndimage import gaussian_filter
from zennit.attribution import Gradient, IntegratedGradients
from zennit.composites import EpsilonPlus
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.torchvision import ResNetCanonizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.ukbb.common.data import (
    seed_everything, RANDOM_STATE, resample_synthetic, NumpyCovarDataset,
    default_transforms, load_ukbb_holdout_meta, load_ukbb_holdout_images,
)
from experiments.ukbb.common.backbone import default_model_params
from cocodeel.model import BaseNetwork
from cocodeel.refit_model import RefitCovarNetwork

# ── run config ────────────────────────────────────────────────────────────────
NTEST = 2500
SRC_RUN = ROOT / "experiments/ukbb/runs/2026-04-26_13-16-37_final_v2"
REFIT_RUN = ROOT / "experiments/ukbb/runs/final_v2_refit"
TEMPLATE = SRC_RUN / "lrp_maps/template_brain.nii.gz"
STRATA = [(1, 1), (1, 0), (0, 1), (0, 0)]


class RefitImageOnly(torch.nn.Module):
    """Expose the refit's image effect f_X(x) as a plain forward for attribution."""

    def __init__(self, refit):
        super().__init__()
        self.refit = refit

    def forward(self, x):
        return self.refit.predict_fx(x, z=None)


def load_dnn(coef, fold, model_params, device):
    """base_full: the DNN a practitioner trains end to end on the whole sample."""
    net = BaseNetwork(**model_params).to(device)
    net.load_state_dict(torch.load(
        SRC_RUN / f"coef={coef}/fold={fold}/base_full.pt", map_location=device))
    net.eval()
    return net


def load_refit(coef, fold, model_params, device):
    """Sample-split refit: backbone trained on h1, last layer refit on the disjoint h2."""
    backbone = BaseNetwork(**model_params).to(device)
    backbone.load_state_dict(torch.load(
        SRC_RUN / f"coef={coef}/fold={fold}/base_half.pt", map_location=device))
    refit = RefitCovarNetwork(backbone, num_covariates=1, orthogonalize=False).to(device)
    refit.load_state_dict(torch.load(
        REFIT_RUN / f"coef={coef}/fold={fold}/refit_age.pt", map_location=device))
    refit.eval()
    return RefitImageOnly(refit)


def make_attributor(method, network, canonizer, baseline, n_iter):
    """LRP-EpsilonPlus (the original recipe) or Integrated Gradients."""
    if method == "lrp":
        canonizers = [ResNetCanonizer() if canonizer == "resnet" else SequentialMergeBatchNorm()]
        return Gradient(model=network, composite=EpsilonPlus(canonizers=canonizers))
    baseline_fn = (lambda x: baseline.expand_as(x)) if baseline is not None else torch.zeros_like
    return IntegratedGradients(model=network, n_iter=n_iter, baseline_fn=baseline_fn)


def stratum_map(network, dataset, ix, device, mask, opts):
    """Cohort-mean relevance over `ix`, each subject L1-normalised before averaging.

    Follows the original recipe: rel / rel.abs().sum() per subject, signed accumulation,
    mean over subjects, and |.| taken only at display time. The absolute-value-first
    aggregation is accumulated alongside, since the two orders answer different
    questions -- signed says which regions push the prediction which way on average,
    absolute says which regions are used at all regardless of direction.

    Per-subject normalisation is what makes the arms comparable: the refit head's norm
    is an order of magnitude larger than the trained head's and varies across folds, so
    unnormalised maps would differ in gain before they differ in pattern.
    """
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, ix), batch_size=opts["batch_size"],
        shuffle=False, num_workers=0, pin_memory=True,
    )
    acc = np.zeros(mask.shape, dtype=np.float64)
    acc_sq = np.zeros(mask.shape, dtype=np.float64)
    acc_abs = np.zeros(mask.shape, dtype=np.float64)
    n = 0
    for batch in loader:
        x = batch["X"].to(device, non_blocking=True)
        x.requires_grad = True
        with make_attributor(opts["method"], network, opts["canonizer"],
                             opts["baseline"], opts["n_iter"]) as attributor:
            _, rel = attributor(x)
        r = rel.detach().cpu().numpy().squeeze(1).astype(np.float64)
        if opts["sigma"] > 0:
            r = np.stack([gaussian_filter(v, opts["sigma"]) for v in r])
        # rel / rel.abs().sum(), over the whole volume as in the original
        r /= np.abs(r).sum(axis=(1, 2, 3))[:, None, None, None]
        acc += r.sum(axis=0)
        acc_sq += (r ** 2).sum(axis=0)
        acc_abs += np.abs(r).sum(axis=0)
        n += r.shape[0]
    mean = acc / n
    var = np.maximum(acc_sq / n - mean ** 2, 0.0) * n / (n - 1)
    tstat = mean / np.sqrt(np.maximum(var, 1e-30) / n)
    return (mean.astype(np.float32), (acc_abs / n).astype(np.float32),
            tstat.astype(np.float32), n)


def voxel_association(stack, v):
    """Per-voxel correlation between image intensity and `v` across the cohort.

    A mass-univariate map of the kind used in voxel-based morphometry. Built on the
    same preprocessed volumes the networks see, so it is directly comparable to an
    attribution map, and it is model-free — which is what lets it referee three
    models that do not share a backbone.
    """
    x = stack.reshape(stack.shape[0], -1).astype(np.float64)
    x -= x.mean(axis=0)
    v = v.astype(np.float64) - v.mean()
    num = v @ x
    den = np.sqrt((v @ v) * (x ** 2).sum(axis=0))
    return (num / np.maximum(den, 1e-30)).reshape(stack.shape[1:])


def save_slices(vol, out_dir, tag):
    """Write the three centre slices as CSVs; the R panel has no NIfTI reader."""
    mx, my, mz = (s // 2 for s in vol.shape)
    for orient, sl in [("sagittal", np.rot90(vol[mx, :, :])),
                       ("coronal", np.rot90(vol[:, my, :])),
                       ("axial", np.rot90(vol[:, :, mz]))]:
        np.savetxt(out_dir / f"{tag}_{orient}.csv", sl, delimiter=",", fmt="%.6e")


def benjamini_hochberg(p):
    """BH-FDR adjustment over a 1D array of p-values."""
    order = np.argsort(p)
    adj = p[order] * p.size / (np.arange(p.size) + 1)
    adj = np.clip(np.minimum.accumulate(adj[::-1])[::-1], 0.0, 1.0)
    out = np.empty_like(p)
    out[order] = adj
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-per-cell", type=int, default=150)
    parser.add_argument("--method", choices=["lrp", "ig"], default="lrp")
    parser.add_argument("--canonizer", choices=["seqbn", "resnet"], default="seqbn",
                        help="LRP only. seqbn matches the original recipe; resnet is "
                             "zennit's ResNet canonizer, which also handles the "
                             "residual connections.")
    parser.add_argument("--n-iter", type=int, default=20, help="IG path steps.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--smooth-sigma", type=float, default=0.0,
                        help="Per-subject Gaussian sigma in voxels; 0 disables, as in "
                             "the original recipe.")
    parser.add_argument("--baseline", choices=["zero", "cohort-mean"],
                        default="cohort-mean",
                        help="IG reference. cohort-mean attributes deviation from the "
                             "average brain; zero attributes the presence of a brain.")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    seed_everything(RANDOM_STATE)
    out_dir = Path(args.out) if args.out else REFIT_RUN / f"attribution_fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # cohort: the balanced (coef=0) holdout resample the paper evaluates on
    y_all, Z_all = load_ukbb_holdout_meta()
    idx = resample_synthetic(y_all, Z_all, NTEST, 0.0, RANDOM_STATE)
    y_te, Z_te = y_all[idx], Z_all[idx]
    strata_ix = {}
    for sex, y in STRATA:
        cell = np.where((Z_te[:, 1] == sex) & (y_te == y))[0][:args.n_per_cell]
        strata_ix[(sex, y)] = cell
        print(f"stratum sex={sex} y={y}: n={len(cell)}", flush=True)
    keep = np.concatenate([strata_ix[s] for s in STRATA])

    # only the selected volumes are read off disk
    print(f"reading {len(keep)} holdout volumes ...", flush=True)
    X_te = load_ukbb_holdout_images(idx[keep])
    local = {s: np.arange(len(keep))[np.isin(keep, strata_ix[s])] for s in STRATA}
    test_ds = NumpyCovarDataset(X_te, y_te[keep], Z_te[keep], default_transforms())

    mask = nib.load(TEMPLATE).get_fdata() > 1e-3
    print(f"brain mask: {mask.sum()} of {mask.size} voxels", flush=True)

    # reference maps: what age- and sex-related structure looks like in these images
    stack = torch.stack([test_ds[i]["X"] for i in range(len(test_ds))]).numpy().squeeze(1)
    ref_age = voxel_association(stack, Z_te[keep, 0])
    ref_sex = voxel_association(stack, Z_te[keep, 1])
    for name, ref in [("age", ref_age), ("sex", ref_sex)]:
        nib.save(nib.Nifti1Image(ref.astype(np.float32), np.eye(4)),
                 out_dir / f"ref_{name}.nii.gz")
        save_slices(ref.astype(np.float32), out_dir, f"ref_{name}")
        print(f"reference map {name}: |r|max={np.abs(ref[mask]).max():.3f} "
              f"|r|mean={np.abs(ref[mask]).mean():.3f}", flush=True)

    baseline = None
    if args.baseline == "cohort-mean":
        baseline = torch.from_numpy(stack.mean(axis=0)).to(device)[None, None]
    del stack

    model_params = default_model_params()
    arms = {
        "dnn_unconf": lambda: load_dnn(0.0, args.fold, model_params, device),
        "dnn_conf": lambda: load_dnn(2.0, args.fold, model_params, device),
        "refit": lambda: load_refit(2.0, args.fold, model_params, device),
    }

    opts = dict(method=args.method, canonizer=args.canonizer, baseline=baseline,
                n_iter=args.n_iter, batch_size=args.batch_size, sigma=args.smooth_sigma)
    print(f"attribution: {args.method}"
          + (f" ({args.canonizer} canonizer)" if args.method == "lrp" else ""), flush=True)

    maps = {}
    maps_abs = {}
    for arm, load in arms.items():
        net = load()
        for sex, y in STRATA:
            mean, absmean, tstat, n = stratum_map(net, test_ds, local[(sex, y)],
                                                  device, mask, opts)
            # BH-FDR over in-brain voxels, two-sided
            p = 2 * student_t.sf(np.abs(tstat[mask]), df=n - 1)
            p_fdr = benjamini_hochberg(p)
            sig = np.zeros(mask.shape, dtype=bool)
            sig[mask] = p_fdr < 0.05
            maps[(arm, sex, y)] = mean
            maps_abs[(arm, sex, y)] = absmean
            tag = f"{arm}_sex{sex}_y{y}"
            for name, vol in [("mean", mean), ("absmean", absmean),
                              ("sig", sig.astype(np.float32))]:
                nib.save(nib.Nifti1Image(vol, np.eye(4)), out_dir / f"{tag}_{name}.nii.gz")
                save_slices(vol, out_dir, f"{tag}_{name}")
            print(f":: {tag}: n={n} |mean|max={np.abs(mean).max():.3e} "
                  f"absmean_max={absmean.max():.3e} sig={sig.sum()} voxels", flush=True)
        del net
        torch.cuda.empty_cache()
        gc.collect()

    # two readings of the maps: against each other, and against the reference structure.
    # cos_abs uses the quantity actually plotted, |mean|, not the signed mean.
    rows = ["stratum,arm,cos_to_unconf,cos_abs_to_unconf,r_age,r_sex"]
    print("\nmap comparisons, in brain:", flush=True)
    for sex, y in STRATA:
        ref = maps[("dnn_unconf", sex, y)][mask]
        ref_a = np.abs(maps[("dnn_unconf", sex, y)])[mask]
        for arm in ["dnn_unconf", "dnn_conf", "refit"]:
            v = maps[(arm, sex, y)][mask]
            va = np.abs(maps[(arm, sex, y)])[mask]
            cos = float(v @ ref / (np.linalg.norm(v) * np.linalg.norm(ref)))
            cos_a = float(va @ ref_a / (np.linalg.norm(va) * np.linalg.norm(ref_a)))
            r_age = float(np.corrcoef(v, ref_age[mask])[0, 1])
            r_sex = float(np.corrcoef(v, ref_sex[mask])[0, 1])
            print(f"  sex={sex} y={y}  {arm:10s} cos={cos:+.4f}  cos|.|={cos_a:+.4f}  "
                  f"r_age={r_age:+.4f}  r_sex={r_sex:+.4f}", flush=True)
            rows.append(f"sex{sex}_y{y},{arm},{cos:.6f},{cos_a:.6f},{r_age:.6f},{r_sex:.6f}")
    (out_dir / "similarity.csv").write_text("\n".join(rows) + "\n")
    save_slices(nib.load(TEMPLATE).get_fdata(), out_dir, "template")
    print(f"\nwrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
