"""UKBB HighAlc data: HDF5 loading, the synthetic-confounding DGP, and the covariate dataset/loader."""
import os
import sys
import random
from pathlib import Path

import numpy as np
import h5py
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold

# nitorch is vendored (not pip-installable); its transforms back default_transforms().
CODE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE_ROOT / "external" / "nitorch"))
from nitorch.transforms import IntensityRescale, ToTensor

# ── constants ─────────────────────────────────────────────────────────────────
RANDOM_STATE = 45
# HDF5 files with the layout written by data/ukbb/make_dummy_data.py; UKBB_DATA_ROOT points to the real data
DATA_ROOT = Path(os.environ.get("UKBB_DATA_ROOT", CODE_ROOT / "data" / "ukbb"))
TRAIN_H5 = DATA_ROOT / "t1mniz2-l-highalcl0u2-bingeauditl1u3-alcfreq-c-sex-age-n14617.h5"
HOLDOUT_H5 = DATA_ROOT / "holdout-t1mniz2-l-highalcl0u2-bingeauditl1u3-alcfreq-c-sex-age-n4505.h5"
PRETRAINED_RESNET = DATA_ROOT / "pretrained_models" / "r3d50_K_200ep.pth"
N_SPLITS = 5                   # outer CV folds
NTRAIN = 5000                  # resampled training observations per outer fold
NTEST = 2500
# train_backbones.py writes the backbones; the refit, cross-fit and competitor scripts read them
RUN_DIR = CODE_ROOT / "experiments" / "ukbb" / "runs"
BACKBONE_DIR = RUN_DIR / "backbones"


# ── seeding ───────────────────────────────────────────────────────────────────
def seed_everything(seed):
    """Seed torch (incl. CUDA), numpy, and python-random."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# ── data loading ──────────────────────────────────────────────────────────────
def load_ukbb_data():
    """Load the training pool (n=14617) and holdout pool (n=4505): X, y=highalc, Z=[age, sex]."""
    with h5py.File(TRAIN_H5, "r") as f:
        X = f["X"][:]
        y = f["highalc"][:]
        Z = np.column_stack((f["age"][:], f["sex"][:]))
    with h5py.File(HOLDOUT_H5, "r") as f:
        X_test = f["X"][:]
        y_test = f["highalc"][:]
        Z_test = np.column_stack((f["age"][:], f["sex"][:]))
    return dict(X=X, y=y, Z=Z, X_test=X_test, y_test=y_test, Z_test=Z_test)


# ── synthetic confounding DGP ─────────────────────────────────────────────────
def sample_y_z(n, z_coef, rng, age_col):
    """Draw synthetic targets with eta = 2*(sex-0.5) - z_coef*a, a = (age-mu)/(0.9*std), y ~ Bernoulli(sigmoid(eta))."""
    # age marginal
    mu_z, std_z = norm.fit(age_col)
    std_z *= 0.9
    z_range = np.arange(age_col.min(), age_col.max() + 1)
    p_z = norm.pdf(z_range, mu_z, std_z)
    p_z /= p_z.sum()
    z_sample = rng.choice(z_range, n, p=p_z)
    a_tilde = (z_sample - mu_z) / std_z

    # sex marginal, independent of age
    sex_sample = rng.integers(0, 2, n)

    # outcome
    eta = 2.0 * (sex_sample - 0.5) - z_coef * a_tilde
    y_sample = rng.binomial(1, 1 / (1 + np.exp(-eta)), n)
    return np.column_stack((z_sample, sex_sample)), y_sample


def resample_synthetic(ydata, Zdata, n, z_coef, random_state):
    """Draw n synthetic (y, sex, age) targets and match each to a real observation with equal y and sex and the
    nearest age, with replacement; return the n integer indices.

    Two calls give disjoint samples only on disjoint candidate pools, so the caller splits the pool before resampling.
    """
    rng = np.random.default_rng(random_state)
    z_sample, y_sample = sample_y_z(n, z_coef, rng, age_col=Zdata[:, 0])
    sample = []
    for i in range(n):
        idx = np.where((ydata == y_sample[i]) & (Zdata[:, 1] == z_sample[i, 1]))[0]
        dist = np.abs(Zdata[idx, 0] - z_sample[i, 0])
        closest = np.where(dist == dist.min())[0]
        sample.append(idx[rng.choice(closest)])
    return np.array(sample)


def resample_partition(train_ix, y, Z, coef, seed, K):
    """Split an outer training fold into K stratified pools, then resample NTRAIN // K observations from each.

    Splitting before resampling keeps the K parts disjoint although resample_synthetic draws with replacement.
    """
    inner = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
    pools = [train_ix[ix] for _, ix in inner.split(np.zeros(len(train_ix)), y[train_ix])]
    return [pool[resample_synthetic(y[pool], Z[pool], NTRAIN // K, coef, seed + 100 * k)]
            for k, pool in enumerate(pools)]


def outer_fold(y, fold):
    """Training indices of outer CV fold `fold`."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return list(cv.split(np.zeros(len(y)), y))[fold][0]


# ── dataset / loaders ─────────────────────────────────────────────────────────
class NumpyCovarDataset(Dataset):
    """Numpy-backed dataset returning {"X", "y", "Z"} batches; Z is raw (models standardize internally)."""

    def __init__(self, X, y, Z, transform=None):
        self.X = X
        self.y = y[:, None] if y.ndim == 1 else y
        self.Z = Z[:, None] if Z.ndim == 1 else Z
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        image = self.X[idx]
        if self.transform:
            image = self.transform(image)
        if not torch.is_tensor(image):
            image = torch.tensor(image, dtype=torch.float32)
        return {
            "X": image,
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
            "Z": torch.tensor(self.Z[idx], dtype=torch.float32),
        }


def fast_loader(dataset, **kwargs):
    """DataLoader with pin_memory / persistent workers / prefetch for the 3D volumes."""
    return DataLoader(dataset, persistent_workers=True, pin_memory=True, prefetch_factor=4, **kwargs)


def default_transforms():
    """Intensity rescale (brain-masked) then to-tensor — the training/eval image pipeline."""
    return transforms.Compose([IntensityRescale(masked=True), ToTensor()])
