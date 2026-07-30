"""UKBB HighAlc data: HDF5 loading, the synthetic-confounding DGP, and the covariate dataset/loader."""
import sys
import random
from pathlib import Path

import numpy as np
import h5py
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from scipy.stats import norm

# nitorch is vendored (not pip-installable); its transforms back default_transforms().
CODE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE_ROOT / "external" / "nitorch"))
from nitorch.transforms import IntensityRescale, ToTensor

# ── constants ─────────────────────────────────────────────────────────────────
RANDOM_STATE = 45
DATA_ROOT = Path("~/Research/Datasets/proj-orthogonalisation").expanduser()
TRAIN_H5 = DATA_ROOT / "t1mniz2-l-highalcl0u2-bingeauditl1u3-alcfreq-c-sex-age-n14617.h5"
HOLDOUT_H5 = DATA_ROOT / "holdout-t1mniz2-l-highalcl0u2-bingeauditl1u3-alcfreq-c-sex-age-n4505.h5"
PRETRAINED_RESNET = DATA_ROOT / "pretrained_models" / "r3d50_K_200ep.pth"


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
def sample_y_z(n, z_coef, rng, age_col, *, int_coef=0.0, rho_coef=0.0):
    """Draw synthetic targets with eta = 2*(sex-0.5) - z_coef*a + int_coef*(sex-0.5)*a, a=(age-mu)/(0.9*std).

    int_coef makes the image-borne signal interact with the confounder, taking the DGP
    outside the additive model class; rho_coef tilts P(sex=1|age) so signal and confounder
    are no longer independent. Both default to zero, reproducing the Section 6 DGP.
    """
    # age marginal
    mu_z, std_z = norm.fit(age_col)
    std_z *= 0.9
    z_range = np.arange(age_col.min(), age_col.max() + 1)
    p_z = norm.pdf(z_range, mu_z, std_z)
    p_z /= p_z.sum()
    z_sample = rng.choice(z_range, n, p=p_z)
    a_tilde = (z_sample - mu_z) / std_z

    # sex marginal; the uncorrelated branch keeps the released runs bit-exact, as binomial
    # would draw the same distribution but consume the generator stream differently
    if rho_coef == 0.0:
        sex_sample = rng.integers(0, 2, n)
    else:
        sex_sample = rng.binomial(1, 1 / (1 + np.exp(-rho_coef * a_tilde)), n)

    # outcome
    eta = 2.0 * (sex_sample - 0.5) - z_coef * a_tilde + int_coef * (sex_sample - 0.5) * a_tilde
    y_sample = rng.binomial(1, 1 / (1 + np.exp(-eta)), n)
    return np.column_stack((z_sample, sex_sample)), y_sample


def resample_synthetic(ydata, Zdata, n, z_coef, random_state, *, replace=True, int_coef=0.0, rho_coef=0.0):
    """Nearest-neighbour match real obs to synthetic (y, sex, age) targets; return n integer indices.

    Disjointness across two calls holds only when they operate on disjoint candidate
    pools — the caller splits the pool before resampling, not after. With replace=False
    each pool obs is used at most once. int_coef/rho_coef are passed to sample_y_z.
    """
    rng = np.random.default_rng(random_state)
    z_sample, y_sample = sample_y_z(n, z_coef, rng, age_col=Zdata[:, 0],
                                    int_coef=int_coef, rho_coef=rho_coef)
    used = np.zeros(len(ydata), dtype=bool)
    sample = []
    for i in range(n):
        idx = np.where((ydata == y_sample[i]) & (Zdata[:, 1] == z_sample[i, 1]) & ~used)[0]
        dist = np.abs(Zdata[idx, 0] - z_sample[i, 0])
        closest = np.where(dist == dist.min())[0]
        sample.append(idx[rng.choice(closest)])
        if not replace:
            used[sample[-1]] = True
    return np.array(sample)


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
