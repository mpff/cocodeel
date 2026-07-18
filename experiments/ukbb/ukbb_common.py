"""Shared utilities for UKBB HighAlc synthetic confounding experiments.

All experiment scripts in this directory import from here to avoid duplication.
"""
import os
import sys
import random
import datetime
import json
import subprocess
from pathlib import Path

import numpy as np
import h5py
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from scipy.stats import norm

# ── Project path ──────────────────────────────────────────────────────────────
# cocodeel is pip-installed (`pip install -e .` at the code/ root). nitorch
# (vendored) and the experiment-shared backbones module are not pip-installable
# yet — keep their sys.path lines, resolved relative to this file.
code_root = Path(__file__).resolve().parents[2]  # → ovb-ddns/code/
sys.path.insert(0, str(code_root / "external" / "nitorch"))       # for nitorch
sys.path.insert(1, str(code_root / "experiments" / "common"))     # for backbones

from nitorch.transforms import IntensityRescale, ToTensor
from cocodeel.model import BaseNetwork
from cocodeel.refit_model import RefitCovarNetwork
from cocodeel.trainer import covar_trainer
from backbones import ResNet, Bottleneck

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_STATE = 45

_TRAIN_H5 = Path(
    "~/Research/Datasets/proj-orthogonalisation/"
    "t1mniz2-l-highalcl0u2-bingeauditl1u3-alcfreq-c-sex-age-n14617.h5"
).expanduser()

_HOLDOUT_H5 = Path(
    "~/Research/Datasets/proj-orthogonalisation/"
    "holdout-t1mniz2-l-highalcl0u2-bingeauditl1u3-alcfreq-c-sex-age-n4505.h5"
).expanduser()

_PRETRAINED_RESNET = Path(
    "~/Research/Datasets/proj-orthogonalisation/"
    "pretrained_models/r3d50_K_200ep.pth"
).expanduser()

_STUDY = "UKBB_HighAlcSex_Synthetic_Study"


# ── Seeding ───────────────────────────────────────────────────────────────────
def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_ukbb_data() -> dict:
    """Load UKBB training and holdout datasets from HDF5.

    Returns a dict with keys:
        X, y, Z_full (age+sex), Z_age — training pool (n=14617)
        X_test, y_test, Z_full_test, Z_age_test — holdout pool (n=4505)
    """
    data = h5py.File(_TRAIN_H5, "r")
    X = data["X"][:]
    y = data["highalc"][:]
    Z_full = np.column_stack((data["age"][:], data["sex"][:]))
    Z_age = data["age"][:].reshape(-1, 1)
    data.close()

    data_te = h5py.File(_HOLDOUT_H5, "r")
    X_test = data_te["X"][:]
    y_test = data_te["highalc"][:]
    Z_full_test = np.column_stack((data_te["age"][:], data_te["sex"][:]))
    Z_age_test = data_te["age"][:].reshape(-1, 1)
    data_te.close()

    return dict(
        X=X, y=y, Z_full=Z_full, Z_age=Z_age,
        X_test=X_test, y_test=y_test, Z_full_test=Z_full_test, Z_age_test=Z_age_test,
    )


# ── Synthetic resampling ──────────────────────────────────────────────────────
def sample_y_z(n: int, z_coef: float, rng, age_col: np.ndarray):
    """Sample synthetic (z, y) pairs from the DGP.

    DGP: logit(P(y=1)) = 2*(sex - 0.5) + z_coef*(age - mu_age) / (0.9 * std_age)

    Args:
        n: number of samples to draw
        z_coef: confounding strength coefficient
        rng: numpy.random.Generator instance
        age_col: 1-D array of real ages (used to fit mu_age, std_age)
    """
    mu_z, std_z = norm.fit(age_col)
    std_z *= 0.9
    z_range = np.arange(age_col.min(), age_col.max() + 1)
    p_z = norm.pdf(z_range, mu_z, std_z)
    p_z /= p_z.sum()
    z_sample = rng.choice(z_range, n, p=p_z)
    sex_sample = rng.integers(0, 2, n)
    p_y = 1 / (1 + np.exp(-2.0 * (sex_sample - 0.5) + z_coef * (z_sample - mu_z) / std_z))
    y_sample = rng.binomial(1, p_y, n)
    return np.column_stack((z_sample, sex_sample)), y_sample


def resample_synthetic(
    ydata: np.ndarray,
    Zdata: np.ndarray,
    n: int,
    z_coef: float,
    random_state,
    *,
    replace: bool = True,
) -> np.ndarray:
    """Nearest-neighbour matching to generate a synthetic-confounding subsample.

    Matches real observations to target (y, sex, age) drawn from the DGP.
    Returns an array of n integer indices into ydata/Zdata.
    Observations can appear multiple times within the returned array (nearest-
    neighbour matching is sampling with replacement at the population level).
    Disjointness between two calls is guaranteed only when they operate on
    disjoint candidate pools — this is enforced by the caller (split pool before
    resampling, not after).

    Args:
        ydata: labels (0/1) for the candidate pool
        Zdata: covariate matrix (col 0 = age, col 1 = sex) for the candidate pool
        n: desired subsample size
        z_coef: confounding strength
        random_state: int or None — seeds the internal RNG
        replace: if False, each pool obs may appear at most once. Raises if a
            (y, sex) cell exhausts before n draws are made.
    """
    rng = np.random.default_rng(random_state)
    z_sample, y_sample = sample_y_z(n, z_coef, rng, age_col=Zdata[:, 0])
    used = np.zeros(len(ydata), dtype=bool)
    sample = []
    for i in range(n):
        idx = np.where(
            (ydata == y_sample[i]) & (Zdata[:, 1] == z_sample[i, 1]) & ~used
        )[0]
        closest = np.where(
            np.abs(Zdata[idx, 0] - z_sample[i, 0])
            == np.abs(Zdata[idx, 0] - z_sample[i, 0]).min()
        )[0]
        sample.append(idx[rng.choice(closest)])
        if not replace:
            used[sample[-1]] = True
    return np.array(sample)


# ── Dataset ───────────────────────────────────────────────────────────────────
class NumpyCovarDataset(Dataset):
    """Numpy-backed dataset returning {"X", "y", "Z"} batches."""

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


# ── DataLoader factory ────────────────────────────────────────────────────────
def fast_loader(dataset: Dataset, **kwargs) -> DataLoader:
    """DataLoader with performance defaults (pin_memory, persistent_workers)."""
    return DataLoader(
        dataset,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=4,
        **kwargs,
    )


# ── Backbone ──────────────────────────────────────────────────────────────────
class ResNet50(nn.Module):
    """3D ResNet50 wrapper with pretrained weight loading for UKBB MRI."""

    def __init__(self, freeze_feature_extractor: bool = False, pretrained_model: str = ""):
        super().__init__()
        self.num_features = 2048
        self.out_features = 2048
        self.feature_extractor = ResNet(Bottleneck, [3, 4, 6, 3], [64, 128, 256, 512])
        if pretrained_model:
            sd = torch.load(pretrained_model, weights_only=False)
            if "state_dict" in sd:
                sd = sd["state_dict"]
            if sd["conv1.weight"].shape[1] == 3:
                sd["conv1.weight"] = sd["conv1.weight"].sum(1, keepdim=True)
            self.feature_extractor.load_state_dict(sd, strict=False)
        if freeze_feature_extractor:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.feature_extractor(x)


# ── Default configs ───────────────────────────────────────────────────────────
def default_model_params() -> dict:
    return {
        "backbone": ResNet50,
        "backbone_params": {"pretrained_model": str(_PRETRAINED_RESNET)},
        "num_covariates": 0,
        "link": "logit",
    }


def default_trainer_params(gpu: int) -> dict:
    return {
        "device": gpu,
        "epochs": 128,
        "lr": 1.67e-6,
        "weight_decay": 1.02e-5,
        "patience": 20,
        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "scheduler_kwargs": {"patience": 5, "factor": 0.8},
        "use_amp": True,
    }


def default_transforms():
    return transforms.Compose([IntensityRescale(masked=True), ToTensor()])


# ── Z standardization ─────────────────────────────────────────────────────────
def standardize_z(Z_tr: np.ndarray):
    """Compute mean and std of Z_tr for standardization.

    Returns (mean, std) arrays of shape (1, ncov). std is clipped to 1.0 for
    constant columns (e.g. binary sex).
    """
    m = Z_tr.mean(0, keepdims=True)
    s = Z_tr.std(0, keepdims=True)
    s[s < 1e-8] = 1.0
    return m, s


# ── Run infrastructure ────────────────────────────────────────────────────────
def setup_run_dir(suffix: str) -> str:
    """Create a timestamped run directory and return its path."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = proj_path + f"experiments/ukbb/runs/{ts}_{suffix}/"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def write_manifest(results_dir: str, config: dict) -> None:
    config["pid"] = os.getpid()
    config["host"] = os.uname().nodename
    config["start_time"] = datetime.datetime.now().isoformat()
    try:
        config["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=proj_path
        ).decode().strip()
    except Exception:
        config["git_commit"] = "unknown"
    with open(results_dir + "manifest.json", "w") as f:
        json.dump(config, f, indent=2)
