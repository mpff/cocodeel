"""Adapter running the vendored original CIRCE implementation on tensor data.

CIRCE (Pogodin et al., "Efficient Conditionally Invariant Representation
Learning", ICLR 2023) penalizes the kernel conditional dependence of the
model on the covariates given the outcome. Original implementation:
https://github.com/namratadeka/circe, commit 0764872, MIT license, vendored
unmodified at external/circe. All training logic (trainer/circe.py), the
penalty (utils/losses.py), the LOO bandwidth/ridge selection and the model
construction run as released for the dsprites_linear regression experiment;
this module only marshals data in and predictions out.

Deviations from the released dsprites_linear experiment, all at the data or
harness level, none inside the vendored code:
  - the featurizer config replicates this package's TrafficBackbone
    (conv-conv-avgpool-fc, q output features) instead of the 64x64 dsprites
    CNN, mirroring how the released Yale-B config swaps in resnet18; fc1
    takes q inputs, the fc1/fc2/target head stack is as released.
  - images enter at native size with no Normalize/affine preprocessing:
    identical inputs to every other benchmark method.
  - modes are train/val only; the simulation has no OOD splits.
  - the train-split heldout carve for the conditional-mean KRR uses
    train_size=m with m = round(ratio * n) capped (default 10%, cap 1024)
    instead of the released test_size=1-ratio at 1% of ~73k images; same
    sklearn splitter, same random_state=42.
  - kernel_ft sigma2 is None, the source's median-heuristic branch: the
    released 0.01 assumes outputs in [0, 1], which our outcome is not.
    kernel_y is tuned by the released LOO routine (loo_cond_mean: True as
    released); kernel_z keeps the released 1.0 (Z in [0, 1] as in dsprites).
  - batch_size is capped at both split sizes (the vendored loaders set
    drop_last=True for every mode).
  - wandb is stubbed when not installed (only referenced behind the
    wandb=False flag); the early-stopping sys.exit in BaseTrainer.save is
    caught, since we run many fits per process.
  - after training, best.pth (their model selection: total validation
    loss) is loaded back before predictions are returned.
"""
import sys
import tempfile
import types
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# vendored package on sys.path; its internal imports are top-level
CIRCE_ROOT = Path(__file__).resolve().parents[3] / "external" / "circe"
if str(CIRCE_ROOT) not in sys.path:
    sys.path.insert(0, str(CIRCE_ROOT))
try:
    import wandb  # noqa: F401
except ImportError:
    sys.modules.setdefault("wandb", types.ModuleType("wandb"))

from config.config import Config
from data import data as circe_data
from trainer.circe import CIRCE


class _SimSplit(Dataset):
    """One split of a simulation draw in the vendored Dsprites interface."""

    def __init__(self, X, Z, y, heldout=None):
        self.images = X
        self.targets = y.view(-1)
        self.distractors = Z.view(-1)
        self.linear_reg = None
        if heldout is not None:
            ratio, cap = heldout
            n = self.targets.shape[0]
            m = min(int(round(ratio * n)), cap)
            held, keep = train_test_split(range(n), train_size=m, random_state=42)
            self.targets_heldout = self.targets[held].numpy().reshape(-1, 1)
            self.distractors_heldout = self.distractors[held].numpy().reshape(-1, 1)
            self.images = self.images[keep]
            self.targets = self.targets[keep]
            self.distractors = self.distractors[keep]

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, i):
        return {
            "x": self.images[i],
            "y": self.targets[i:i + 1],
            "z": self.distractors[i:i + 1],
        }


class _SplitBuilder:
    """Data-factory builder returning pre-built split datasets by mode key."""

    def __init__(self, splits):
        self.splits = splits

    def __call__(self, path, **_ignored):
        return self.splits[path]


def _featurizer_arch(q):
    """TrafficBackbone (conv-conv-avgpool-fc, q features) in the vendored Network layer-list format."""
    return [
        {"Conv2d": {"in_channels": 1, "out_channels": 16, "kernel_size": 3, "padding": 1}},
        {"ReLU": {"inplace": True}},
        {"Conv2d": {"in_channels": 16, "out_channels": 32, "kernel_size": 3, "padding": 1}},
        {"ReLU": {"inplace": True}},
        {"AdaptiveAvgPool2d": {"output_size": (4, 4)}},
        {"Flatten": {"start_dim": 1}},
        {"Linear": {"in_features": 512, "out_features": q}},
    ]


def circe_fit(X_tr, Z_tr, y_tr, X_va, Z_va, y_va, lam=10.0, lr=1e-4, epochs=200,
              patience=25, batch_size=1024, q=32, heldout_ratio=0.1,
              heldout_cap=1024, workdir=None):
    """Train the vendored CIRCE pipeline on one draw; returns the trainer with best-checkpoint weights loaded."""
    workdir = Path(tempfile.mkdtemp(prefix="circe_")) if workdir is None else Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # datasets in the vendored interface
    splits = {
        "train": _SimSplit(X_tr, Z_tr, y_tr, heldout=(heldout_ratio, heldout_cap)),
        "val": _SimSplit(X_va, Z_va, y_va),
    }
    circe_data.factory.register_builder("simtraffic", _SplitBuilder(splits))

    # configs mirroring config/dsprites_linear/circe.yml
    bs = min(batch_size, len(splits["train"]), len(splits["val"]))
    data_cfg = Config("circe_adapter", dict(
        data_key="simtraffic", train="train", val="val",
        batch_size=bs, num_workers=0, regress=True,
    ))
    model_cfg = Config("circe_adapter", dict(
        model_key="regressor", trainer_key="circe", modes=["train", "val"],
        epochs=epochs, patience=patience, lamda=lam, ridge_lambda=0.1,
        kernel_ft={"gaussian": {"sigma2": None}},
        kernel_y={"gaussian": {"sigma2": 0.01}},
        kernel_z={"gaussian": {"sigma2": 1.0}},
        n_last_reg_layers=1, zy_cov=True, loo_cond_mean=True,
        biased=False, centered_circe=True,
        network={
            "featurizer": _featurizer_arch(q),
            "fc1": [{"Linear": {"in_features": q, "out_features": 128}},
                    {"LeakyReLU": {"inplace": True}}],
            "fc2": [{"Linear": {"in_features": 128, "out_features": 64}}],
            "target": [{"Linear": {"in_features": 64, "out_features": 1}}],
        },
        optimizer={"AdamW": {"lr": lr, "weight_decay": 0.01}},
        scheduler={"CosineAnnealingLR": {"T_max": epochs}},
    ))
    exp_cfg = Config("circe_adapter", dict(
        description="CIRCE on simulated data",
        output_location=str(workdir), version="simtraffic",
        run_name="simtraffic", wandb=False, load=None, resume=False,
    ))

    # train under the source protocol
    trainer = CIRCE(data_cfg=data_cfg, model_cfg=model_cfg, exp_cfg=exp_cfg)
    assert hasattr(trainer, "W_1"), "heldout KRR precompute failed"
    try:
        trainer.run()
    except SystemExit:
        # BaseTrainer.save signals early stopping by exiting the process
        pass

    # restore their model selection: best total validation loss. The
    # checkpoint was written by this very fit; weights_only=False because
    # their save_model stores the loss as a numpy scalar (torch-1.12-era).
    ckpt = torch.load(workdir / "best.pth", map_location=trainer.device, weights_only=False)
    trainer.model.load_state_dict(ckpt["model"])
    trainer.model.eval()
    return trainer


@torch.no_grad()
def circe_predict(trainer, X, batch_size=200):
    """Model outputs on images X, batched on the trainer's device."""
    trainer.model.eval()
    outs = []
    for i in range(0, X.shape[0], batch_size):
        _, y_ = trainer.model(X[i:i + batch_size].to(trainer.device))
        outs.append(y_.cpu())
    return torch.cat(outs)


class CirceRosterModel(torch.nn.Module):
    """Fitted CIRCE in the benchmark prediction interface: the Z-free model
    output, centered on the reference loader's sample like every other
    method's f_X (the carved-out KRR heldout is part of that sample)."""

    def __init__(self, trainer, center_loader):
        super().__init__()
        self.trainer = trainer
        self.num_covariates = 0
        self.offset = circe_predict(trainer, center_loader.dataset.X).mean().item()

    def eval(self):
        self.trainer.model.eval()
        return self

    def forward(self, x, z=None):
        _, y_ = self.trainer.model(x.to(self.trainer.device))
        return y_

    def predict_fx(self, x, z=None):
        return self.forward(x) - self.offset

    def predict_fz(self, z):
        return torch.zeros(z.size(0), 1)
