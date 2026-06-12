"""NAM with a true MLP shape function for the covariate effect f_z.

`cocodeel.benchmarking.model.CovarNetwork` is a NAM in name only: f_x is a deep network
(image backbone + linear last layer) but f_z is a single linear map of Z.
For the concurvity-exploration sweep we want a "proper" NAM where f_z is
also a small neural network, so both component shape functions are
non-trivial.

Centering caveat. `CovarNetwork.center_effects` works because the linear
f_z commutes with input-centering: f_z(z - μ_z) is zero-mean iff f_z is
linear. With an MLP f_z this no longer holds. The subclass below ignores
the input-centering of Z (the inherited `center_z` buffer is fitted by
`_fit_centers_from_loader` but never applied — `predict_fz` takes raw z)
and instead records the empirical training-set output mean of the MLP in
a buffer `fz_output_mean`; `predict_fz` returns `MLP(z) - fz_output_mean`.
The intercept is shifted by the same mean so the total prediction is
unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from cocodeel.benchmarking.model import CovarNetwork


class CovarNetworkMLPfz(CovarNetwork):
    """CovarNetwork with an MLP shape function for f_z (post-training centering)."""

    def __init__(self, backbone, backbone_params=None, num_covariates=1,
                 link="identity", fz_hidden: int = 32):
        super().__init__(backbone, backbone_params, num_covariates, link=link)
        # Replace the inherited Linear(num_covariates, 1, bias=False) f_z.
        # Last layer keeps bias=False so any post-training shift is absorbed
        # by fz_output_mean / intercept, not split between two parameters.
        self.fz = nn.Sequential(
            nn.Linear(num_covariates, fz_hidden),
            nn.ReLU(),
            nn.Linear(fz_hidden, fz_hidden),
            nn.ReLU(),
            nn.Linear(fz_hidden, 1, bias=False),
        )
        self.register_buffer("fz_output_mean", torch.zeros(1))

    def predict_fz(self, z):
        return self.fz(z) - self.fz_output_mean

    @torch.no_grad()
    def center_effects(self, dataloader):
        if self.is_centered:
            return self
        self._fit_centers_from_loader(dataloader)  # center_x.mean is used below
        # μ_fz := E_train[MLP(Z)] on the raw MLP (buffer still zero here so
        # predict_fz == MLP(z) at this point).
        self.eval()
        device = next(self.parameters()).device
        fz_sum = torch.zeros(1, device=device)
        n = 0
        for batch in dataloader:
            z = batch["Z"].to(device)
            fz_sum = fz_sum + self.fz(z).sum(dim=0)
            n += z.size(0)
        mu_fz = (fz_sum / max(1, n)).detach()
        self.fz_output_mean.copy_(mu_fz)
        # Re-parameterisation: total prediction is unchanged.
        self.intercept.data += self.fx(self.center_x.mean) + mu_fz
        self.is_centered.fill_(True)
        return self
