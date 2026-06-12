"""End-to-end competitor model.

`CovarNetwork` trains f(X) and f(Z) jointly by SGD (NAM-style). It is a
benchmark for the post-hoc refit, not part of the method: under
concurvity the jointly trained effects need not identify f(X) and f(Z)
separately, which is the failure mode the paper's simulations expose.
"""
import torch
import torch.nn as nn

from cocodeel.model import _BaseCovarNetwork


class CovarNetwork(_BaseCovarNetwork):
    """Covariate Network: includes covariate (Z) effects."""

    def __init__(self, backbone, backbone_params=None, num_covariates=1, link="identity"):
        backbone_params = backbone_params or {}
        super().__init__(backbone, backbone_params, num_covariates, link=link)

        # Add feature and covariate-specific components
        self.fx = nn.Linear(self.backbone.out_features, 1, bias=False)
        self.fz = nn.Linear(num_covariates, 1, bias=False)

    def forward(self, x, z):
        eta = self.intercept + self.predict_fx(x) + self.predict_fz(z)
        return self.output_func(eta)

    def predict_fx(self, x):
        x = self.backbone(x)
        x = self.center_x(x)
        return self.fx(x)

    def predict_fz(self, z):
        z = self.center_z(z)
        return self.fz(z)

    @torch.no_grad()
    def center_effects(self, dataloader):
        """Fit centering means and shift the intercept so predictions are unchanged."""
        if self.is_centered:
            return self
        self._fit_centers_from_loader(dataloader)
        self.intercept.data += self.fx(self.center_x.mean) + self.fz(self.center_z.mean)
        self.is_centered.fill_(True)
        return self
