"""End-to-end competitor models: NAM-style networks training f(X) and f(Z) jointly by SGD."""
import torch
import torch.nn as nn

from cocodeel.model import _BaseCovarNetwork
from cocodeel.transform import Center


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

    def predict_fx(self, x, z=None):
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


class MLPCovarNetwork(CovarNetwork):
    """Covariate Network with an MLP covariate effect f(Z)."""

    def __init__(self, backbone, backbone_params=None, num_covariates=1, link="identity",
                 fz_hidden=32):
        super().__init__(backbone, backbone_params, num_covariates, link=link)
        self.fz = nn.Sequential(
            nn.Linear(num_covariates, fz_hidden),
            nn.ReLU(),
            nn.Linear(fz_hidden, fz_hidden),
            nn.ReLU(),
            nn.Linear(fz_hidden, 1),
        )
        self.center_fz = Center(1)

    def predict_fz(self, z):
        # nonlinear fz is centered on its output: E[fz(Z)] != fz(E[Z])
        return self.center_fz(self.fz(z))

    @torch.no_grad()
    def center_effects(self, dataloader):
        """Fit centering means and shift the intercept so predictions are unchanged."""
        if self.is_centered:
            return self
        X, Z, y = self._extract_features_from_loader(dataloader)
        self.center_x.fit(X)
        self.center_y.fit(y)
        self.center_fz.fit(self.fz(Z))
        self.intercept.data += self.fx(self.center_x.mean) + self.center_fz.mean
        self.is_centered.fill_(True)
        return self
