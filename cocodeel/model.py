import torch
import torch.nn as nn
from cocodeel.transform import Center


class _BaseCovarNetwork(nn.Module):
    """Base class for networks with feature and covariate handling logic."""

    def __init__(self, backbone, backbone_params, num_covariates):
        super().__init__()
        self.backbone = backbone(**backbone_params)
        self.backbone_params = backbone_params
        self.num_covariates = num_covariates

        # Centering modules
        self.center_x = Center(self.backbone.out_features)
        self.center_y = Center(1)
        self.is_centered = False

        # Output components
        self.fx = nn.Linear(self.backbone.out_features, 1, bias=False)
        self.intercept = nn.Parameter(torch.zeros(1), requires_grad=True)

    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _extract_features_from_loader(self, dataloader):
        """Extract backbone features (f(X)), covariates (Z), and targets (y)."""
        self.eval()
        device = next(self.parameters()).device

        Xs, Zs, ys = [], [], []
        for batch in dataloader:
            Xb = self.backbone(batch["X"].to(device))
            Xs.append(Xb)
            Zs.append(batch["Z"].to(device))
            ys.append(batch["y"].to(device))

        X = torch.cat(Xs, dim=0)
        Z = torch.cat(Zs, dim=0)
        y = torch.cat(ys, dim=0)
        return X, Z, y

    # -------------------------------------------------------------------------
    # Forward structure (to be specialized by subclasses)
    # -------------------------------------------------------------------------
    def forward(self, x, z=None):
        raise NotImplementedError("Subclasses must implement `forward`.")

    def predict_fx(self, x, z=None):
        """Feature contribution."""
        x = self.backbone(x)
        x = self.center_x(x)
        return self.fx(x)

    def predict_fz(self, z):
        """Covariate contribution (default: zeros)."""
        return torch.zeros(z.shape[0], 1, device=z.device)

    # -------------------------------------------------------------------------
    # Centering logic
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def center_effects(self, dataloader):
        """Fit centering modules for X and y, and adjust intercept."""
        X, _, y = self._extract_features_from_loader(dataloader)

        self.center_x.fit(X)
        self.center_y.fit(y)
        self.intercept.data = self.center_y.mean
        self.is_centered = True
        return self


class BaseNetwork(_BaseCovarNetwork):
    """Base Network: centers features (X) and target (y), ignores covariates (Z)."""

    def __init__(self, backbone, backbone_params=None, num_covariates=0):
        backbone_params = backbone_params or {}
        super().__init__(backbone, backbone_params, num_covariates=0)

    def forward(self, x, z=None):
        return self.intercept + self.predict_fx(x, z)


class CovarNetwork(_BaseCovarNetwork):
    """Covariate Network: centers features (X), covariates (Z), and target (y)."""

    def __init__(self, backbone, backbone_params=None, num_covariates=1):
        backbone_params = backbone_params or {}
        super().__init__(backbone, backbone_params, num_covariates)

        # Add covariate-specific components
        self.center_z = Center(num_covariates)
        self.fz = nn.Linear(num_covariates, 1, bias=False)

    def forward(self, x, z):
        return self.intercept + self.predict_fx(x, z) + self.predict_fz(z)

    def predict_fz(self, z):
        z = self.center_z(z)
        return self.fz(z)

    @torch.no_grad()
    def center_effects(self, dataloader):
        """Fit centering modules for X, Z, and y, and adjust intercept."""
        X, Z, y = self._extract_features_from_loader(dataloader)

        self.center_x.fit(X)
        self.center_z.fit(Z)
        self.center_y.fit(y)
        self.intercept.data = self.center_y.mean
        self.is_centered = True
        return self
