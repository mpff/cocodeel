import torch
import torch.nn as nn

from cocodeel.links import LINKS
from cocodeel.transform import Center


class _BaseCovarNetwork(nn.Module):
    """Base class for networks with feature and covariate handling logic."""

    def __init__(self, backbone, backbone_params, num_covariates, link="identity"):
        super().__init__()
        self.backbone = backbone(**backbone_params)
        self.backbone_params = backbone_params
        self.num_covariates = num_covariates
        self.link = link
        self._link = LINKS[link]

        # Centering modules
        self.center_x = Center(self.backbone.out_features)
        self.center_z = Center(num_covariates) if num_covariates > 0 else None
        self.center_y = Center(1)
        self.register_buffer('is_centered', torch.tensor(False))

        # Output components
        self.intercept = nn.Parameter(torch.zeros(1), requires_grad=True)

    # -------------------------------------------------------------------------
    # Forward structure (to be specialized by subclasses)
    # -------------------------------------------------------------------------
    def forward(self, x, z=None):
        raise NotImplementedError("Subclasses must implement `forward`.")

    def predict_fx(self, x, z=None):
        raise NotImplementedError("Subclasses must implement `predict_fx`.")

    def predict_fz(self, z):
        raise NotImplementedError("Subclasses must implement `predict_fz`.")

    def output_func(self, eta):
        """Apply g⁻¹: linear predictor η → predicted mean μ."""
        return self._link.inverse(eta)

    # -------------------------------------------------------------------------
    # Centering logic
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def center_effects(self, dataloader):
        """Center X, Z, and y using the dataloader."""
        raise NotImplementedError("Subclasses must implement `center_effects`.")

    @torch.no_grad()
    def _fit_centers_from_loader(self, dataloader):
        """Fit center_x, center_z (if present), and center_y on the loader's
        data. Does NOT touch the intercept or `is_centered` — callers handle
        those based on the subclass's centering contract.
        """
        X, Z, y = self._extract_features_from_loader(dataloader)
        self.center_x.fit(X)
        if self.center_z is not None:
            self.center_z.fit(Z)
        self.center_y.fit(y)
    
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
    

class BaseNetwork(_BaseCovarNetwork):
    """Base Network: ignores covariates (Z)."""

    def __init__(self, backbone, backbone_params=None, num_covariates=0, link="identity"):
        backbone_params = backbone_params or {}
        super().__init__(backbone, backbone_params, num_covariates=0, link=link)
        
        # Add feature-specific components.
        self.fx = nn.Linear(self.backbone.out_features, 1, bias=False)

    def forward(self, x, z=None):
        eta = self.intercept + self.predict_fx(x)
        return self.output_func(eta)
    
    def predict_fx(self, x, z=None):
        x = self.backbone(x)
        x = self.center_x(x)
        return self.fx(x)
    
    def predict_fz(self, z):
        return torch.zeros(z.size(0), 1)
    
    @torch.no_grad()
    def center_effects(self, dataloader):
        """Fit centering means and shift the intercept so predictions are unchanged."""
        if self.is_centered:
            return self
        self._fit_centers_from_loader(dataloader)
        self.intercept.data += self.fx(self.center_x.mean)
        self.is_centered.fill_(True)
        return self