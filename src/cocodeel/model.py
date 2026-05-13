import torch
import torch.nn as nn
from cocodeel.transform import Center


class _BaseCovarNetwork(nn.Module):
    """Base class for networks with feature and covariate handling logic."""

    def __init__(self, backbone, backbone_params, num_covariates, link="identity"):
        super().__init__()
        self.backbone = backbone(**backbone_params)
        self.backbone_params = backbone_params
        self.num_covariates = num_covariates
        self.link = link

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

    def predict_fx(self, x):
        raise NotImplementedError("Subclasses must implement `predict_fx`.")

    def predict_fz(self, z):
        raise NotImplementedError("Subclasses must implement `predict_fz`.")
    
    def output_func(self, eta):
        """Applies the link function to the linear predictor eta."""
        if self.link == "identity":
            return eta
        elif self.link == "logit":
            return torch.sigmoid(eta)
        elif self.link == "log":
            return torch.exp(eta)
        else:
            raise ValueError(f"Unsupported link function: {self.link}")

    # -------------------------------------------------------------------------
    # Centering logic
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def center_effects(self, dataloader):
        """Center X, Z, and y using the dataloader."""
        raise NotImplementedError("Subclasses must implement `center_effects`.")
    
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
    
    @torch.no_grad()
    def _variance_function(self, mu):
        """
        General variance function V(mu) for exponential family.
        Modify this function for different GLM families:
        - Gaussian: V(mu) = 1
        - Binomial: V(mu) = mu * (1 - mu)
        - Poisson: V(mu) = mu
        """
        if self.link == "logit":  # Bernoulli / logit
            return mu * (1 - mu)
        elif self.link == "identity":  # Gaussian
            return torch.ones_like(mu)
        elif self.link == "log":  # Poisson/log-link (approximates exp)
            return mu
        else:
            raise NotImplementedError("Variance function not implemented for this output function.")
    
    @torch.no_grad()
    def _link_derivative(self, mu):
        """
        Derivative of the link function g' for use in working response calculation.
        Modify for different GLM link functions:
        - Gaussian (identity): g'(mu) = 1
        - Binomial (logit): g'(mu) = mu * (1 - mu)
        - Poisson (log): g'(mu) = 1 / mu
        """
        if self.link == "logit":  # Bernoulli / logit
            return mu * (1 - mu)
        elif self.link == "identity":  # Gaussian
            return torch.ones_like(mu)
        elif self.link == "log":  # Poisson/log-link (approximates exp)
            return 1 / (mu + 1e-6)
        else:
            raise NotImplementedError("Link derivative not implemented for this output function.")

    @torch.no_grad()
    def _link_function(self, eta):
        """Applies the link function to y."""
        if self.link == "identity":
            return eta
        elif self.link == "log":
            return torch.log(eta + 1e-6)
        elif self.link == "logit":
            return torch.log(eta / (1 - eta + 1e-6) + 1e-6)
        else:
            raise ValueError(f"Unsupported link function: {self.link}")


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
        """Center X, Z, and y using the dataloader."""

        if self.is_centered:
            return self
    
        X, _, y = self._extract_features_from_loader(dataloader)

        self.center_x.fit(X)
        self.center_y.fit(y)

        self.intercept.data += self.fx(self.center_x.mean)
        self.is_centered.fill_(True)
        return self


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
        """Center X, Z, and y using the dataloader."""
        
        if self.is_centered:
            return self
        
        X, Z, y = self._extract_features_from_loader(dataloader)

        self.center_x.fit(X)
        self.center_z.fit(Z)
        self.center_y.fit(y)

        self.intercept.data += self.fx(self.center_x.mean) + self.fz(self.center_z.mean)
        self.is_centered.fill_(True)
        return self