import torch
import torch.nn as nn

from cocodeel.benchmarking.model import CovarNetwork
from cocodeel.model import BaseNetwork
from cocodeel.transform import Center


class PostHocOrthNetwork(BaseNetwork):
    """Post-hoc orthogonalization baseline (Weber recipe): centres effects
    and removes the linear Z-component from fX via lstsq. No covariate
    refit — fz is identically zero. Fit on the full sample.
    """

    def __init__(self, model, num_covariates):
        super().__init__(
            backbone=model.backbone.__class__,
            backbone_params=model.backbone_params,
            link=model.link
        )
        self.load_state_dict(model.state_dict())
        self.num_covariates = num_covariates

        self.orth = nn.Linear(self.num_covariates, 1, bias=False)
        self.orth.weight.data.fill_(0.0)  # Initialize orthogonalization to 0.

        self.center_z = Center(self.num_covariates)

    def forward(self, x, z):
        return self.output_func(self.intercept + self.predict_fx(x, z))

    def predict_fx(self, x, z):
        x = self.backbone(x)
        x = self.center_x(x)
        z = self.center_z(z)
        fx = self.fx(x) - self.orth(z)
        return fx

    def predict_fz(self, z):
        # return vector of (batch, 1) of zeros
        return torch.zeros(z.shape[0], 1, device=z.device)

    def fit(self, train_dataloader, val_dataloader=None):
        """Fit on the training data; `val_dataloader` is accepted and ignored (interface parity with `RefitCovarNetwork.fit`)."""
        self.center_effects(train_dataloader)
        self._fit_orthogonalization(train_dataloader)
        return self

    @torch.no_grad()
    def center_effects(self, dataloader):
        """Center X, Z, and y using the dataloader."""
        X, Z, y = self._extract_features_from_loader(dataloader)
        self.center_z.fit(Z)
        self.center_y.fit(y)
        # A wrapped model that was already centered carries its intercept
        # shift in the loaded state dict; shifting again would bias eta by
        # fx(mean_x).
        if not self.is_centered:
            self.center_x.fit(X)
            self.intercept.data += self.fx(self.center_x.mean)
            self.is_centered.data = torch.tensor(True)
        return self

    @torch.no_grad()
    def _fit_orthogonalization(self, dataloader):
        """Fit linear orthogonalization term via least squares on (Z, fX)."""
        X, Z, _ = self._extract_features_from_loader(dataloader)
        X = self.center_x(X)
        Z = self.center_z(Z)
        fX = self.fx(X)
        # Solve Z * beta = fX. predict_fx subtracts orth on centered Z, a
        # mean-zero term on the fit sample — the intercept needs no
        # compensation.
        solution = torch.linalg.lstsq(Z, fX).solution
        self.orth.weight.copy_(solution.T)


class FeatureErasureNetwork(BaseNetwork):
    """Erases Z from the backbone features phi of a fitted model and refits the head by least squares.

    With phi and z centred on the fit sample:
      "residualize" (Weber et al.):  r(phi, z) = phi - B^T z,  B = argmin_B ||phi - Z B||  (QR of Z)
      "leace" (Belrose et al. 2023): r(phi) = phi - W^+ P W phi,  W = Sigma_phiphi^{-1/2},
                                     P the orthogonal projection onto the column space of W Sigma_phiz
    then f_X = r^T w with w = argmin_w ||y - ybar - R w||. Fit on the full sample, fz is identically zero.
    """

    def __init__(self, model, num_covariates, eraser):
        super().__init__(
            backbone=model.backbone.__class__,
            backbone_params=model.backbone_params,
            link=model.link
        )
        self.load_state_dict(model.state_dict())
        self.num_covariates = num_covariates
        self.eraser = eraser
        self.center_z = Center(num_covariates)
        q = self.backbone.out_features
        # residualize: phi_r = phi - z B;  leace: phi_r = phi - phi M^T
        self.register_buffer("B", torch.zeros(num_covariates, q))
        self.register_buffer("M", torch.zeros(q, q))

    def forward(self, x, z):
        return self.output_func(self.intercept + self.predict_fx(x, z))

    def erase(self, phi, z):
        if self.eraser == "residualize":
            return phi - self.center_z(z) @ self.B
        return phi - phi @ self.M.T

    def predict_fx(self, x, z):
        return self.fx(self.erase(self.center_x(self.backbone(x)), z))

    def predict_fz(self, z):
        return torch.zeros(z.shape[0], 1, device=z.device)

    @torch.no_grad()
    def fit(self, train_dataloader, val_dataloader=None):
        """Fit on the training data; `val_dataloader` is accepted and ignored (interface parity with `RefitCovarNetwork.fit`)."""
        phi, Z, y = self._extract_features_from_loader(train_dataloader)
        self.center_x.fit(phi)
        self.center_z.fit(Z)
        self.center_y.fit(y)
        phi = self.center_x(phi).double()
        Z = self.center_z(Z).double()

        if self.eraser == "residualize":
            Q, R_z = torch.linalg.qr(Z)
            B = torch.linalg.solve_triangular(R_z, Q.T @ phi, upper=True)
            self.B.copy_(B)
            R = phi - Z @ B
        else:
            n = phi.shape[0]
            evals, evecs = torch.linalg.eigh(phi.T @ phi / n)
            # pseudo-inverse square roots on the numerical support of Sigma_phiphi
            keep = evals > evals.max() * 1e-10
            V, s = evecs[:, keep], evals[keep].sqrt()
            W = V @ torch.diag(1 / s) @ V.T
            W_pinv = V @ torch.diag(s) @ V.T
            U, sv, _ = torch.linalg.svd(W @ (phi.T @ Z / n), full_matrices=False)
            # W Sigma_phiz is the covariance of whitened phi with z; its singular
            # values are bounded by sd(z), so float32 rounding sits near 1e-7 sd(z)
            U = U[:, sv > 1e-5 * Z.std(dim=0).max()]
            M = W_pinv @ U @ U.T @ W
            self.M.copy_(M)
            R = phi - phi @ M.T

        w =torch.linalg.lstsq(R, (y - self.center_y.mean).double()).solution
        self.fx.weight.copy_(w.T)
        self.intercept.copy_(self.center_y.mean)
        self.is_centered.fill_(True)
        return self


class SemiStructuredNetwork(CovarNetwork):
    """Semi-structured network baseline (Rügamer et al.): wraps a fitted
    CovarNetwork and adds a post-hoc orthogonalization term via
    lstsq(Z, fX). Total η is unchanged — the term moves the linear
    Z-component of fX into fz.
    """

    def __init__(self, model):
        super().__init__(
            backbone=model.backbone.__class__,
            backbone_params=model.backbone_params,
            num_covariates=model.num_covariates,
            link=model.link
        )
        self.load_state_dict(model.state_dict())
        self.orth = nn.Linear(model.num_covariates, 1, bias=False)
        self.orth.weight.data.fill_(0.0)  # Initialize orthogonalization to 0.

    def forward(self, x, z):
        eta = self.intercept + self.predict_fx(x, z) + self.predict_fz(z)
        return self.output_func(eta)

    def predict_fx(self, x, z):
        x = self.backbone(x)
        x = self.center_x(x)
        z = self.center_z(z)
        fx = self.fx(x) - self.orth(z)
        return fx

    def predict_fz(self, z):
        z = self.center_z(z)
        fz = self.fz(z) + self.orth(z)
        return fz

    def fit(self, train_dataloader, val_dataloader=None):
        """Fit on the training data; `val_dataloader` is accepted and ignored (interface parity with `RefitCovarNetwork.fit`)."""
        self.center_effects(train_dataloader)
        self._fit_orthogonalization(train_dataloader)
        return self

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
        self.is_centered.data = torch.tensor(True)
        return self

    @torch.no_grad()
    def _fit_orthogonalization(self, dataloader):
        """Fit linear orthogonalization term via least squares on (Z, fX)."""

        X, Z, _ = self._extract_features_from_loader(dataloader)

        X = self.center_x(X)
        Z = self.center_z(Z)
        fX = self.fx(X)

        # Solve Z * beta = fX
        solution = torch.linalg.lstsq(Z, fX).solution
        self.orth.weight.copy_(solution.T)
