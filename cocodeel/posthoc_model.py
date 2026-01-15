from os import link
import torch
import torch.nn as nn
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

from cocodeel.model import BaseNetwork
from cocodeel.transform import Center

# Load glmnet from R
glmnet = importr("glmnet")


class PostHocCovarNetwork(BaseNetwork):
    """
    Post-hoc fitted Neural Network with centered features and linear covariate effects.
    Optionally orthogonalizes covariates to feature effects.

    Steps:
    1. Centers features, covariates, and targets.
    2. Fits a ridge regression (glmnet in R) combining feature and covariate contributions.
    3. Optionally fits an orthogonalization matrix so that covariates are orthogonal to features.
    """

    def __init__(self, model, num_covariates, orthogonalize=False):
        super().__init__(
            backbone=model.backbone.__class__,
            backbone_params=model.backbone_params,
            num_covariates=num_covariates,
            link=model.link
        )
        self.load_state_dict(model.state_dict())
        self.num_covariates = num_covariates
        self.orthogonalize = orthogonalize

        # Covariate components.
        self.center_z = Center(num_covariates)
        self.fz = nn.Linear(num_covariates, 1, bias=False)

        # Optional orthogonalization.
        self.orth = nn.Linear(num_covariates, 1, bias=False)
        self.orth.weight.data.zero_()

        # Penalization Paramter for ridge refit.
        self.lam = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    # -------------------------------------------------------------------------
    # Forward & prediction methods
    # -------------------------------------------------------------------------
    def forward(self, x, z):
        eta = self.intercept + self.predict_fx(x, z) + self.predict_fz(z)
        return self.output_func(eta)

    def predict_fx(self, x, z=None):
        """Feature contribution, optionally orthogonalized against Z."""
        x = self.backbone(x)
        x = self.center_x(x)
        fx = self.fx(x)
        if self.orthogonalize:
            z = self.center_z(z)
            fx -= self.orth(z)
        return fx

    def predict_fz(self, z):
        """Covariate contribution, optionally corrected for orthogonalization."""
        z = self.center_z(z)
        fz = self.fz(z)
        if self.orthogonalize:
            fz += self.orth(z)
        return fz

    # -------------------------------------------------------------------------
    # Fitting methods
    # -------------------------------------------------------------------------
    def fit(self, dataloader, lam=None, max_iters=10, tol=1e-6):
        """Fit post-hoc ridge regression (and optionally orthogonalization)."""
        self.center_effects(dataloader)
        self._fit_effects(dataloader, lam=lam, max_iters=max_iters, tol=tol)
        if self.orthogonalize:
            self._fit_orthogonalization(dataloader)
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

    # -------------------------------------------------------------------------
    # Internal methods
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _fit_effects(self, dataloader, lam, max_iters, tol):
        """
        IRLS-based fitting of generalized additive effects with ridge penalty.
        Model: y = g(fx(x) + fz(z))
        Using weighted least squares with GLM variance + link derivative.
        """
        
        # Grab data.
        X, Z, y = self._extract_features_from_loader(dataloader)
        X = self.center_x(X)
        Z = self.center_z(Z)

        # Initialize intercept.
        eta_mean = self.center_y.mean
        self.intercept.data = self._link_function(eta_mean)

        # Initialize old weights for convergence check.
        fz_old = torch.zeros_like(self.fz.weight.data)
        fx_old = torch.zeros_like(self.fx.weight.data)

        # Iteratively reweighted least squares until convergence.
        for i in range(max_iters):
            
            # ---- Prepare reweighted data ----

            # Get current predictions, weight matrix, and working response.
            eta = self.intercept + self.fx(X) + self.fz(Z)
            mu = self.output_func(eta)
            var = self._variance_function(mu)
            g_prime = self._link_derivative(mu)
            weights = g_prime**2 / (var + 1e-6)
            W = torch.diag(weights.flatten())
            y_work = eta + (y - mu) / (g_prime + 1e-6)
            
            # Update intercept.
            self.intercept.data = y_work.mean()

            # Centered reweighting. (Note: not equal to demeaning in case of weights!)
            yw = torch.sqrt(W) @ (y_work - (W @ y_work).sum() / weights.sum())
            Xw = torch.sqrt(W) @ (X - (W @ X).sum(dim=0, keepdim=True) / weights.sum())
            Zw = torch.sqrt(W) @ (Z - (W @ Z).sum(dim=0, keepdim=True) / weights.sum())
            
            # --- Fitting Procedure ---
            
            # 1. Regress yw on Zw to get residuals.
            resid_y = yw - Zw @ torch.linalg.lstsq(Zw, yw).solution
        
            # 2. Regress Xw on Zw to get residuals.
            resid_X = Xw - Zw @ torch.linalg.lstsq(Zw, Xw).solution
            
            # 3. Use glmnet to fit cross-validated ridge regression of resid_y on resid_X.
            # NOTE: glmnet does not seem to be reliable for refitting. We do CV for lam if not given
            # and then refit with the best lam using torch.linalg.lstsq.
            if lam is None:
                # 3a. Prepare data for R.
                resid_X_np = resid_X.cpu().numpy()
                resid_y_np = resid_y.cpu().numpy()
                with localconverter(ro.default_converter + numpy2ri.converter):
                    X_r = ro.conversion.py2rpy(resid_X_np)
                    y_r = ro.conversion.py2rpy(resid_y_np)
                # 3b. Fit ridge regression in R (glmnet), either with given lam or CV.
                cv_fit = glmnet.cv_glmnet(
                    X_r, y_r,
                    alpha=0,
                    intercept=False
                )
                best_lambda = ro.r["as.numeric"](cv_fit.rx2("lambda.min"))[0]
                self.lam.copy_(torch.tensor(best_lambda, dtype=torch.float32))
            else:
                self.lam.copy_(torch.tensor(lam, dtype=torch.float32))
            # 3c. Refit ridge regression with best lam using torch.
            beta_fx = torch.linalg.solve(
                resid_X.T @ resid_X + self.lam.item() * torch.eye(resid_X.shape[1]),
                resid_X.T @ resid_y
            )
            self.fx.weight.data.copy_(beta_fx.view(1, -1))
            
            # 4. Compute fz weights.
            resid_y_new = yw - self.fx(Xw)
            beta_fz = torch.linalg.solve(Zw.T @ Zw, Zw.T @ resid_y_new)
            self.fz.weight.data.copy_(beta_fz.view(1, -1))
            
            # 5. Check convergence via relative change in fx weights and deviance (TODO)
            if self.link == "identity":
                break  # No need for IRLS iterations for identity link
            delta_fx = torch.norm(self.fx.weight.data - fx_old) / (torch.norm(fx_old) + 1e-8)
            delta_fz = torch.norm(self.fz.weight.data - fz_old) / (torch.norm(fz_old) + 1e-8)
            if delta_fx < tol and delta_fz < tol:
                break
            fx_old = self.fx.weight.data.clone()
            fz_old = self.fz.weight.data.clone()

        return self
        
    @torch.no_grad()
    def _fit_orthogonalization(self, dataloader):
        """Fit linear orthogonalization term via least squares on (Z, fX)."""
        self.eval()
        X, Z, _ = self._extract_features_from_loader(dataloader)

        X = self.center_x(X)
        Z = self.center_z(Z)
        fX = self.fx(X)

        # Solve Z * beta = fX
        solution = torch.linalg.lstsq(Z, fX).solution
        self.orth.weight.copy_(solution.T)
