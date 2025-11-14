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
            backbone_params=model.backbone_params
        )
        # Copy pretrained weights
        self.load_state_dict(model.state_dict())

        self.num_covariates = num_covariates
        self.orthogonalize = orthogonalize

        # Covariate components
        self.center_z = Center(num_covariates)
        self.fz = nn.Linear(num_covariates, 1, bias=False)

        # Optional orthogonalization
        self.orth = nn.Linear(num_covariates, 1, bias=False)
        self.orth.weight.data.zero_()

        # Store the fitted lambda
        self.lam = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    # -------------------------------------------------------------------------
    # Forward & prediction methods
    # -------------------------------------------------------------------------
    def forward(self, x, z):
        return self.intercept + self.predict_fx(x, z) + self.predict_fz(z)

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
    # Fitting logic
    # -------------------------------------------------------------------------
    def fit(self, dataloader, lam=None):
        """Fit post-hoc ridge regression (and optionally orthogonalization)."""
        self.center_effects(dataloader)
        self._fit_linear_effects(dataloader, lam)
        if self.orthogonalize:
            self._fit_orthogonalization(dataloader)
        return self

    @torch.no_grad()
    def center_effects(self, dataloader):
        """Center X, Z, and y using the dataloader."""
        X, Z, y = self._extract_features_from_loader(dataloader)

        self.center_x.fit(X)
        self.center_z.fit(Z)
        self.center_y.fit(y)

        self.intercept.data = self.center_y.mean
        self.is_centered.data = torch.tensor(True)
        return self

    # -------------------------------------------------------------------------
    # Internal methods
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _fit_linear_effects(self, dataloader, lam=None):
        """Fit linear weights on top of frozen backbone using R's glmnet."""
        X, Z, y = self._extract_features_from_loader(dataloader)

        # Center data
        X = self.center_x(X)
        Z = self.center_z(Z)
        y = self.center_y(y)

        # Prepare numpy arrays
        X_np = torch.cat([X, Z], dim=1).cpu().numpy()
        y_np = y.cpu().numpy()
        p_fac_np = np.ones(X_np.shape[1])
        #p_fac_np[X.shape[1]:] = 0.0  # No penalty on covariates

        with localconverter(ro.default_converter + numpy2ri.converter):
            X_r = ro.conversion.py2rpy(X_np)
            y_r = ro.conversion.py2rpy(y_np)
            p_fac_r = ro.conversion.py2rpy(p_fac_np)

        if lam is not None:
            fit = glmnet.glmnet(
                X_r, y_r, alpha=0,
                lambda_=ro.FloatVector([lam]),
                penalty_factor=p_fac_r,
                intercept=False
            )
            coefs = ro.r["as.matrix"](glmnet.coef_glmnet(fit, s=lam))
            best_lambda = lam
        else:
            cv_fit = glmnet.cv_glmnet(
                X_r, y_r, alpha=0,
                penalty_factor=p_fac_r,
                intercept=False
            )
            coefs = ro.r["as.matrix"](glmnet.coef_cv_glmnet(cv_fit, s="lambda.min"))
            best_lambda = ro.r["as.numeric"](cv_fit.rx2("lambda.min"))[0]

        # Store best Lambda
        self.lam.copy_(torch.tensor(best_lambda, dtype=torch.float32))
        
        # Update model parameters (Depreciated)
        #coefs_np = np.array(coefs).flatten()
        #n_feat = X.shape[1]
        #self.fx.weight.copy_(torch.tensor(coefs_np[1:n_feat + 1], dtype=torch.float32))
        #self.fz.weight.copy_(torch.tensor(coefs_np[n_feat + 1:], dtype=torch.float32))
        
        # Refit using Backfitting in Pytorch.
        # Estimate fx weights by LS fit on residuals:
        # 1. Regress y_tilde on Z to get residuals.
        resid_y = y - Z @ torch.linalg.lstsq(Z, y).solution
        # 2. Regress X on Z to get residuals.
        resid_X = X - Z @ torch.linalg.lstsq(Z, X).solution
        # 3. Fit fx on residuals.
        I = torch.eye(resid_X.shape[1], device=resid_X.device)
        beta_X = torch.linalg.solve(
            resid_X.T @ resid_X + self.lam * I,
            resid_X.T @ resid_y
        )

        # Estimate fz weights by LS fit on new residuals:
        # 1. Remove fx from y.
        resid_y = y - X @ beta_X
        # 2. Fit fz on residuals.
        I = torch.eye(Z.shape[1], device=Z.device)
        beta_Z = torch.linalg.solve(
            Z.T @ Z + self.lam * I,
            Z.T @ resid_y
        )

        # Update weights.
        self.fx.weight.data.copy_(beta_X.view(1, -1))
        self.fz.weight.data.copy_(beta_Z.view(1, -1))

        
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