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
        self.is_centered = True
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
        #penalty_factor = np.ones(X_np.shape[1])  # all penalized equally (can be customized!)

        with localconverter(ro.default_converter + numpy2ri.converter):
            X_r = ro.conversion.py2rpy(X_np)
            y_r = ro.conversion.py2rpy(y_np)
            #p_fac_r = ro.conversion.py2rpy(penalty_factor)

        if lam is not None:
            fit = glmnet.glmnet(
                X_r, y_r, alpha=0,
                lambda_=ro.FloatVector([lam]),
                #penalty_factor=p_fac_r,
                intercept=False
            )
            coefs = ro.r["as.matrix"](glmnet.coef_glmnet(fit, s=lam))
            best_lambda = lam
        else:
            cv_fit = glmnet.cv_glmnet(
                X_r, y_r, alpha=0,
                #penalty_factor=p_fac_r,
                intercept=False
            )
            coefs = ro.r["as.matrix"](glmnet.coef_cv_glmnet(cv_fit, s="lambda.min"))
            best_lambda = ro.r["as.numeric"](cv_fit.rx2("lambda.min"))[0]

        coefs_np = np.array(coefs).flatten()
        n_feat = X.shape[1]

        # Update model parameters
        self.fx.weight.copy_(torch.tensor(coefs_np[1:n_feat + 1], dtype=torch.float32))
        self.fz.weight.copy_(torch.tensor(coefs_np[n_feat + 1:], dtype=torch.float32))
        self.lam.copy_(torch.tensor(best_lambda, dtype=torch.float32))

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