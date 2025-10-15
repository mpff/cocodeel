import torch
import torch.nn as nn

import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

# import glmnet from R
glmnet = importr("glmnet")

from cocodeel.model import BaseNetwork
from cocodeel.transform import Center

class PostHocCovarNetwork(BaseNetwork):
    def __init__(self, model, num_covariates, orthogonalize=False):
        """ Post-hoc fitted Neural Network with CENTRED effects and LINEAR covariate effects.
        Can include orthogonalization of covariates to features.
        Fit in two steps:
        1) Center the features, covariates and target (removing mean effects).
        2a) Fit a linear model with ridge penalty on the features and no penalty on the covariates.
        2b) If orthogonalize=True, orthogonalize covariates to features using linear regression.
        This is done using the glmnet package in R via rpy2.
        Parameters:
            model (BaseNetwork): Prefitted model with a backbone and fx layer.
            num_covariates (int): The number of covariates used in the last layer.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        Methods:
            forward: Defines the forward computation at every call.
            _fit: Fits the model to the training data.
        """
        super().__init__(
            backbone=model.backbone.__class__,
            backbone_params=model.backbone_params
        )
        self.load_state_dict(model.state_dict())
        self.num_covariates = num_covariates
        self.orthogonalize = orthogonalize

        self.fz = nn.Linear(self.num_covariates, 1, bias=False)
        self.orth = nn.Linear(self.num_covariates, 1, bias=False)
        self.orth.weight.data.fill_(0.0)  # Initialize orthogonalization to 0.

        self.center_z = Center(self.num_covariates)
        self.is_centered = False

        self.lam = nn.Parameter(torch.tensor(0.0), requires_grad=False)  # Store lambda used for fitting.

    def forward(self, x, z):
        return self.intercept + self.predict_fx(x, z) + self.predict_fz(z)

    def predict_fx(self, x, z=None):
        x = self.backbone(x)
        x = self.center_x(x)
        fx = self.fx(x)
        if self.orthogonalize and z is not None:
            z = self.center_z(z)
            fx = fx - self.orth(z)
        return fx

    def predict_fz(self, z):
        z = self.center_z(z)
        fz = self.fz(z)
        if self.orthogonalize:
            fz = fz + self.orth(z)
        return fz

    def fit(self, train_dataloader, lam=None):
        self.center_effects(train_dataloader)
        self._linear_fit_from_loader(train_dataloader, lam)
        if self.orthogonalize:
            self._fit_orthogonalization(train_dataloader)
        return self

    def center_effects(self, dataloader):
        self.eval()
        # Gather all features, covariates and targets.
        X, Z, y = [], [], []
        device = next(self.parameters()).device
        with torch.no_grad():
            for batch in dataloader:
                Xb, Zb, yb = batch["X"].to(device), batch["Z"].to(device), batch["y"].to(device)
                Xb = self.backbone(Xb)
                X.append(Xb.cpu())
                Z.append(Zb.cpu())
                y.append(yb.cpu())
        X = torch.cat(X, dim=0)
        Z = torch.cat(Z, dim=0)
        y = torch.cat(y, dim=0)
        # Fit centering modules.
        self.center_x.fit(X)
        self.center_z.fit(Z)
        self.center_y.fit(y)
        # Adjust intercept.
        self.intercept.data = self.center_y.mean
        self.is_centered = True
        return self

    def _linear_fit_from_loader(self, train_dataloader, lam=None):
        device = next(self.parameters()).device
        features = []
        covariates = []
        targets = []
        with torch.no_grad():
            for batch in train_dataloader:
                x = batch["X"].to(device)
                z = batch["Z"].to(device)
                y = batch["y"].to(device)
                x = self.backbone(x)
                x = self.center_x(x)
                z = self.center_z(z)
                y = self.center_y(y)
                features.append(x)
                covariates.append(z)
                targets.append(y)
        X = torch.cat(features, dim=0)
        Z = torch.cat(covariates, dim=0)
        y = torch.cat(targets, dim=0)

        # Convert to numpy for glmnet
        X_np = torch.cat([X, Z], dim=1).cpu().numpy()
        y_np = y.cpu().numpy()
        p_fac_np = torch.ones(X_np.shape[1]).cpu().numpy()
        #p_fac_np[X.shape[1]:] = 0.0  # No penalty on covariates

        # Example conversion for X, y, p_fac:
        with localconverter(ro.default_converter + numpy2ri.converter):
            X_r = ro.conversion.py2rpy(X_np)
            y_r = ro.conversion.py2rpy(y_np)
            p_fac_r = ro.conversion.py2rpy(p_fac_np)

        if lam is not None:
            # Fit with fixed lambda ridge, intercept disabled.
            glmnet_fit = glmnet.glmnet(X_r, y_r, alpha=0,
                                       lambda_=ro.FloatVector([lam]),
                                       penalty_factor=p_fac_r,
                                       intercept=False)
            coefs = ro.r['as.matrix'](glmnet.coef_glmnet(glmnet_fit, s=lam))
            best_lambda = lam
        else:
            # Fit with CV ridge, intercept disabled.
            cv_fit = glmnet.cv_glmnet(X_r, y_r, alpha=0,
                                      penalty_factor=p_fac_r,
                                      intercept=False)
            # Extract coefficients (WARNING: Intercept is 0 here).
            coefs = ro.r['as.matrix'](glmnet.coef_cv_glmnet(cv_fit, s="lambda.min"))
            best_lambda = ro.r['as.numeric'](cv_fit.rx2("lambda.min"))[0]

        coefs_np = np.array(coefs).flatten()

        # Assign back to model
        with torch.no_grad():
            self.fx.weight.copy_(torch.tensor(coefs_np[1:X.shape[1]+1], dtype=torch.float32))
            self.fz.weight.copy_(torch.tensor(coefs_np[X.shape[1]+1:], dtype=torch.float32))
        # Save best lambda
        self.lam.copy_(torch.tensor(best_lambda, dtype=torch.float32))


    def _fit_orthogonalization(self, train_dataloader):
        # Fit orthogonalization parameters (coefs) using torch.linagl.lstsq over Z and fx!
        self.eval()
        device = next(self.parameters()).device
        fX = []
        Z = []
        with torch.no_grad():
            for batch in train_dataloader:
                x = batch["X"].to(device)
                z = batch["Z"].to(device)
                x = self.backbone(x)
                x = self.center_x(x)
                fx = self.fx(x)
                z = self.center_z(z)
                fX.append(fx)
                Z.append(z)
            fX = torch.cat(fX, dim=0)
            Z = torch.cat(Z, dim=0)
            # Solve Z * coef = fX in least squares sense.
            self.orth.weight.data = torch.linalg.lstsq(Z, fX).solution

