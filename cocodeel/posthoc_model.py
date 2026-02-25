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
        self.fz.weight.data.zero_()

        # Optional orthogonalization on fx. (I - Z(Z'Z)^{-1}Z'X)
        self.orth = nn.Linear(num_covariates, 1, bias=False)
        self.orth.weight.data.zero_()
        # Optional IWLS-weighted orthogonalization on phi(X). (I - Z(Z'WZ)^{-1}Z'WX).
        self.worth = nn.Linear(num_covariates, self.backbone.out_features, bias=False)
        self.worth.weight.data.zero_()

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
    def fit(self, dataloader, lam=None, max_iters=25, tol=1e-6):
        """Fit post-hoc ridge regression (and optionally orthogonalization)."""
        self.center_effects(dataloader)
        self._fit_effects(dataloader, lam=lam, max_iters=max_iters, tol=tol)
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

        # NOTE: Hacky centering of intercept if fx already centered.
        # TODO: better way to handle this?
        if self.is_centered:
            self.intercept.data += self.fz(self.center_z.mean)
        else:
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
        
        # Rescale X and Z to have unit variance (helps with stability).
        Xc = self.center_x(X)
        Zc = self.center_z(Z)
        X_std = Xc.std(dim=0, keepdim=True).clamp_min(1e-8)
        Z_std = Zc.std(dim=0, keepdim=True).clamp_min(1e-8)
        Xnorm = Xc / X_std
        Znorm = Zc / Z_std

        # Preperation
        Zfull = torch.hstack([Znorm, torch.ones((Znorm.shape[0], 1)).to(Znorm.device)])
        Xorth = Xnorm - Zfull @ torch.linalg.solve(Zfull.T @ Zfull, Zfull.T @ Xnorm)
        
        # Initialize intercept as mean of y, fx and fz as zero. This is a common starting point for IRLS.
        eta_mean = self.center_y.mean
        self.intercept.data = self._link_function(eta_mean).view(1)  # mean defined as shape (1,)!
        self.fx.weight.data.zero_()
        self.fz.weight.data.zero_()  # Zero from initialization (relevant only when refit a second time).
        # Alternative: Update old fx weights to account for rescaling.
        #self.fx.weight.data = self.fx.weight.data * X_std.view(-1)
        # TODO: better initialization using a first glmnet fit without backfitting?
        #self._linear_glmnet_fit(Xnorm, Znorm, y)

        # Initialize old predictors for convergence check.
        fz_old = torch.zeros_like(y)
        fx_old = torch.zeros_like(y)

        # Iteratively reweighted least squares until convergence.
        for i in range(max_iters):

            # Test using glmnet directly!

            # # Concatenate X and Z
            # X_np = np.hstack([Xnorm.cpu().numpy(), Znorm.cpu().numpy()])
            # y_np = y.cpu().numpy()
            # n, p_total = X_np.shape
            # # Define penalty factors: 1 for X, 0 for Z
            # penalty_factor = np.array([1]*X.shape[1] + [0]*Z.shape[1])
            # with localconverter(ro.default_converter + numpy2ri.converter):
            #     X_r = ro.conversion.py2rpy(X_np)
            #     y_r = ro.conversion.py2rpy(y_np)
            #     pf_r = ro.conversion.py2rpy(penalty_factor)
            # # 3b. Fit ridge regression in R (glmnet), either with given lam or CV.
            # cv_fit = glmnet.cv_glmnet(
            #     X_r, y_r,
            #     family="binomial",
            #     alpha=0,
            #     intercept=True,
            #     penalty_factor=pf_r,
            #     standardize=False  # We already standardized X and Z ourselves.
            # )
            # best_lambda = ro.r["as.numeric"](cv_fit.rx2("lambda.min"))[0]
            # self.lam.copy_(torch.tensor(best_lambda, dtype=torch.float32))
            # best_coefs = ro.r["as.matrix"](glmnet.coef_glmnet(cv_fit, s="lambda.min"))
            # coefs_np = np.array(best_coefs)
            # coefs_tensor = torch.tensor(coefs_np, dtype=torch.float32).to(X.device)
            # self.intercept.data = coefs_tensor[0].view(1)
            # self.fx.weight.data = coefs_tensor[1:X.shape[1]+1].view(1, -1)
            # self.fz.weight.data = coefs_tensor[X.shape[1]+1:].view(1, -1)
            # break  # Skip IRLS iterations since glmnet already does the fitting with the correct weights!

            
            # ---- Prepare reweighted data ----

            # Get current predictions, weight matrix, and working response.
            eta = self.intercept + self.fx(Xorth) + self.fz(Znorm)
            mu = self.output_func(eta)
            var = self._variance_function(mu)
            g_prime = self._link_derivative(mu)
            weights = g_prime**2 / (var + 1e-8)
            # Set any mu within 1e-5 of 0 or 1 to 0 or 1. Set weights to 1e-5 in these cases to avoid instability.
            close_to_zero = mu < 1e-5
            close_to_one = mu > 1 - 1e-5
            mu = torch.where(close_to_zero, torch.tensor(0).to(mu.device), mu)
            mu = torch.where(close_to_one, torch.tensor(1).to(mu.device), mu)
            weights = torch.where(close_to_zero | close_to_one, torch.tensor(1e-5).to(weights.device), weights)
            # Compute working responses and reweighting matrices.
            W = torch.diag(weights.flatten())
            y_work = eta + (y - mu) / (g_prime + 1e-8)

            # Update intercept.
            #self.intercept.data = y_work.mean().view(1)

            # Centered reweighting. (Note: not equal to demeaning in case of weights!)
            #yw = torch.sqrt(W) @ (y_work - (W @ y_work).sum() / weights.sum())
            #Xw = torch.sqrt(W) @ (Xnorm - (W @ Xnorm).sum(dim=0, keepdim=True) / weights.sum())
            #Zw = torch.sqrt(W) @ (Znorm - (W @ Znorm).sum(dim=0, keepdim=True) / weights.sum())
            yw = torch.sqrt(W) @ y_work
            Zw = torch.sqrt(W) @ Zfull
            Xorth = Xnorm - Zfull @ torch.linalg.solve(Zw.T @ Zw, Zfull.T @ W @ Xnorm)
            Xw = torch.sqrt(W) @ Xorth  # Residualize X against Z for stability.

            # --- Fitting Procedure ---

            # 1. Regress yw on Zw to get residuals.
            resid_y = yw - Zw @ torch.linalg.solve(Zw.T @ Zw, Zw.T @ yw)
            resid_y = yw
        
            # 2. Regress Xw on Zw to get residuals.
            resid_X = Xw - Zw @ torch.linalg.solve(Zw.T @ Zw, Zw.T @ Xw)
            resid_X = Xw
            
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
                    w_r = ro.conversion.py2rpy(weights.cpu().numpy())
                # 3b. Fit ridge regression in R (glmnet), either with given lam or CV.
                cv_fit = glmnet.cv_glmnet(
                    X_r, y_r,
                    alpha=0,
                    intercept=True,
                    #weights=w_r,
                )
                best_lambda = ro.r["as.numeric"](cv_fit.rx2("lambda.min"))[0]
                self.lam.copy_(torch.tensor(best_lambda, dtype=torch.float32))
            else:
                self.lam.copy_(torch.tensor(lam, dtype=torch.float32))
            # 3c. Refit ridge regression with best lam using torch.
            beta_fx = torch.linalg.solve(
                resid_X.T @ resid_X + self.lam.item() * torch.eye(resid_X.shape[1]).to(resid_X.device),
                resid_X.T @ resid_y
            )
            self.fx.weight.data.copy_(beta_fx.view(1, -1))
            
            # 4. Compute fz weights.
            resid_y_new = yw - self.fx(Xorth)
            beta_fz = torch.linalg.solve(Zw.T @ Zw, Zw.T @ yw)
            # Update fz weights (excluding intercept).
            self.fz.weight.data.copy_(beta_fz[:-1].view(1, -1))
            # Update intercept.
            self.intercept.data.copy_(beta_fz[-1].view(1))
            
            # 5. Check convergence via relative change in fx and fz (TODO)
            if self.link == "identity":
                break  # No need for IRLS iterations for identity link
            delta_fx = torch.norm(self.fx(Xnorm) - fx_old) / (torch.norm(fx_old) + 1e-8)
            delta_fz = torch.norm(self.fz(Znorm) - fz_old) / (torch.norm(fz_old) + 1e-8)
            if delta_fx < tol and delta_fz < tol:
                break
            fx_old = self.fx(Xnorm).clone()
            fz_old = self.fz(Znorm).clone()
            if i == max_iters - 1:
                print(f"IRLS did not converge after {max_iters} iterations (delta_fx={delta_fx:.4e}, delta_fz={delta_fz:.4e}). Consider increasing max_iters or tol.")
        
        # Unscale fx and fz weights to account for initial standardization.
        self.fx.weight.data = self.fx.weight.data / X_std.view(-1)
        self.fz.weight.data = self.fz.weight.data / Z_std.view(-1)

        # Final intercept correction to account for centering and rescaling.
        self.intercept.data += self.fx(self.center_x.mean) + self.fz(self.center_z.mean)

        return self
    
    @torch.no_grad()
    def _linear_glmnet_fit(self, X, Z, y):
        """Initial linear fit of fx and fz via least squares glmnet on (X, Z, y)."""
        # Prepare data for R.
        X_np = X.cpu().numpy()
        Z_np = Z.cpu().numpy()
        y_np = y.cpu().numpy()
        XZ_np = np.hstack([X_np, Z_np])
        with localconverter(ro.default_converter + numpy2ri.converter):
            XZ_r = ro.conversion.py2rpy(XZ_np)
            y_r = ro.conversion.py2rpy(y_np)
        # Fit linear model in R (glmnet).
        cv_fit = glmnet.cv_glmnet(
            XZ_r, y_r,
            alpha=0,
            penalty_factor=ro.FloatVector([1.0]*X.shape[1] + [0.0]*Z.shape[1]),
            intercept=True
        )
        best_coefs = ro.r["as.numeric"](glmnet.coef_glmnet(cv_fit, s="lambda.min"))
        coefs_np = np.array(best_coefs)
        coefs_tensor = torch.tensor(coefs_np, dtype=torch.float32).to(X.device)
        self.intercept.data = coefs_tensor[0].view(1)
        self.fx.weight.data = coefs_tensor[1:X.shape[1]+1].view(1, -1)
        self.fz.weight.data = coefs_tensor[X.shape[1]+1:].view(1, -1)
        return self
        
    @torch.no_grad()
    def _fit_orthogonalization(self, dataloader):
        """Fit linear orthogonalization term via least squares on (Z, fX)."""
        self.eval()
        X, Z, _ = self._extract_features_from_loader(dataloader)

        X = self.center_x(X)
        Z = self.center_z(Z)
        fX = self.fx(X)

        # Get IWLS weights for orthogonalization fit.
        eta = self.intercept + self.fx(X) + self.fz(Z)
        mu = self.output_func(eta)
        var = self._variance_function(mu)
        g_prime = self._link_derivative(mu)
        weights = g_prime**2 / (var + 1e-8)
        close_to_zero = mu < 1e-5
        close_to_one = mu > 1 - 1e-5
        mu = torch.where(close_to_zero, torch.tensor(0).to(mu.device), mu)
        mu = torch.where(close_to_one, torch.tensor(1).to(mu.device), mu)
        weights = torch.where(close_to_zero | close_to_one, torch.tensor(1e-5).to(weights.device), weights)
        W = torch.diag(weights.flatten())

        # Solve Z * beta = fX
        solution = torch.linalg.solve(Z.T @ W @ Z, Z.T @ W @ fX)
        self.orth.weight.copy_(solution.T)
