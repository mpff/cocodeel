import torch
import torch.nn as nn
import numpy as np

from cocodeel.model import BaseNetwork
from cocodeel.transform import Center


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
    def fit(self, train_dataloader, val_dataloader, lam=None, max_iters=25, tol=1e-6):
        """Fit post-hoc ridge regression (and optionally orthogonalization)."""
        self.center_effects(train_dataloader)
        self._fit_effects(train_dataloader, val_dataloader, lam=lam, max_iters=max_iters, tol=tol)
        if self.orthogonalize:
            self._fit_orthogonalization(train_dataloader)
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
    def _fit_effects(self, train_loader, val_loader, lam=None, max_iters=50, tol=1e-6, n_lambdas=50):

        # ---- Extract training data ----
        X_train, Z_train, y_train = self._extract_features_from_loader(train_loader)
        X_train = self.center_x(X_train)
        Z_train = self.center_z(Z_train)

        X_std = X_train.std(dim=0, keepdim=True) + 1e-6
        Z_std = Z_train.std(dim=0, keepdim=True) + 1e-6

        X_train = X_train / X_std
        Z_train = Z_train / Z_std
        Zfull_train = torch.cat([Z_train, torch.ones_like(Z_train[:, :1])], dim=1)

        # ---- Extract validation data ----
        X_val, Z_val, y_val = self._extract_features_from_loader(val_loader)
        X_val = self.center_x(X_val) / X_std
        Z_val = self.center_z(Z_val) / Z_std

        # ---- Build lambda path ----
        lambda_max = torch.linalg.norm(X_train, 2)**2
        lambda_min = 1e-4 * lambda_max  # avoid λ → 0 when p ≥ n

        lambda_path = torch.logspace(
            torch.log10(lambda_max),
            torch.log10(lambda_min),
            steps=n_lambdas,
            device=X_train.device,
        )
        if lam is not None:
            lambda_path = torch.tensor([lam], device=X_train.device)

        # ---- Initialize parameters (warm start seed) ----
        self.fx.weight.data.zero_()
        self.fz.weight.data.zero_()
        self.intercept.data.zero_()

        best_val_loss = float("inf")
        best_state = None
        best_lambda = None

        for lam in lambda_path:

            # ---- Train at fixed λ ----
            self._solve_fixed_lambda(
                X_train,
                Zfull_train,
                y_train,
                lam,
                max_iters,
                tol,
            )
            # Update fx, fz weights to account for standardization.
            self.fx.weight.data /= X_std
            self.fz.weight.data /= Z_std

            # ---- Validation evaluation ----
            eta_val = self.intercept + self.fx(X_val) + self.fz(Z_val)

            # Use loss function corresponding to the link function for validation evaluation.
            if self.link == "identity":
                val_loss = torch.nn.MSELoss()(mu_val, y_val)
            elif self.link == "logit":
                val_loss = torch.nn.BCEWithLogitsLoss()(eta_val, y_val)
            else:
                raise ValueError(f"Unsupported link function: {self.link}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_lambda = lam
                best_state = {
                    "fx": self.fx.weight.data.clone(),
                    "fz": self.fz.weight.data.clone(),
                    "intercept": self.intercept.data.clone(),
                }

        # ---- Restore best model ----
        self.fx.weight.data.copy_(best_state["fx"])
        self.fz.weight.data.copy_(best_state["fz"])
        self.intercept.data.copy_(best_state["intercept"])

        self.lam.data.copy_(best_lambda)

        return self

    @torch.no_grad()
    def _solve_fixed_lambda(self, X, Zfull, y, lam, max_iters, tol):

        eps = 1e-5  # Small constant for numerical stability.

        # Initialize old predictors for convergence check.
        fz_old = torch.zeros_like(y)
        fx_old = torch.zeros_like(y)

        # Iteratively reweighted least squares until convergence.
        for i in range(max_iters):
           
            # ---- Prepare reweighted data ----

            eta = self.intercept + self.fx(X) + self.fz(Zfull[:, :-1])
            mu = self.output_func(eta)
            var = self._variance_function(mu)
            g_prime = self._link_derivative(mu)
            weights = g_prime**2 / (var + eps)

            # Clip probabilities for numerical stability in binomial case.
            if self.link == "logit":
                close_to_zero = mu < eps
                close_to_one = mu > 1 - eps
                mu = torch.where(close_to_zero, torch.tensor(0).to(mu.device), mu)
                mu = torch.where(close_to_one, torch.tensor(1).to(mu.device), mu)
                weights = torch.where(close_to_zero | close_to_one, torch.tensor(eps).to(weights.device), weights)
            
            y_work = eta + (y - mu) / (g_prime + eps)

            # Reweight data.
            sqrt_w = torch.sqrt(weights)
            yw = sqrt_w * y_work
            Xw = sqrt_w * X
            Zw = sqrt_w * Zfull 

            # --- Fitting Procedure ---

            # 1. Regress yw on Zw to get residuals.
            resid_y = yw - Zw @ torch.linalg.solve(Zw.T @ Zw + eps * torch.eye(Zw.shape[1]).to(Zw.device), Zw.T @ yw)
        
            # 2. Regress Xw on Zw to get residuals.
            resid_X = Xw - Zw @ torch.linalg.solve(Zw.T @ Zw + eps * torch.eye(Zw.shape[1]).to(Zw.device), Zw.T @ Xw)
            
            beta_fx = torch.linalg.solve(
                resid_X.T @ resid_X + lam * torch.eye(resid_X.shape[1]).to(resid_X.device),
                resid_X.T @ resid_y
            )
            self.fx.weight.data.copy_(beta_fx.view(1, -1))
            
            # 4. Compute fz weights.
            resid_y_new = yw - self.fx(Xw)
            beta_fz = torch.linalg.solve(Zw.T @ Zw + eps * torch.eye(Zw.shape[1]).to(Zw.device), Zw.T @ resid_y_new)
            # Update fz weights (excluding intercept).
            self.fz.weight.data.copy_(beta_fz[:-1].view(1, -1))
            self.intercept.data.copy_(beta_fz[-1].view(1))
            
            # 5. Check convergence via relative change in fx and fz (TODO)
            if self.link == "identity":
                break  # No need for IRLS iterations for identity link
            delta_fx = torch.norm(self.fx(X) - fx_old) / (torch.norm(fx_old) + eps)
            delta_fz = torch.norm(self.fz(Zfull[:, :-1]) - fz_old) / (torch.norm(fz_old) + eps)
            if delta_fx < tol and delta_fz < tol:
                break
            fx_old = self.fx(X).clone()
            fz_old = self.fz(Zfull[:, :-1]).clone()
            if i == max_iters - 1:
                print(f"IRLS did not converge after {max_iters} iterations (delta_fx={delta_fx:.4e}, delta_fz={delta_fz:.4e}, lambda={lam:.4e}). Consider increasing max_iters or tol.")
        
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
        best_lambda = ro.r["as.numeric"](cv_fit.rx2("lambda.min"))[0]
        best_coefs = ro.r["as.numeric"](glmnet.coef_glmnet(cv_fit, s="lambda.min"))
        coefs_np = np.array(best_coefs)
        coefs_tensor = torch.tensor(coefs_np, dtype=torch.float32).to(X.device)
        self.intercept.data = coefs_tensor[0].view(1)
        self.fx.weight.data = coefs_tensor[1:X.shape[1]+1].view(1, -1)
        self.fz.weight.data = coefs_tensor[X.shape[1]+1:].view(1, -1)
        return torch.tensor(best_lambda, dtype=torch.float32).to(X.device)
        
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
