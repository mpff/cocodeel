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
    def fit(self, train_dataloader, val_dataloader, lam=None, max_iters=50, tol=1e-2, penalty_z=None, n_lambdas=100):
        """Fit post-hoc ridge regression (and optionally orthogonalization).

        Args:
            penalty_z: Optional fixed penalty matrix for fz, shape (num_covariates, num_covariates).
                Build on the *standardized* Z scale (after center_z and Z_std division).
                _solve_fixed_lambda receives already-standardized Z, so penalty_z is applied
                in that scale directly.
                _fit_orthogonalization still operates on the centered (not standardized) scale.
                Typical use: P-spline roughness penalty for spline-expanded covariates.
        """
        self.center_effects(train_dataloader)
        self._fit_effects(train_dataloader, val_dataloader, lam=lam, max_iters=max_iters, tol=tol, penalty_z=penalty_z, n_lambdas=n_lambdas)
        if self.orthogonalize:
            self._fit_orthogonalization(train_dataloader, penalty_z=penalty_z)
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
    def _fit_effects(self, train_loader, val_loader,
        lam, max_iters, tol, n_lambdas=100, max_expansions=6, penalty_z=None):

        self.max_iters_ = max_iters
        self.tol_ = tol
        self.n_lambdas_ = n_lambdas

        # ---- Extract training data ----
        X_train, Z_train, y_train = self._extract_features_from_loader(train_loader)
        X_train = self.center_x(X_train)
        Z_train = self.center_z(Z_train)

        X_std = X_train.std(dim=0, keepdim=True) + 1e-6
        Z_std = Z_train.std(dim=0, keepdim=True) + 1e-6

        X_train = X_train / X_std
        Z_train = Z_train / Z_std

        # ---- Extract validation data ----
        # Keep X_val and Z_val on the *centered-raw* scale: fx.weight and
        # fz.weight are de-standardised before each val eval (see below),
        # so val inputs must match the raw scale the weights now expect.
        # Standardising Z_val (as a stray line once did) biases val loss
        # by 1/Z_std and distorts lambda selection — especially for
        # orthogonalisation, where a wrong lambda gives a wrong fx.weight
        # which then miscalibrates the orth OLS fit.
        X_val, Z_val, y_val = self._extract_features_from_loader(val_loader)
        X_val = self.center_x(X_val)
        Z_val = self.center_z(Z_val)

        # ---- Build lambda path ---- (see glmnet paper)
        lambda_max = self._get_lambda_max(X_train, Z_train, y_train)
        if X_train.shape[0] < X_train.shape[1]:
            lambda_min = 1e-3 * lambda_max  # More aggressive regularization for high-dimensional case.
        else:
            lambda_min = 1e-6 * lambda_max  # Default glmnet choice for low-dimensional case.

        records = []

        for expansion_idx in range(max_expansions):

            lambda_path = torch.logspace(
                torch.log10(lambda_max),
                torch.log10(lambda_min),
                steps=n_lambdas,
                device=X_train.device,
            )
            if lam is not None:
                lambda_path = torch.tensor([lam], device=X_train.device)
            else:
                print("Lambda max: {:.4e}, Lambda min: {:.4e}".format(lambda_max.item(), lambda_min.item()))

            # ---- Initialize parameters ----
            self.fx.weight.data.zero_()
            self.fz.weight.data.zero_()
            self.intercept.data = self._link_function(self.center_y.mean).view(1)  # mean defined as shape (1,)!

            best_val_loss = float("inf")
            best_val_loss_any = float("inf")  # fallback: best val loss ignoring convergence
            best_state = None
            best_state_any = None
            best_lambda = None
            best_lambda_any = None

            for lambd in lambda_path:

                # ---- Train at fixed λ ----
                diag = self._solve_fixed_lambda(
                    X_train,
                    Z_train,
                    y_train,
                    lambd,
                    max_iters,
                    tol,
                    penalty_z=penalty_z,
                )
                # Update fx/fz weights to account for X/Z standardization.
                self.fx.weight.data /= X_std
                self.fz.weight.data /= Z_std

                beta_fx_norm = float(self.fx.weight.data.norm())
                beta_fz_norm = float(self.fz.weight.data.norm())

                # ---- Validation evaluation ----
                eta_val = self.intercept + self.fx(X_val) + self.fz(Z_val)

                # Use loss function corresponding to the link function for validation evaluation.
                if self.link == "identity":
                    val_loss = torch.nn.MSELoss()(eta_val, y_val)
                elif self.link == "logit":
                    val_loss = torch.nn.BCEWithLogitsLoss()(eta_val, y_val)
                else:
                    raise ValueError(f"Unsupported link function: {self.link}")

                records.append({
                    "expansion":    expansion_idx,
                    "lambda":       float(lambd),
                    "val_loss":     float(val_loss),
                    "converged":    diag["converged"],
                    "n_iters":      diag["n_iters"],
                    "delta_fx":     diag["delta_fx"],
                    "delta_fz":     diag["delta_fz"],
                    "beta_fx_norm": beta_fx_norm,
                    "beta_fz_norm": beta_fz_norm,
                })

                if val_loss < best_val_loss and diag["converged"] == True:
                    best_val_loss = val_loss
                    best_lambda = lambd
                    best_state = {
                        "fx": self.fx.weight.data.clone(),
                        "fz": self.fz.weight.data.clone(),
                        "intercept": self.intercept.data.clone(),
                    }

                # Unconditional fallback — used if no lambda converges.
                if val_loss < best_val_loss_any:
                    best_val_loss_any = val_loss
                    best_lambda_any = lambd
                    best_state_any = {
                        "fx": self.fx.weight.data.clone(),
                        "fz": self.fz.weight.data.clone(),
                        "intercept": self.intercept.data.clone(),
                    }

                self.fx.weight.data.zero_()
                self.fz.weight.data.zero_()
                self.intercept.data = self._link_function(self.center_y.mean).view(1)  # mean defined as shape (1,)!

            # If no lambda converged, fall back to best unconverged state.
            if best_state is None:
                print("Warning: no converged solution found across lambda path — using best unconverged state.")
                best_state = best_state_any
                best_lambda = best_lambda_any

            # check if best lambda is at the edge of the path, if so expand the path and repeat.
            if len(lambda_path) == 1:
                break
            if best_lambda == lambda_path[0]:
                lambda_min = lambda_max + 1e-4 * lambda_max
                lambda_max *= 100
                print("Expanding lambda path: new lambda max = {:.4e}".format(lambda_max.item()))
            elif best_lambda == lambda_path[-1]:
                lambda_max = lambda_min - 1e-4 * lambda_min
                lambda_min /= 100
                print("Expanding lambda path: new lambda min = {:.4e}".format(lambda_min.item()))
            else:
                break

        # ---- Restore best model ----
        self.fx.weight.data.copy_(best_state["fx"])
        self.fz.weight.data.copy_(best_state["fz"])
        self.intercept.data.copy_(best_state["intercept"])

        self.lam.data.copy_(best_lambda)
        self.lambda_path_ = records
        print("Best lambda: {:.4e}".format(best_lambda.item()))

        return self

    @torch.no_grad()
    def _get_lambda_max(self, X, Z, y):
        """Compute lambda_max for ridge regression."""
        # See glmnet paper for details on lambda path calculation.
        # N * alpha * lambda_max = max_p |X_j'y| with centered X and y.
        # for ridge: alpha = 0.001 in lambda max calculation.
        N = X.shape[0]
        y = self.center_y(y)
        X = self.center_x(X)
        alpha = 0.001
        lambda_max = torch.max(torch.abs(X.T @ y)) / N / alpha
        # if self.link == "identity":
        #     lambda_max = 1 * torch.linalg.norm(X, 2)**2
        # elif self.link == "logit":
        #     eps = 1e-5  # Small constant for numerical stability.
        #     mu = self.center_y.mean
        #     var = self._variance_function(mu)
        #     g_prime = self._link_derivative(mu)
        #     weights = g_prime**2 / (var + eps)
        #     # Clip probabilities for numerical stability in binomial case.
        #     close_to_zero = mu < eps
        #     close_to_one = mu > 1 - eps
        #     mu = torch.where(close_to_zero, torch.tensor(0).to(mu.device), mu)
        #     mu = torch.where(close_to_one, torch.tensor(1).to(mu.device), mu)
        #     weights = torch.where(close_to_zero | close_to_one, torch.tensor(eps).to(weights.device), weights)
        #     # Get lambda_max using the weighted design matrix.
        #     sqrt_weights = torch.sqrt(weights)
        #     lambda_max = 10 * torch.linalg.norm(sqrt_weights * X, 2)**2
        # else:
        #     raise ValueError(f"Unsupported link function: {self.link}")        
        return lambda_max

    @torch.no_grad()
    def _solve_fixed_lambda(self, X, Z, y, lam, max_iters, tol, penalty_z=None):

        eps = 1e-5  # Small constant for numerical stability.

        # Initialize old coefficients for convergence check.
        fx_weight_old = self.fx.weight.data.clone()
        fz_weight_old = self.fz.weight.data.clone()

        delta_fx = 0.0
        delta_fz = 0.0
        converged = False

        # Penalty matrix in centered Z scale (fixed across IRLS iterations).
        P_z = penalty_z.to(X.device) if penalty_z is not None \
            else torch.zeros(Z.shape[1], Z.shape[1], device=X.device)

        # Iteratively reweighted least squares until convergence.
        for i in range(max_iters):

            # ---- Prepare reweighted data ----

            eta = self.intercept + self.fx(X) + self.fz(Z)
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
            sqrt_weights = torch.sqrt(weights)

            y_work = eta + (y - mu) / (g_prime + eps)

            # Update intercept.
            self.intercept.data = y_work.mean().view(1)

            # Centered reweighting. (Note: not equal to demeaning in case of weights!)
            yw = sqrt_weights * (y_work - (weights * y_work).sum() / weights.sum())
            Xw = sqrt_weights * (X - (weights * X).sum(dim=0, keepdim=True) / weights.sum())
            Zw = sqrt_weights * (Z - (weights * Z).sum(dim=0, keepdim=True) / weights.sum())

            # --- Fitting Procedure ---

            # 1. Regress yw on Zw to get residuals.
            resid_y = yw - Zw @ torch.linalg.solve(Zw.T @ Zw + P_z + eps * torch.eye(Zw.shape[1]).to(Zw.device), Zw.T @ yw)

            # 2. Regress Xw on Zw to get residuals.
            resid_X = Xw - Zw @ torch.linalg.solve(Zw.T @ Zw + P_z + eps * torch.eye(Zw.shape[1]).to(Zw.device), Zw.T @ Xw)

            beta_fx = torch.linalg.solve(
                resid_X.T @ resid_X + lam * torch.eye(resid_X.shape[1]).to(resid_X.device),
                resid_X.T @ resid_y
            )
            self.fx.weight.data.copy_(beta_fx.view(1, -1))

            # 4. Compute fz weights.
            resid_y_new = yw - self.fx(Xw)
            beta_fz = torch.linalg.solve(Zw.T @ Zw + P_z + eps * torch.eye(Zw.shape[1]).to(Zw.device), Zw.T @ resid_y_new)
            # Update fz weights (excluding intercept).
            self.fz.weight.data.copy_(beta_fz.view(1, -1))

            # 5. Check convergence via relative change in coefficients.
            if self.link == "identity":
                converged = True
                break  # No need for IRLS iterations for identity link
            delta_fx = torch.norm(self.fx.weight.data - fx_weight_old) / (torch.norm(fx_weight_old) + eps)
            delta_fz = torch.norm(self.fz.weight.data - fz_weight_old) / (torch.norm(fz_weight_old) + eps)
            if delta_fx < tol and delta_fz < tol:
                converged = True
                break
            fx_weight_old = self.fx.weight.data.clone()
            fz_weight_old = self.fz.weight.data.clone()
            if i == max_iters - 1:
                print(f"IRLS did not converge after {max_iters} iterations (delta_fx={delta_fx:.4e}, delta_fz={delta_fz:.4e}, lambda={lam:.4e}). Consider increasing max_iters or tol.")

        return {
            "converged": converged,
            "n_iters":   i + 1,
            "delta_fx":  float(delta_fx),
            "delta_fz":  float(delta_fz),
        }
        
    @torch.no_grad()
    def _fit_orthogonalization(self, dataloader, penalty_z=None):
        """Fit linear orthogonalization term via least squares on (Z, fX).
        penalty_z is on the centered Z scale (no Z_std correction needed here since
        orthogonalization operates directly on center_z(Z), not standardised Z).
        """
        self.eval()
        X, Z, _ = self._extract_features_from_loader(dataloader)

        X = self.center_x(X)
        Z = self.center_z(Z)
        fX = self.fx(X)

        eps = 1e-5
        P = penalty_z.to(Z.device) if penalty_z is not None \
            else torch.zeros(Z.shape[1], Z.shape[1], device=Z.device)
        solution = torch.linalg.solve(Z.T @ Z + P + eps * torch.eye(Z.shape[1], device=Z.device), Z.T @ fX)
        self.orth.weight.copy_(solution.T)
        return self
