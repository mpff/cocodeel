import copy
import torch
import lightning

class PostHocOrthogonalizedModel(lightning.LightningModule):

    def __init__(self, model, train_dataloader):
        """ Orthogonalizes a pre-trained model over the training data set. """
        super().__init__()
        self.model = copy.deepcopy(model)
        self.loss_func = model.loss_func
        self.output_func = model.output_func
        # Update last layer with orthogonalization.
        self.model.ortho_parameters = torch.nn.Parameter(torch.zeros((self.model.num_covars, 1)), requires_grad=False)
        with torch.no_grad():
            # See Rügamer (2023) ... Algorithm ...
            xtx = torch.zeros((self.model.num_covars, self.model.num_covars), device=self.model.device)
            xte = torch.zeros((self.model.num_covars, 1), device=self.model.device)
            for batch in train_dataloader:
                u = batch["image"].to(self.model.device)
                x = batch["covar"].to(self.model.device)
                h = self.model.backbone(u)
                eta_h = self.model.deep_predictor(h)
                xtx += x.T @ x
                xte += x.T @ eta_h
            self.model.ortho_parameters.data = torch.linalg.solve(xtx, xte)
            self.model.struct_predictor.weight.data += self.model.ortho_parameters.T

    def forward(self, u, x):
        h = self.model.backbone(u)
        eta_deep = self.model.deep_predictor(h) - x @ self.model.ortho_parameters
        eta_struct = self.model.struct_predictor(x)
        eta = eta_deep + eta_struct
        return self.model.output_func(eta)

    def test_step(self, batch, batch_idx):
        y = batch["label"].to(self.model.device)
        eta = self.predict_struct(batch) + self.predict_deep(batch)
        loss = self.loss_func(eta.squeeze(), y)
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        u, x, y = batch["image"], batch["covar"], batch["label"]
        u, x, y = u.to(self.model.device), x.to(self.model.device), y.to(self.model.device)
        yhat = self(u, x)
        return yhat

    def predict_deep(self, batch, batch_idx=None):
        u, x = batch["image"], batch["covar"]
        u, x = u.to(self.model.device), x.to(self.model.device)
        h = self.model.backbone(u)
        return self.model.deep_predictor(h) - x @ self.model.ortho_parameters

    def predict_struct(self, batch, batch_idx=None):
        x = batch["covar"]
        x = x.to(self.model.device)
        return self.model.struct_predictor(x)

    def struct_coefs(self):
        return self.model.struct_predictor.weight.detach().cpu().numpy().squeeze()


class PostHocLSModel(lightning.LightningModule):
    def __init__(self, model, train_dataloader, num_covars = None, pen_factor = 1e-6, orthogonalize=False):
        """
        Applies LS to re-estimate last layer parameters of a pre-trained model over
        the training data set, using a ridge penalty on the features if pen_factor > 0.
        """
        super().__init__()
        self.model = copy.deepcopy(model)  # Pre-trained CovarNeuralNetwork model
        self.loss_func = model.loss_func
        self.output_func = model.output_func
        self.pen_factor = pen_factor  # Ridge penalty factor
        self.orthogonalize = orthogonalize  # Whether to orthogonalize the features before LS

        # Initialize parameters for LS on struct_predictor and deep_predictor layers
        if num_covars is not None:
            self.model.num_covars = num_covars
        # IMPORTANT: No explicit bias term, as we assume at least a covariate vector of ones!
        self.model.deep_predictor = torch.nn.Linear(self.model.num_features, 1, bias=False)
        self.model.struct_predictor = torch.nn.Linear(self.model.num_covars, 1, bias=False)

        # Initialize parameters for orthogonalization.
        self.model.ortho_parameters = torch.nn.Parameter(
            torch.zeros((self.model.num_covars, self.model.num_features)),
            requires_grad=False
        )

        # Estimate orthogonalization of features.
        self._orth_estimation(train_dataloader)

        # Re-estimate last layer weights using LS.
        self._ls_estimation(train_dataloader)

    def _orth_estimation(self, train_dataloader):
        # See Rügamer (2023) ... Algorithm ...
        xtx = torch.zeros((self.model.num_covars, self.model.num_covars), device=self.model.device)
        xth = torch.zeros((self.model.num_covars, self.model.num_features), device=self.model.device)
        for batch in train_dataloader:
            u = batch["image"].to(self.model.device)
            x = batch["covar"].to(self.model.device)
            h = self.model.backbone(u)
            xtx += x.T @ x
            xth += x.T @ h
        self.model.ortho_parameters.data = torch.linalg.lstsq(xtx, xth).solution

    def _ls_estimation(self, train_dataloader):
        """Perform IRLS to re-estimate the weights of struct_predictor and deep_predictor."""

        # Initialize accumulators for least squares
        xtx = torch.zeros((self.model.num_covars, self.model.num_covars), device=self.model.device)
        xtr = torch.zeros((self.model.num_covars, 1), device=self.model.device)
        hth = torch.zeros((self.model.num_features, self.model.num_features), device=self.model.device)
        hty = torch.zeros((self.model.num_features, 1), device=self.model.device)

        with torch.no_grad():
            for batch in train_dataloader:
                u = batch["image"].to(self.model.device)
                x = batch["covar"].to(self.model.device)
                y = batch["label"].to(self.model.device).unsqueeze(1)

                # Forward pass to obtain linear predictor and response
                h = self.model.backbone(u)
                h_orth = h - x @ self.model.ortho_parameters

                # Update weighted matrices
                hth += h_orth.T @ h_orth
                hty += h_orth.T @ y

        # Solve for deep_predictor weights using the ridge penalty.
        ridge_penalty_hth = hth + self.pen_factor * torch.eye(self.model.num_features, device=self.model.device)
        deep_weights = torch.linalg.lstsq(ridge_penalty_hth, hty).solution
        self.model.deep_predictor.weight.data = deep_weights.T

        with torch.no_grad():
            for batch in train_dataloader:
                u = batch["image"].to(self.model.device)
                x = batch["covar"].to(self.model.device)
                y = batch["label"].to(self.model.device).unsqueeze(1)

                # Forward pass to obtain linear predictor and response
                h = self.model.backbone(u)
                if self.orthogonalize:
                    h = h - x @ self.model.ortho_parameters

                # Update weighted matrices
                xtx += x.T @ x
                xtr += x.T @ (y - self.model.deep_predictor(h))

        # Solve for struct_predictor weights, guaranteeing invertibility.
        #invertible_xtx = xtx + 1e-6 * torch.eye(self.model.num_covars, device=self.model.device)
        struct_weights = torch.linalg.solve(xtx, xtr)
        self.model.struct_predictor.weight.data = struct_weights.T

    def forward(self, u, x):
        """Modified forward pass using re-estimated weights."""
        h = self.model.backbone(u)
        # TODO: Orthogonalization only for deep_predictor?
        if self.orthogonalize:
            h = h - x @ self.model.ortho_parameters
        eta = self.model.deep_predictor(h) + self.model.struct_predictor(x)
        return self.model.output_func(eta)

    def test_step(self, batch, batch_idx):
        y = batch["label"].to(self.model.device)
        eta = self.predict_struct(batch) + self.predict_deep(batch)
        loss = self.loss_func(eta.squeeze(), y)
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        u, x, y = batch["image"], batch["covar"], batch["label"]
        u, x, y = u.to(self.model.device), x.to(self.model.device), y.to(self.model.device)
        yhat = self(u, x)
        return yhat

    def predict_deep(self, batch, batch_idx=None):
        u, x = batch["image"], batch["covar"]
        u, x = u.to(self.model.device), x.to(self.model.device)
        h = self.model.backbone(u)
        if self.orthogonalize:
            h = h - x @ self.model.ortho_parameters
        return self.model.deep_predictor(h)

    def predict_struct(self, batch, batch_idx=None):
        x = batch["covar"]
        x = x.to(self.model.device)
        return self.model.struct_predictor(x)

    def struct_coefs(self):
        return self.model.struct_predictor.weight.detach().cpu().numpy().squeeze()


class PostHocIRLSModel(lightning.LightningModule):
    def __init__(self, model, train_dataloader, num_covars = None, pen_factor = 1e-6, orthogonalize=False, max_iters=25, tol=1e-3):
        """
        Applies IRLS to re-estimate last layer parameters of a pre-trained model over
        the training data set for generalized linear models with responses from the exponential family.
        """
        super().__init__()
        self.to(model.device)
        self.model = copy.deepcopy(model).to(model.device)  # Pre-trained CovarNeuralNetwork model
        self.max_iters = max_iters  # Maximum iterations for IRLS
        self.tol = tol  # Tolerance for convergence in IRLS
        self.loss_func = model.loss_func
        self.output_func = model.output_func
        self.pen_factor = pen_factor  # Ridge penalty factor
        self.orthogonalize = orthogonalize  # Whether to orthogonalize the features before LS

        # Initialize parameters for LS on struct_predictor and deep_predictor layers
        if num_covars is not None:
            self.model.num_covars = num_covars
        # IMPORTANT: No explicit bias term, as we assume at least a covariate vector of ones!
        self.model.deep_predictor = torch.nn.Linear(self.model.num_features, 1, bias=False, device=self.model.device)
        self.model.struct_predictor = torch.nn.Linear(self.model.num_covars, 1, bias=False, device=self.model.device)

        # Initialize parameters for orthogonalization.
        self.model.ortho_parameters = torch.nn.Parameter(
            torch.zeros((self.model.num_covars, self.model.num_features)),
            requires_grad=False
        )

        # Estimate orthogonalization of features.
        if self.orthogonalize:
            self._orth_estimation(train_dataloader)
        # Re-estimate last layer weights using IRLS
        self._irls_estimation(train_dataloader)

    def _orth_estimation(self, data_loader):
        # See Rügamer (2023) ... Algorithm ...
        xtx = torch.zeros((self.model.num_covars, self.model.num_covars), device=self.model.device)
        xth = torch.zeros((self.model.num_covars, self.model.num_features), device=self.model.device)
        for batch in data_loader:
            u = batch["image"].to(self.model.device)
            x = batch["covar"].to(self.model.device)
            h = self.model.backbone(u)
            xtx += x.T @ x
            xth += x.T @ h
        pen_xtx = xtx + self.pen_factor * torch.eye(self.model.num_covars, device=self.model.device)
        self.model.ortho_parameters.data = torch.linalg.lstsq(pen_xtx, xth).solution

    def _irls_orth_estimation(self, data_loader):
        xtwx_orth = torch.zeros((self.model.num_covars, self.model.num_covars), device=self.model.device)
        xtwh_orth = torch.zeros((self.model.num_covars, self.model.num_features), device=self.model.device)
        for batch in data_loader:
            u = batch["image"].to(self.model.device)
            x = batch["covar"].to(self.model.device)
            h = self.model.backbone(u)
            if self.orthogonalize:
                h = h - x @ self.model.ortho_parameters

            # Compute weights for each sample.
            eta = self.model.deep_predictor(h) + self.model.struct_predictor(x)
            mu = self.model.output_func(eta)  # Response according to GLM output function
            var = self._variance_function(mu)  # General variance function
            weights = 1 / (var + 1e-6)
            W = torch.diag(weights.flatten())

            # Update weighted matrices.
            xtwx_orth += x.T @ W @ x
            xtwh_orth += x.T @ W @ h

        pen_xtwx_orth = xtwx_orth + self.pen_factor * torch.eye(self.model.num_covars, device=self.model.device)
        return torch.linalg.lstsq(pen_xtwx_orth, xtwh_orth).solution

    def _irls_estimation(self, data_loader):
        """Perform IRLS to re-estimate the weights of struct_predictor and deep_predictor."""

        # Initialize weights for convergence check.
        struct_weights_old = self.tol * torch.ones((self.model.num_covars, 1), device=self.model.device)
        deep_weights_old = self.tol * torch.ones((self.model.num_features, 1), device=self.model.device)

        for iter in range(self.max_iters):

            # Initialize accumulators for weighted least squares
            xtwx = torch.zeros((self.model.num_covars, self.model.num_covars), device=self.model.device)
            xtwr = torch.zeros((self.model.num_covars, 1), device=self.model.device)
            htwh = torch.zeros((self.model.num_features, self.model.num_features), device=self.model.device)
            htwz = torch.zeros((self.model.num_features, 1), device=self.model.device)

            # Get projection parameters.
            irls_ortho_params = self._irls_orth_estimation(data_loader)

            # Deep weights.
            with torch.no_grad():
                for batch in data_loader:
                    u = batch["image"].to(self.model.device)
                    x = batch["covar"].to(self.model.device)
                    y = batch["label"].to(self.model.device).unsqueeze(1)
                    h = self.model.backbone(u)
                    if self.orthogonalize:
                        h = h - x @ self.model.ortho_parameters
                    # Compute variance and weight for each sample
                    eta = self.model.deep_predictor(h) + self.model.struct_predictor(x)
                    mu = self.model.output_func(eta)  # Response according to GLM output function
                    var = self._variance_function(mu)
                    weights = 1 / (var + 1e-6)
                    W = torch.diag(weights.flatten())
                    # Working response
                    g_prime = self._link_derivative(mu)
                    z = eta + (y - mu) / (g_prime + 1e-6)
                    h_orth = h - x @ irls_ortho_params
                    # Update weighted matrices
                    htwh += h_orth.T @ W @ h_orth
                    htwz += h_orth.T @ W @ z

            # Solve for deep_predictor weights using the ridge penalty.
            ridge_penalty_htwh = htwh + self.pen_factor * torch.eye(self.model.num_features, device=self.model.device)
            deep_weights = torch.linalg.lstsq(ridge_penalty_htwh, htwz).solution
            self.model.deep_predictor.weight.data = deep_weights.T

            # Struct weights.
            with torch.no_grad():
                for batch in data_loader:
                    u = batch["image"].to(self.model.device)
                    x = batch["covar"].to(self.model.device)
                    y = batch["label"].to(self.model.device).unsqueeze(1)
                    h = self.model.backbone(u)
                    if self.orthogonalize:
                        h = h - x @ self.model.ortho_parameters
                    # Compute variance and weight for each sample
                    eta = self.model.deep_predictor(h) + self.model.struct_predictor(x)
                    mu = self.model.output_func(eta)  # Response according to GLM output function
                    var = self._variance_function(mu)
                    weights = 1 / (var + 1e-6)
                    W = torch.diag(weights.flatten())
                    g_prime = self._link_derivative(mu)
                    # Working response.
                    z = eta + (y - mu) / (g_prime + 1e-6)
                    # Update weighted matrices
                    xtwx += x.T @ W @ x
                    xtwr += x.T @ W @ (z - self.model.deep_predictor(h))

            # Solve for struct_predictor weights, guaranteeing invertibility.
            invertible_xtwx = xtwx + 1e-6 * torch.eye(self.model.num_covars, device=self.model.device)
            struct_weights = torch.linalg.lstsq(invertible_xtwx, xtwr).solution
            self.model.struct_predictor.weight.data = struct_weights.T

            # Check whether weights have converged.
            d_struct = torch.norm(self.model.struct_predictor.weight.data - struct_weights_old)/torch.norm(struct_weights_old)
            d_deep = torch.norm(self.model.deep_predictor.weight.data - deep_weights_old)/torch.norm(deep_weights_old)
            if d_struct < self.tol and d_deep < self.tol:
                break
            else:
                struct_weights_old = self.model.struct_predictor.weight.data
                deep_weights_old = self.model.deep_predictor.weight.data

    def _variance_function(self, mu):
        """
        General variance function V(mu) for exponential family.
        Modify this function for different GLM families:
        - Gaussian: V(mu) = 1
        - Binomial: V(mu) = mu * (1 - mu)
        - Poisson: V(mu) = mu
        """
        # Assuming a generic GLM variance function; adapt as needed
        # Here we check the type of link used and return the appropriate variance
        if isinstance(self.model.output_func, torch.nn.Sigmoid):  # Binomial
            return mu * (1 - mu)
        elif isinstance(self.model.output_func, torch.nn.Identity):  # Gaussian
            return torch.ones_like(mu)
        elif isinstance(self.model.output_func, torch.nn.Softplus):  # Poisson/log-link (approximates exp)
            return mu
        else:
            raise NotImplementedError("Variance function not implemented for this output function.")

    def _link_derivative(self, mu):
        """
        Derivative of the link function g' for use in working response calculation.
        Modify for different GLM link functions:
        - Binomial (logit): g'(mu) = mu * (1 - mu)
        - Gaussian (identity): g'(mu) = 1
        - Poisson (log): g'(mu) = 1 / mu
        """
        # Assuming a generic link derivative; adapt as needed
        if isinstance(self.model.output_func, torch.nn.Sigmoid):  # Binomial
            return mu * (1 - mu)
        elif isinstance(self.model.output_func, torch.nn.Identity):  # Gaussian
            return torch.ones_like(mu)
        elif isinstance(self.model.output_func, torch.nn.Softplus):  # Poisson/log-link (approximates exp)
            return 1 / (mu + 1e-6)
        else:
            raise NotImplementedError("Link derivative not implemented for this output function.")

    def forward(self, u, x):
        """Modified forward pass using re-estimated weights."""
        h = self.model.backbone(u)
        if self.orthogonalize:
            h = h - x @ self.model.ortho_parameters
        eta = self.model.deep_predictor(h) + self.model.struct_predictor(x)
        return self.model.output_func(eta)

    def test_step(self, batch, batch_idx):
        y = batch["image"], batch["covar"], batch["label"].to(self.model.device)
        eta = self.predict_struct(batch) + self.predict_deep(batch)
        loss = self.loss_func(eta.squeeze(), y)
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        u, x, y = batch["image"], batch["covar"], batch["label"]
        u, x, y = u.to(self.model.device), x.to(self.model.device), y.to(self.model.device)
        yhat = self(u, x)
        return yhat

    def predict_deep(self, batch, batch_idx=None):
        u, x = batch["image"], batch["covar"]
        u, x = u.to(self.model.device), x.to(self.model.device)
        h = self.model.backbone(u)
        if self.orthogonalize:
            h = h - x @ self.model.ortho_parameters
        return self.model.deep_predictor(h)

    def predict_struct(self, batch, batch_idx=None):
        x = batch["covar"]
        x = x.to(self.model.device)
        return self.model.struct_predictor(x)

    def struct_coefs(self):
        return self.model.struct_predictor.weight.detach().cpu().numpy().squeeze()


class PostHocBFModel(lightning.LightningModule):
    def __init__(self, model, train_dataloader, num_covars = None, pen_factor = 1e-6, orthogonalize=False):
        """
        Applies LS to re-estimate last layer parameters of a pre-trained model over
        the training data set, using a ridge penalty on the features if pen_factor > 0.
        """
        super().__init__()
        self.model = copy.deepcopy(model)  # Pre-trained CovarNeuralNetwork model
        self.loss_func = model.loss_func
        self.output_func = model.output_func
        self.pen_factor = pen_factor  # Ridge penalty factor
        self.orthogonalize = orthogonalize  # Whether to orthogonalize the features before LS

        # Initialize parameters for LS on struct_predictor and deep_predictor layers
        if num_covars is not None:
            self.model.num_covars = num_covars
        # IMPORTANT: No explicit bias term, as we assume at least a covariate vector of ones!
        self.model.deep_predictor = torch.nn.Linear(self.model.num_features, 1, bias=False)
        self.model.struct_predictor = torch.nn.Linear(self.model.num_covars, 1, bias=False)

        # Re-estimate last layer weights using LS.
        self._bf_estimation(train_dataloader)

    def _bf_estimation(self, train_dataloader):
        """Perform IRLS to re-estimate the weights of struct_predictor and deep_predictor."""

        # Initialize accumulators for least squares
        xtx = torch.zeros((self.model.num_covars, self.model.num_covars), device=self.model.device)
        hth = torch.zeros((self.model.num_features, self.model.num_features), device=self.model.device)

        # Convergence parameters.
        converged = False
        tol = 1e-5
        max_iter = 1000
        iter = 0

        while ((not converged) and (iter < max_iter)):
            iter += 1
            print(f"\t ----- STARTING BACKFITTING ITERATION {iter} ------")

            # Initialize accumulators for least squares.
            xtr = torch.zeros((self.model.num_covars, 1), device=self.model.device)
            htr = torch.zeros((self.model.num_features, 1), device=self.model.device)

            with torch.no_grad():
                for batch in train_dataloader:
                    # Batch data.
                    u = batch["image"].to(self.model.device)
                    x = batch["covar"].to(self.model.device)
                    y = batch["label"].to(self.model.device).unsqueeze(1)

                    # Current prediction based on u.
                    h = self.model.backbone(u)
                    eta_h = self.model.deep_predictor(h)

                    # Get xtr.
                    xtr += x.T @ (y - eta_h)

                    # Get xtx (only once!)
                    if iter == 1:
                        xtx += x.T @ x

                # Update covariate effect.
                struct_weights_new = torch.linalg.solve(xtx, xtr)

                for batch in train_dataloader:
                    # Batch data.
                    u = batch["image"].to(self.model.device)
                    x = batch["covar"].to(self.model.device)
                    y = batch["label"].to(self.model.device).unsqueeze(1)

                    # Current prediction based on x.
                    eta_x = x @ struct_weights_new

                    # Get htr.
                    h = self.model.backbone(u)
                    htr += h.T @ (y - eta_x)

                    # Get hth (only once!)
                    if iter == 1:
                        hth += h.T @ h

                # Update image effect.
                deep_weights_new = torch.linalg.lstsq(hth, htr).solution

                # Check convergence.
                delta_eta_h = 0
                delta_eta_x = 0
                for batch in train_dataloader:
                    u = batch["image"].to(self.model.device)
                    x = batch["covar"].to(self.model.device)
                    h = self.model.backbone(u)
                    # Estimate change in effects.
                    eta_h_old = self.model.deep_predictor(h)
                    eta_x_old = self.model.struct_predictor(x)
                    eta_h_new = h @ deep_weights_new
                    eta_x_new = x @ struct_weights_new
                    delta_eta_h += torch.sum((eta_h_new - eta_h_old)**2)
                    delta_eta_x += torch.sum((eta_x_new - eta_x_old)**2)
                # Average over batches.
                delta_eta_h = delta_eta_x/len(train_dataloader)
                delta_eta_x = delta_eta_h/len(train_dataloader)
                # Finally, check convergence.
                converged = (delta_eta_h < tol) and (delta_eta_x < tol)

                # Update weights.
                self.model.struct_predictor.weight.data = struct_weights_new.T
                self.model.deep_predictor.weight.data = deep_weights_new.T


    def forward(self, u, x):
        """Modified forward pass using re-estimated weights."""
        h = self.model.backbone(u)
        eta = self.model.deep_predictor(h) + self.model.struct_predictor(x)
        return self.model.output_func(eta)

    def test_step(self, batch, batch_idx):
        u, x, y = batch["image"], batch["covar"], batch["label"]
        yhat = self(u, x)
        loss = self.model.loss_func(yhat.squeeze(), y)
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        u, x, y = batch["image"], batch["covar"], batch["label"]
        yhat = self(u, x)
        return yhat

    def predict_deep(self, batch, batch_idx=None):
        u, x = batch["image"], batch["covar"]
        h = self.model.backbone(u)
        return self.model.deep_predictor(h)

    def predict_struct(self, batch, batch_idx=None):
        x = batch["covar"]
        return self.model.struct_predictor(x)

    def struct_coefs(self):
        return self.model.struct_predictor.weight.detach().numpy().squeeze()


class PostHocTEModel(lightning.LightningModule):
    def __init__(self, model, train_dataloader, num_covars = None, pen_factor = 1e-6, orthogonalize=False):
        """
        Applies LS to re-estimate last layer parameters of a pre-trained model over
        the training data set, using a ridge penalty on the features if pen_factor > 0.
        """
        super().__init__()
        self.model = copy.deepcopy(model)  # Pre-trained CovarNeuralNetwork model
        self.loss_func = model.loss_func
        self.output_func = model.output_func
        self.pen_factor = pen_factor  # Ridge penalty factor
        self.orthogonalize = orthogonalize  # Whether to orthogonalize the features before LS

        # Initialize parameters for LS on struct_predictor and deep_predictor layers
        if num_covars is not None:
            self.model.num_covars = num_covars
        # IMPORTANT: No explicit bias term, as we assume at least a covariate vector of ones!
        self.model.deep_predictor = torch.nn.Linear(self.model.num_features, 1, bias=False)
        self.model.struct_predictor = torch.nn.Linear(self.model.num_covars, 1, bias=False)

        # Initialize parameters for orthogonalization.
        self.model.ortho_parameters = torch.nn.Parameter(
            torch.zeros((self.model.num_covars, self.model.num_features)),
            requires_grad=False
        )
        self.model.ortho_parameters_reverse = torch.nn.Parameter(
            torch.zeros((self.model.num_features, self.model.num_covars)),
            requires_grad=False
        )

        # Estimate orthogonalization of features.
        self._orth_estimation(train_dataloader)

        # Re-estimate last layer weights using LS.
        self._ls_estimation(train_dataloader)

    def _orth_estimation(self, train_dataloader):
        # See Rügamer (2023) ... Algorithm ...
        xtx = torch.zeros((self.model.num_covars, self.model.num_covars), device=self.model.device)
        xth = torch.zeros((self.model.num_covars, self.model.num_features), device=self.model.device)
        hth = torch.zeros((self.model.num_features, self.model.num_features), device=self.model.device)
        for batch in train_dataloader:
            u = batch["image"].to(self.model.device)
            x = batch["covar"].to(self.model.device)
            h = self.model.backbone(u)
            xtx += x.T @ x
            xth += x.T @ h
            hth += h.T @ h
        self.model.ortho_parameters.data = torch.linalg.lstsq(xtx, xth).solution
        self.model.ortho_parameters_reverse.data = torch.linalg.lstsq(hth, xth.T).solution

    def _ls_estimation(self, train_dataloader):
        """Perform IRLS to re-estimate the weights of struct_predictor and deep_predictor."""

        # Initialize accumulators for least squares
        xtx = torch.zeros((self.model.num_covars, self.model.num_covars), device=self.model.device)
        xtr = torch.zeros((self.model.num_covars, 1), device=self.model.device)
        hth = torch.zeros((self.model.num_features, self.model.num_features), device=self.model.device)
        hty = torch.zeros((self.model.num_features, 1), device=self.model.device)

        with torch.no_grad():
            for batch in train_dataloader:
                u = batch["image"].to(self.model.device)
                x = batch["covar"].to(self.model.device)
                y = batch["label"].to(self.model.device).unsqueeze(1)

                # Forward pass to obtain linear predictor and response
                h = self.model.backbone(u)
                h_orth = h - x @ self.model.ortho_parameters

                # Update weighted matrices
                hth += h.T @ h
                hty += h_orth.T @ y

        # Solve for deep_predictor weights using the ridge penalty.
        ridge_penalty_hth = hth + self.pen_factor * torch.eye(self.model.num_features, device=self.model.device)
        deep_weights = torch.linalg.lstsq(ridge_penalty_hth, hty).solution
        self.model.deep_predictor.weight.data = deep_weights.T

        with torch.no_grad():
            for batch in train_dataloader:
                u = batch["image"].to(self.model.device)
                x = batch["covar"].to(self.model.device)
                y = batch["label"].to(self.model.device).unsqueeze(1)

                # Forward pass to obtain linear predictor and response
                h = self.model.backbone(u)
                #if self.orthogonalize:
                #    h = h - x @ self.model.ortho_parameters

                x_orth = x - h @ self.model.ortho_parameters_reverse

                # Update weighted matrices
                xtx += x.T @ x
                xtr += x.T @ y #- self.model.deep_predictor(h))

        # Solve for struct_predictor weights, guaranteeing invertibility.
        #invertible_xtx = xtx + 1e-6 * torch.eye(self.model.num_covars, device=self.model.device)
        struct_weights = torch.linalg.solve(xtx, xtr)
        self.model.struct_predictor.weight.data = struct_weights.T

    def forward(self, u, x):
        """Modified forward pass using re-estimated weights."""
        h = self.model.backbone(u)
        # TODO: Orthogonalization only for deep_predictor?
        if self.orthogonalize:
            h = h - x @ self.model.ortho_parameters
        eta = self.model.deep_predictor(h) + self.model.struct_predictor(x)
        return self.model.output_func(eta)

    def test_step(self, batch, batch_idx):
        u, x, y = batch["image"], batch["covar"], batch["label"]
        yhat = self(u, x)
        loss = self.model.loss_func(yhat.squeeze(), y)
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        u, x, y = batch["image"], batch["covar"], batch["label"]
        yhat = self(u, x)
        return yhat

    def predict_deep(self, batch, batch_idx=None):
        u, x = batch["image"], batch["covar"]
        h = self.model.backbone(u)
        if self.orthogonalize:
            h = h - x @ self.model.ortho_parameters
        return self.model.deep_predictor(h)

    def predict_struct(self, batch, batch_idx=None):
        x = batch["covar"]
        return self.model.struct_predictor(x)

    def struct_coefs(self):
        return self.model.struct_predictor.weight.detach().numpy().squeeze()


class PostHocProjModel(lightning.LightningModule):
    def __init__(self, model, train_dataloader, num_covars = None, pen_factor = 1e-6, orthogonalize=False, max_iters=10, tol=1e-3):
        """
        Applies IRLS to re-estimate last layer parameters of a pre-trained model over
        the training data set for generalized linear models with responses from the exponential family.
        Uses the projection approach from the paper.
        """
        super().__init__()
        self.to(model.device)
        self.model = copy.deepcopy(model).to(model.device)  # Pre-trained CovarNeuralNetwork model
        self.max_iters = max_iters  # Maximum iterations for IRLS
        self.tol = tol  # Tolerance for convergence in IRLS
        self.loss_func = model.loss_func
        self.output_func = model.output_func
        self.pen_factor = pen_factor  # Ridge penalty factor
        self.orthogonalize = orthogonalize  # Whether to orthogonalize the features before LS

        # Initialize parameters for LS on struct_predictor and deep_predictor layers
        if num_covars is not None:
            self.model.num_covars = num_covars
        # IMPORTANT: No explicit bias term, as we assume at least a covariate vector of ones!
        self.model.deep_predictor = torch.nn.Linear(self.model.num_features, 1, bias=False, device=self.model.device)
        self.model.struct_predictor = torch.nn.Linear(self.model.num_covars, 1, bias=False, device=self.model.device)

        # Initialize parameters for orthogonalization.
        self.model.ortho_parameters = torch.nn.Parameter(
            torch.zeros((self.model.num_covars, self.model.num_features)),
            requires_grad=False
        )

        with torch.no_grad():
            # Estimate orthogonalization of features.
            if self.orthogonalize:
                self._orth_estimation(train_dataloader)
            # Re-estimate last layer weights using IRLS
            self._irls_estimation(train_dataloader)

    def _orth_estimation(self, data_loader):
        # See Rügamer (2023) ... Algorithm ...
        xtx = torch.zeros((self.model.num_covars, self.model.num_covars), device=self.model.device)
        xth = torch.zeros((self.model.num_covars, self.model.num_features), device=self.model.device)
        for batch in data_loader:
            u = batch["image"].to(self.model.device)
            x = batch["covar"].to(self.model.device)
            h = self.model.backbone(u)
            xtx += x.T @ x
            xth += x.T @ h
        pen_xtx = xtx + self.pen_factor * torch.eye(self.model.num_covars, device=self.model.device)
        self.model.ortho_parameters.data = torch.linalg.lstsq(pen_xtx, xth).solution

    def _irls_orth_estimation(self, data_loader):
        xtwx_orth = torch.zeros((self.model.num_covars, self.model.num_covars), device=self.model.device)
        xtwh_orth = torch.zeros((self.model.num_covars, self.model.num_features), device=self.model.device)
        for batch in data_loader:
            u = batch["image"].to(self.model.device)
            x = batch["covar"].to(self.model.device)
            h = self.model.backbone(u)
            if self.orthogonalize:
                h = h - x @ self.model.ortho_parameters

            # Compute weights for each sample.
            eta = self.model.deep_predictor(h) + self.model.struct_predictor(x)
            mu = self.model.output_func(eta)  # Response according to GLM output function
            var = self._variance_function(mu)  # General variance function
            weights = 1 / (var + 1e-6)
            W = torch.diag(weights.flatten())

            # Update weighted matrices.
            xtwx_orth += x.T @ W @ x
            xtwh_orth += x.T @ W @ h

        pen_xtwx_orth = xtwx_orth + self.pen_factor * torch.eye(self.model.num_covars, device=self.model.device)
        return torch.linalg.lstsq(pen_xtwx_orth, xtwh_orth).solution

    def _irls_estimation(self, data_loader):
        """Perform IRLS to re-estimate the weights of struct_predictor and deep_predictor."""

        # Initialize weights for convergence check.
        struct_weights_old = self.tol * torch.ones((self.model.num_covars, 1), device=self.model.device)
        deep_weights_old = self.tol * torch.ones((self.model.num_features, 1), device=self.model.device)

        for iter in range(self.max_iters):

            # Initialize accumulators for weighted least squares
            xtwx = torch.zeros((self.model.num_covars, self.model.num_covars), device=self.model.device)
            xtwr = torch.zeros((self.model.num_covars, 1), device=self.model.device)
            htwh = torch.zeros((self.model.num_features, self.model.num_features), device=self.model.device)
            htwz = torch.zeros((self.model.num_features, 1), device=self.model.device)

            # Get projection parameters.
            #irls_ortho_params = self._irls_orth_estimation(data_loader)

            # Struct weights.
            with torch.no_grad():
                for batch in data_loader:
                    u = batch["image"].to(self.model.device)
                    x = batch["covar"].to(self.model.device)
                    y = batch["label"].to(self.model.device).unsqueeze(1)
                    h = self.model.backbone(u)
                    if self.orthogonalize:
                        h = h - x @ self.model.ortho_parameters
                    # Compute variance and weight for each sample
                    eta = self.model.deep_predictor(h) + self.model.struct_predictor(x)
                    mu = self.model.output_func(eta)  # Response according to GLM output function
                    var = self._variance_function(mu)
                    weights = 1 / (var + 1e-6)
                    W = torch.diag(weights.flatten())
                    g_prime = self._link_derivative(mu)
                    # Working response.
                    z = eta + (y - mu) / (g_prime + 1e-6)
                    # Update weighted matrices. Total effect!
                    xtwx += x.T @ W @ x
                    xtwr += x.T @ W @ z

            # Solve for struct_predictor weights, guaranteeing invertibility.
            invertible_xtwx = xtwx + 1e-6 * torch.eye(self.model.num_covars, device=self.model.device)
            struct_weights = torch.linalg.lstsq(invertible_xtwx, xtwr).solution


            # Deep weights.
            with torch.no_grad():
                for batch in data_loader:
                    u = batch["image"].to(self.model.device)
                    x = batch["covar"].to(self.model.device)
                    y = batch["label"].to(self.model.device).unsqueeze(1)
                    h = self.model.backbone(u)
                    if self.orthogonalize:
                        h = h - x @ self.model.ortho_parameters
                    # Compute variance and weight for each sample
                    eta = self.model.deep_predictor(h) + self.model.struct_predictor(x)
                    mu = self.model.output_func(eta)  # Response according to GLM output function
                    var = self._variance_function(mu)
                    weights = 1 / (var + 1e-6)
                    W = torch.diag(weights.flatten())
                    g_prime = self._link_derivative(mu)
                    # Working response.
                    z = eta + (y - mu) / (g_prime + 1e-6)
                    #h_orth = h - x @ irls_ortho_params
                    # Update weighted matrices
                    htwh += h.T @ W @ h
                    htwz += h.T @ W @ (z - self.model.struct_predictor(x))

            # Solve for deep_predictor weights using the ridge penalty.
            ridge_penalty_htwh = htwh + self.pen_factor * torch.eye(self.model.num_features, device=self.model.device)
            deep_weights = torch.linalg.lstsq(ridge_penalty_htwh, htwz).solution

            # Update weights.
            self.model.struct_predictor.weight.data = struct_weights.T
            self.model.deep_predictor.weight.data = deep_weights.T

            # Check whether weights have converged.
            d_struct = torch.norm(self.model.struct_predictor.weight.data - struct_weights_old)/torch.norm(struct_weights_old)
            d_deep = torch.norm(self.model.deep_predictor.weight.data - deep_weights_old)/torch.norm(deep_weights_old)
            if d_struct < self.tol and d_deep < self.tol:
                break
            else:
                struct_weights_old = self.model.struct_predictor.weight.data
                deep_weights_old = self.model.deep_predictor.weight.data

    def _variance_function(self, mu):
        """
        General variance function V(mu) for exponential family.
        Modify this function for different GLM families:
        - Gaussian: V(mu) = 1
        - Binomial: V(mu) = mu * (1 - mu)
        - Poisson: V(mu) = mu
        """
        # Assuming a generic GLM variance function; adapt as needed
        # Here we check the type of link used and return the appropriate variance
        if isinstance(self.model.output_func, torch.nn.Sigmoid):  # Binomial
            return mu * (1 - mu)
        elif isinstance(self.model.output_func, torch.nn.Identity):  # Gaussian
            return torch.ones_like(mu)
        elif isinstance(self.model.output_func, torch.nn.Softplus):  # Poisson/log-link (approximates exp)
            return mu
        else:
            raise NotImplementedError("Variance function not implemented for this output function.")

    def _link_derivative(self, mu):
        """
        Derivative of the link function g' for use in working response calculation.
        Modify for different GLM link functions:
        - Binomial (logit): g'(mu) = mu * (1 - mu)
        - Gaussian (identity): g'(mu) = 1
        - Poisson (log): g'(mu) = 1 / mu
        """
        # Assuming a generic link derivative; adapt as needed
        if isinstance(self.model.output_func, torch.nn.Sigmoid):  # Binomial
            return mu * (1 - mu)
        elif isinstance(self.model.output_func, torch.nn.Identity):  # Gaussian
            return torch.ones_like(mu)
        elif isinstance(self.model.output_func, torch.nn.Softplus):  # Poisson/log-link (approximates exp)
            return 1 / (mu + 1e-6)
        else:
            raise NotImplementedError("Link derivative not implemented for this output function.")

    def forward(self, u, x):
        """Modified forward pass using re-estimated weights."""
        h = self.model.backbone(u)
        eta = self.model.deep_predictor(h) + self.model.struct_predictor(x)
        return self.model.output_func(eta)

    def test_step(self, batch, batch_idx):
        y = batch["image"], batch["covar"], batch["label"].to(self.model.device)
        eta = self.predict_struct(batch) + self.predict_deep(batch)
        loss = self.loss_func(eta.squeeze(), y)
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        u, x, y = batch["image"], batch["covar"], batch["label"]
        u, x, y = u.to(self.model.device), x.to(self.model.device), y.to(self.model.device)
        yhat = self(u, x)
        return yhat

    def predict_deep(self, batch, batch_idx=None):
        u, x = batch["image"], batch["covar"]
        u, x = u.to(self.model.device), x.to(self.model.device)
        h = self.model.backbone(u)
        return self.model.deep_predictor(h)

    def predict_struct(self, batch, batch_idx=None):
        x = batch["covar"]
        x = x.to(self.model.device)
        return self.model.struct_predictor(x)

    def struct_coefs(self):
        return self.model.struct_predictor.weight.detach().cpu().numpy().squeeze()
