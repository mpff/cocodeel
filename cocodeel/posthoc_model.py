import copy
import torch
import lightning

class PostHocOrthogonalizedModel(lightning.LightningModule):

    def __init__(self, model, train_dataloader):
        """ Orthogonalizes a pre-trained model over the training data set. """
        super().__init__()
        self.model = copy.deepcopy(model)
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


class PostHocLSModel(lightning.LightningModule):
    def __init__(self, model, train_dataloader, pen_factor = 1e-6, orthogonalize=False):
        """
        Applies LS to re-estimate last layer parameters of a pre-trained model over
        the training data set, using a ridge penalty on the features if pen_factor > 0.
        """
        super().__init__()
        self.model = copy.deepcopy(model)  # Pre-trained CovarNeuralNetwork model
        self.train_dataloader = train_dataloader  # Training data
        self.pen_factor = pen_factor  # Ridge penalty factor
        self.orthogonalize = orthogonalize  # Whether to orthogonalize the features before LS

        # Initialize parameters for LS on struct_predictor and deep_predictor layers
        self.model.struct_predictor.weight.requires_grad = False
        self.model.deep_predictor.weight.requires_grad = False

        # Initialize parameters for orthogonalization.
        self.model.ortho_parameters = torch.nn.Parameter(
            torch.zeros((self.model.num_covars, self.model.num_features)),
            requires_grad=False
        )

        # Estimate orthogonalization of features.
        self._orth_estimation()

        # Re-estimate last layer weights using LS.
        self._ls_estimation()

    def _orth_estimation(self):
        # See Rügamer (2023) ... Algorithm ...
        xtx = torch.zeros((self.model.num_covars, self.model.num_covars), device=self.model.device)
        xth = torch.zeros((self.model.num_covars, self.model.num_features), device=self.model.device)
        for batch in self.train_dataloader:
            u = batch["image"].to(self.model.device)
            x = batch["covar"].to(self.model.device)
            h = self.model.backbone(u)
            xtx += x.T @ x
            xth += x.T @ h
        self.model.ortho_parameters.data = torch.linalg.solve(xtx, xth)

    def _ls_estimation(self):
        """Perform IRLS to re-estimate the weights of struct_predictor and deep_predictor."""

        # Initialize accumulators for least squares
        xtx = torch.zeros((self.model.num_covars, self.model.num_covars), device=self.model.device)
        xtr = torch.zeros((self.model.num_covars, 1), device=self.model.device)
        hth = torch.zeros((self.model.num_features, self.model.num_features), device=self.model.device)
        hty = torch.zeros((self.model.num_features, 1), device=self.model.device)

        with torch.no_grad():
            for batch in self.train_dataloader:
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
        ridge_penality_hth = hth + self.pen_factor * torch.eye(self.model.num_features, device=self.model.device)
        deep_weights = torch.linalg.solve(ridge_penality_hth, hty)
        self.model.deep_predictor.weight.data = deep_weights.T

        with torch.no_grad():
            for batch in self.train_dataloader:
                u = batch["image"].to(self.model.device)
                x = batch["covar"].to(self.model.device)
                y = batch["label"].to(self.model.device).unsqueeze(1)

                # Forward pass to obtain linear predictor and response
                h = self.model.backbone(u)
                if self.orthogonalize:
                    h = h - x @ self.model.ortho_parameters

                # Update weighted matrices
                xtx += x.T @ x
                xtr += x.T @ (y - h @ deep_weights)

        # Solve for struct_predictor weights, guaranteeing invertibility.
        invertible_xtx = xtx + 1e-6 * torch.eye(self.model.num_covars, device=self.model.device)
        struct_weights = torch.linalg.solve(invertible_xtx, xtr)
        self.model.struct_predictor.weight.data = struct_weights.T

    def forward(self, u, x):
        """Modified forward pass using re-estimated weights."""
        h = self.model.backbone(u)
        if self.orthogonalize:
            h = h - x @ self.model.ortho_parameters
        eta = self.model.deep_predictor(h) + self.model.struct_predictor(x)
        return self.model.output_func(eta).squeeze(1)


class PostHocIRLSModel(lightning.LightningModule):
    def __init__(self, model, train_dataloader, max_iters=10, tol=1e-6):
        """
        Applies IRLS to re-estimate last layer parameters of a pre-trained model over
        the training data set for generalized linear models with responses from the exponential family.
        """
        super().__init__()
        self.model = copy.deepcopy(model)  # Pre-trained CovarNeuralNetwork model
        self.train_dataloader = train_dataloader  # Training data
        self.max_iters = max_iters  # Maximum iterations for IRLS
        self.tol = tol  # Tolerance for convergence in IRLS

        # Initialize parameters for IRLS on struct_predictor and deep_predictor layers
        self.model.struct_predictor.weight.requires_grad = False
        self.model.deep_predictor.weight.requires_grad = False

        # Re-estimate last layer weights using IRLS
        self._irls_estimation()

    def _irls_estimation(self):
        """Perform IRLS to re-estimate the weights of struct_predictor and deep_predictor."""
        for iter in range(self.max_iters):
            # Initialize accumulators for weighted least squares
            xtwx = torch.zeros((self.model.num_covars, self.model.num_covars), device=self.model.device)
            xtwz = torch.zeros((self.model.num_covars, 1), device=self.model.device)
            hth = torch.zeros((self.model.num_features, self.model.num_features), device=self.model.device)
            htz = torch.zeros((self.model.num_features, 1), device=self.model.device)

            total_residual = 0

            with torch.no_grad():
                for batch in self.train_dataloader:
                    u = batch["image"].to(self.model.device)
                    x = batch["covar"].to(self.model.device)
                    y = batch["label"].to(self.model.device).unsqueeze(1)

                    # Forward pass to obtain linear predictor and response
                    h = self.model.backbone(u)
                    eta = self.model.deep_predictor(h) + self.model.struct_predictor(x)
                    mu = self.model.output_func(eta)  # Response according to GLM output function

                    # Compute variance and weight for each sample
                    var = self._variance_function(mu)  # General variance function
                    weights = 1 / (var + 1e-6)

                    # Calculate the working response `z`
                    g_prime = self._link_derivative(mu)  # General derivative of the link function
                    z = eta + (y - mu) / (g_prime + 1e-6)  # Working response

                    # Update weighted matrices
                    W = torch.diag(weights.flatten())
                    xtwx += x.T @ W @ x
                    xtwz += x.T @ W @ z
                    hth += h.T @ W @ h
                    htz += h.T @ W @ z

                    # Track the mean absolute residual for convergence check
                    total_residual += torch.sum(torch.abs(y - mu)).item()

            # Solve for struct_predictor weights
            struct_weights = torch.linalg.solve(
                xtwx + 1e-6 * torch.eye(self.model.num_covars, device=self.model.device), xtwz)
            self.model.struct_predictor.weight.data = struct_weights.T

            # Solve for deep_predictor weights
            deep_weights = torch.linalg.solve(hth + 1e-6 * torch.eye(self.model.num_features, device=self.model.device),
                                              htz)
            self.model.deep_predictor.weight.data = deep_weights.T

            # Convergence check
            if total_residual < self.tol:
                break

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
        return self.model.output_func(eta).squeeze(1)
