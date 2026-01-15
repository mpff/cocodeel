import torch
import torch.nn as nn

from cocodeel.model import BaseNetwork, CovarNetwork
from cocodeel.transform import Center


class PostHocOrthNetwork(BaseNetwork):
    def __init__(self, model, num_covariates):
        """ Neural Network with CENTRED effects and post-hoc ORTHOGONALIZATION.
        Parameters:
            model (BaseNetwork): Prefitted model with a backbone and fx layer.
            num_covariates (int): The number of covariates used in the last layer.
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

        self.orth = nn.Linear(self.num_covariates, 1, bias=False)
        self.orth.weight.data.fill_(0.0)  # Initialize orthogonalization to 0.

        self.center_z = Center(self.num_covariates)
        self.register_buffer('is_centered', torch.tensor(False))

    def forward(self, x, z):
        return self.intercept + self.predict_fx(x, z)

    def predict_fx(self, x, z):
        x = self.backbone(x)
        x = self.center_x(x)
        z = self.center_z(z)
        fx = self.fx(x) - self.orth(z)
        return fx

    def predict_fz(self, z):
        # return vector of (batch, 1) of zeros
        return torch.zeros(z.shape[0], 1, device=z.device)

    def fit(self, train_dataloader):
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

        # Solve Z * beta = fX
        solution = torch.linalg.lstsq(Z, fX).solution
        self.orth.weight.copy_(solution.T)

        # Update intercept to account for orthogonalization shift
        self.intercept.data -= self.orth(self.center_z.mean)


class SemiStructuredNetwork(CovarNetwork):
    def __init__(self, model):
        """ SSN with CENTRED effects and post-hoc ORTHOGONALIZATION.
        Parameters:
            model (CovarNetwork): Prefitted model with a backbone and fx layer.
        Methods:
            forward: Defines the forward computation at every call.
            _fit: Fits the model to the training data.
        """
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

    def fit(self, train_dataloader):
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
