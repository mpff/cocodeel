import torch
import torch.nn as nn

from cocodeel.model import BaseNetwork
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
        self.is_centered = False

    def forward(self, x, z=None):
        return self.intercept + self.predict_fx(x, z)

    def predict_fx(self, x, z=None):
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

