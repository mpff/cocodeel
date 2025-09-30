import torch
import torch.nn as nn

from cocodeel.model import BaseNetwork
from cocodeel.transform import Center

class PostHocLinearCovarNetwork(BaseNetwork):
    def __init__(self, model, num_covariates):
        """ Post-hoc fitted Neural Network with CENTRED effects and LINEAR covariate effects.
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

        self.fz = nn.Linear(self.num_covariates, 1, bias=False)
        self.center_z = Center(self.num_covariates)

        # # Ensure features and covariates are centered
        # if not self.is_centered:
        #     self.center_features(train_dataloader)
        # self.center_z.fit_from_loader(train_dataloader, nn.Identity(), key='Z', device=next(self.parameters()).device)
        # self.is_centered = True
        #
        # # Fit fx and fz using least squares
        # self._linear_fit_from_loader(train_dataloader)

    def forward(self, x, z):
        x = self.backbone(x)
        x = self.center_x(x)
        z = self.center_z(z)
        y = self.intercept +  self.fx(x) + self.fz(z)
        return y

    def fit(self, dataloader):
        # Ensure features and covariates are centered
        if not self.is_centered:
            self.center_features(dataloader)
        self.center_z.fit_from_loader(dataloader, nn.Identity(), key='Z', device=next(self.parameters()).device)
        self.is_centered = True

        # Fit fx and fz using least squares
        self._linear_fit_from_loader(dataloader)

    def _linear_fit(self, X, Z, y):
        """
        Fit fx and fz weights using least squares:
        Solve: y = β0 + X β_X + Z β_Z
        Assumes X and Z are already centered. y may be centered or not.
        """

        # Demean y.
        y_mean = y.mean()
        y_tilde = y - y_mean

        # Estimate fx weights via Frisch-Waugh-Lovell theorem:
        # 1. Regress y_tilde on Z to get residuals.
        resid_y = y_tilde - Z @ torch.linalg.lstsq(Z, y_tilde).solution
        # 2. Regress X on Z to get residuals.
        resid_X = X - Z @ torch.linalg.lstsq(Z, X).solution
        # 3. Fit fx on residuals.
        beta_X = torch.linalg.lstsq(resid_X, resid_y).solution

        # Estimate fz weights by LS fit on new residuals:
        # 1. Remove fx from y_tilde.
        resid_y = y_tilde - X @ beta_X
        # 2. Fit fz on residuals.
        beta_Z = torch.linalg.lstsq(Z, resid_y).solution

        # Update weights.
        self.fx.weight.data.copy_(beta_X.view(1, -1))
        self.fz.weight.data.copy_(beta_Z.view(1, -1))

        # Store intercept
        self.intercept.data.copy_(y_mean)

    def _linear_fit_from_loader(self, dataloader):
        device = next(self.parameters()).device
        features = []
        covariates = []
        targets = []

        with torch.no_grad():
            for batch in dataloader:
                x = batch["X"].to(device)
                z = batch["Z"].to(device)
                y = batch["y"].to(device)

                x_feat = self.backbone(x)
                x_feat = self.center_x(x_feat)
                z = self.center_z(z)

                features.append(x_feat)
                covariates.append(z)
                targets.append(y)

        X = torch.cat(features, dim=0)
        Z = torch.cat(covariates, dim=0)
        y = torch.cat(targets, dim=0)

        self._linear_fit(X, Z, y)