import torch
import torch.nn as nn

from cocodeel.transform import Center

class BaseNetwork(nn.Module):
    def __init__(self, backbone, backbone_params={}, num_covariates=0):
        """ Base Network class for a model with CENTRED features and INTERCEPT.
        Parameters:
            backbone (nn.Module): the CNN backbone for feature extraction.
            backbone_params (dict): parameters to give to the backbone model.
        Methods:
            forward: defines the forward computation at every call.
            center_effects: centers the features and adjusts the intercept accordingly.
        """
        super(BaseNetwork, self).__init__()
        self.backbone = backbone(**backbone_params)
        self.backbone_params = backbone_params
        self.num_covariates = 0
        # Centering.
        self.center_x = Center(self.backbone.out_features)
        self.center_y = Center(1)
        self.is_centered = False
        # Last layer effects.
        self.fx = nn.Linear(self.backbone.out_features, 1, bias=False)
        self.intercept = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x, z=None):
        return self.intercept + self.predict_fx(x, z)

    def predict_fx(self, x, z=None):
        x = self.backbone(x)
        x = self.center_x(x)
        fx = self.fx(x)
        return fx

    def predict_fz(self, z):
        # return vector of (batch, 1) of zeros
        return torch.zeros(z.shape[0], 1, device=z.device)

    def center_effects(self, dataloader):
        self.eval()
        # Gather all features, covariates and targets.
        X, y = [], []
        device = next(self.parameters()).device
        with torch.no_grad():
            for batch in dataloader:
                Xb, yb = batch["X"].to(device), batch["y"].to(device)
                Xb = self.backbone(Xb)
                X.append(Xb.cpu())
                y.append(yb.cpu())
        X = torch.cat(X, dim=0)
        y = torch.cat(y, dim=0)
        # Fit centering modules.
        self.center_x.fit(X)
        self.center_y.fit(y)
        # Adjust intercept.
        with torch.no_grad():
            self.intercept.data = self.center_y.mean
        self.is_centered = True
        return self


class CovarNetwork(nn.Module):
    def __init__(self, backbone, backbone_params={}, num_covariates=1):
        """ Covariate Network class for a model with CENTRED features, CENTRED covariates and INTERCEPT.
        Parameters:
            backbone (nn.Module): the CNN backbone for feature extraction.
            backbone_params (dict): parameters to give to the backbone model.
            num_covariates (int): number of covariates added in the last layer.
        Methods:
            forward: defines the forward computation at every call.
            center_effects: centers the features and covariates, and adjusts the intercept accordingly.
        """
        super(CovarNetwork, self).__init__()
        self.backbone = backbone(**backbone_params)
        self.backbone_params = backbone_params
        self.num_covariates = num_covariates
        # Centering.
        self.center_x = Center(self.backbone.out_features)  # Assumes backbone has attribute out_features.
        self.center_z = Center(self.num_covariates)
        self.center_y = Center(1)
        self.is_centered = False
        # Last layer effects.
        self.fx = nn.Linear(self.backbone.out_features, out_features=1, bias=False)
        self.fz = nn.Linear(self.num_covariates, out_features=1, bias=False)
        self.intercept = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x, z):
        return self.intercept + self.predict_fx(x, z) + self.predict_fz(z)

    def predict_fx(self, x, z=None):
        x = self.backbone(x)
        x = self.center_x(x)
        fx = self.fx(x)
        return fx

    def predict_fz(self, z):
        z = self.center_z(z)
        fz = self.fz(z)
        return fz

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
        with torch.no_grad():
            self.intercept.data = self.center_y.mean
        self.is_centered = True
        return self