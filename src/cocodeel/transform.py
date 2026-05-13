import torch
from torch import nn


class Center(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_features))

    def forward(self, x):
        return x - self.mean

    def fit(self, x):
        mean = x.mean(dim=0)
        self.mean.copy_(mean)

    def fit_from_loader(self, dataloader, feature_extractor=nn.Identity(), key="X", device="cpu"):
        self.eval()
        features_list = []
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[key].to(device)
                features = feature_extractor(inputs)
                features_list.append(features.cpu())
        features = torch.cat(features_list, dim=0)
        self.fit(features)


class LinearRegressOut(nn.Module):
    def __init__(self, n_covariates):
        """ Linear regressor to regress out the effect of z from fx. """
        super().__init__()
        self.dz = torch.nn.Linear(n_covariates, 1, bias=False)

    def forward(self, fx, z):
        return fx - self.dz(z)

    def fit(self, fx, z):
        with torch.no_grad():
            solution = torch.linalg.lstsq(z, fx).solution  # shape: [n_covariates, 1]
            self.dz.weight.copy_(solution.T)

    def fit_from_loader(self, dataloader, feature_extractor):
        self.eval()
        device = next(feature_extractor.parameters()).device
        fx_list = []
        z_list = []
        with torch.no_grad():
            for batch in dataloader:
                x, z = batch[0].to(device), batch[1].to(device)
                fx_features = feature_extractor(x)
                fx_list.append(fx_features.cpu())
                z_list.append(z.cpu())
        fx = torch.cat(fx_list, dim=0)
        z = torch.cat(z_list, dim=0)
        self.fit(fx, z)


