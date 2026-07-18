"""Cross-fitted ensemble of already-fitted, disjoint-fold RefitCovarNetwork models."""
import torch


class CrossFitEnsemble:
    """K-fold cross-fit ensemble: eta_hat = (1/K) sum_k eta_k, with the link applied once."""

    # The link is never applied per fold: mean_k(g^-1(eta_k)) != g^-1(mean_k(eta_k))
    # for a nonlinear link (Jensen). Call .recenter(loader) before reporting
    # f_X/f_Z so all folds share one reference population; recentering never
    # changes eta or predictions.

    def __init__(self, models):
        self.models = list(models)
        self.num_covariates = self.models[0].num_covariates

    def recenter(self, loader):
        """Recenter every fold onto `loader`'s sample, in place."""
        for m in self.models:
            m.recenter(loader)
        return self

    def predict_eta(self, x, z):
        etas = torch.stack([m.predict_eta(x, z) for m in self.models])
        return etas.mean(dim=0)

    def predict_fx(self, x, z=None):
        fxs = torch.stack([m.predict_fx(x, z) for m in self.models])
        return fxs.mean(dim=0)

    def predict_fz(self, z):
        fzs = torch.stack([m.predict_fz(z) for m in self.models])
        return fzs.mean(dim=0)

    def forward(self, x, z):
        eta = self.predict_eta(x, z)
        return self.models[0].output_func(eta)

    def __call__(self, x, z):
        return self.forward(x, z)

    def eval(self):
        for m in self.models:
            m.eval()
        return self

    def to(self, device):
        self.models = [m.to(device) for m in self.models]
        return self
