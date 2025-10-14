import unittest
import torch

from torch import nn
from torch.utils.data import DataLoader

from cocodeel.dataset import CovarDataset
from cocodeel.transform import Center, LinearRegressOut


class DummyFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.linear(x)


class TestCenterWithCovarDataset(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.X = torch.randn(100, 4)
        self.Z = torch.randn(100, 4)
        self.y = torch.randn(100, 1)
        self.dataset = CovarDataset(self.X, self.Z, self.y)
        self.dataloader = DataLoader(self.dataset, batch_size=10)
        self.device = torch.device("cpu")
        self.feature_extractor = DummyFeatureExtractor().to(self.device)

    def test_fit_from_loader_on_X(self):
        center = Center(n_features=4).to(self.device)
        center.fit_from_loader(self.dataloader, self.feature_extractor, key="X")

        with torch.no_grad():
            feats = self.feature_extractor(self.X.to(self.device))
            centered = center(feats)
            mean = centered.mean(dim=0)
            self.assertTrue(torch.allclose(mean, torch.zeros_like(mean), atol=1e-4))

    def test_fit_from_loader_on_Z(self):
        center = Center(n_features=4).to(self.device)
        center.fit_from_loader(self.dataloader, self.feature_extractor, key="Z")

        with torch.no_grad():
            feats = self.feature_extractor(self.Z.to(self.device))
            centered = center(feats)
            mean = centered.mean(dim=0)
            self.assertTrue(torch.allclose(mean, torch.zeros_like(mean), atol=1e-4))


class TestLinearRegressOut(unittest.TestCase):

    def test_regressout_removes_linear_component(self):
        torch.manual_seed(0)
        z = torch.randn(100, 3)
        true_w = torch.randn(3, 1)
        fx = z @ true_w + 0.01 * torch.randn(100, 1)  # linear + noise

        ro = LinearRegressOut(n_covariates=3)
        ro.fit(fx, z)
        fx_resid = ro(fx, z)

        corr = (z.T @ fx_resid) / len(z)
        self.assertTrue(torch.allclose(corr, torch.zeros_like(corr), atol=1e-2))

    def test_regressout_exact_solution(self):
        z = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
        w = torch.tensor([[1.], [2.]])
        fx = z @ w  # perfect linear relationship

        ro = LinearRegressOut(n_covariates=2)
        ro.fit(fx, z)
        fx_resid = ro(fx, z)

        self.assertTrue(torch.allclose(fx_resid, torch.zeros_like(fx_resid), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
