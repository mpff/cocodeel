import unittest
import torch

from torch import nn
from torch.utils.data import DataLoader, Dataset

from cocodeel.transform import Center
from cocodeel.dataset import CovarDataset
from cocodeel.model import BaseNetwork
from cocodeel.posthoc_model import PostHocCovarNetwork



class DummyBackbone(nn.Module):
    def __init__(self, out_features=2):
        super().__init__()
        self.linear = nn.Linear(3, out_features)
        self.out_features = out_features

    def forward(self, x):
        return self.linear(x)

class TestPostHocLinearCovarNetwork(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.n = 100
        self.num_features = 2
        self.num_covariates = 1

        self.X = torch.randn(self.n, 3)
        self.Z = torch.randn(self.n, self.num_covariates)
        self.y = 2 * self.X[:, 0] + 3 * self.Z[:, 0] + 1.5  # Known linear relation
        self.y = self.y.unsqueeze(1)  # (n, 1)

        self.dataset = CovarDataset(self.X, self.Z, self.y)
        self.dataloader = DataLoader(self.dataset, batch_size=32)

        # Pre-train BaseNetwork (to simulate backbone + fx weights)
        self.base_model = BaseNetwork(backbone=DummyBackbone)

        # Set backbone weights to simulate pre-training.
        # Only X[0] gets a weight of 1, rest is zeroed out.
        with torch.no_grad():
            self.base_model.backbone.linear.weight[0, 0] = 1.0
            self.base_model.backbone.linear.weight[0, 1:] = torch.zeros(1, self.num_features - 1)
            self.base_model.backbone.linear.bias[0] = 0.0

    def test_posthoc_fit_close_to_true_weights(self):
        model = PostHocCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
            train_dataloader=self.dataloader
        )

        # Check that simulation_images is close to expected linear form
        with torch.no_grad():
            x = self.X
            z = self.Z
            pred = model(x, z).squeeze()

        true = self.y.squeeze()
        error = (pred - true).abs().mean()
        self.assertTrue(torch.allclose(error, torch.zeros_like(error), atol=1e-5))

    def test_output_shape(self):
        model = PostHocCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
            train_dataloader=self.dataloader
        )
        with torch.no_grad():
            out = model(self.X, self.Z)
        self.assertEqual(out.shape, (self.n, 1))

    def test_features_are_centered(self):
        model = PostHocCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
            train_dataloader=self.dataloader
        )
        with torch.no_grad():
            features = model.backbone(self.X)
            centered_features = model.center_x(features)
            mean = centered_features.mean(dim=0)
            self.assertTrue(torch.allclose(mean, torch.zeros_like(mean), atol=1e-5))

    def test_covariates_are_centered(self):
        model = PostHocCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
            train_dataloader=self.dataloader
        )
        with torch.no_grad():
            centered_z = model.center_z(self.Z)
            mean = centered_z.mean(dim=0)
            self.assertTrue(torch.allclose(mean, torch.zeros_like(mean), atol=1e-5))

    def test_intercept_is_mean_of_y(self):
        model = PostHocCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
            train_dataloader=self.dataloader
        )
        y_mean = self.y.mean().item()
        intercept = model.intercept.item()
        self.assertAlmostEqual(intercept, y_mean, places=5)


if __name__ == '__main__':
    unittest.main()
