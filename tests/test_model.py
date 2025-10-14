import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cocodeel.dataset import CovarDataset
from cocodeel.model import BaseNetwork


class DummyBackbone(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Flatten input to match num_features.
        return x.view(x.size(0), -1)


class TestBaseNetwork(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.out_features = 6
        self.backbone = DummyBackbone
        self.backbone_params = {'out_features': self.out_features}
        self.model = BaseNetwork(self.backbone, self.backbone_params)

    def test_forward_shape(self):
        x = torch.randn(4, 2, 3)  # batch of 4 samples, reshaped to 6 features
        y = self.model(x)
        self.assertEqual(y.shape, (4, 1))

    def test_center_features_updates_mean_and_intercept(self):
        # Create fake data: features will average to 1.0 for each dim
        X = torch.ones(10, 2, 3)  # 10 samples, flatten to shape (10, 6)
        dataset = CovarDataset(X, torch.zeros(10), torch.zeros(10))  # Z,y values don't matter
        loader = DataLoader(dataset, batch_size=5)

        # Before centering.
        intercept_before = self.model.intercept.clone()
        predictions_before = self.model.forward(X)

        # Center the features.
        self.assertTrue(self.model.is_centered is False)
        self.model.center_effects(loader)
        self.assertTrue(self.model.is_centered is True)

        # Check that the mean is approx. 1.0
        self.assertTrue(torch.allclose(self.model.center_x.mean, torch.ones(self.out_features), atol=1e-6))
        # Check that the intercept is updated (from 0 to fx.weight @ mean).
        expected_shift = self.model.fx.weight @ self.model.center_x.mean
        self.assertTrue(torch.allclose(
            self.model.intercept,
            intercept_before + expected_shift,
            atol=1e-6
        ))
        # Check that predictions are not affected by centering.
        predictions_after = self.model.forward(X)
        self.assertTrue(torch.allclose(predictions_before, predictions_after, atol=1e-6))


if __name__ == "__main__":
    unittest.main()