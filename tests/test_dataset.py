import unittest
import torch
from torchvision import transforms

from cocodeel.dataset import CovarDataset


class TestCovarDataset(unittest.TestCase):

    def setUp(self):
        # Create dummy data: 10 samples, images 3x2x2, 4 covariates, 1 target value
        self.X = torch.arange(120).reshape(10, 3, 2, 2).float()
        self.Z = torch.arange(40).reshape(10, 4).float()
        self.y = torch.arange(10).float()

        # Simple transform: multiply image tensor by 2
        self.transform = transforms.Lambda(lambda x: x * 2)

        self.dataset = CovarDataset(self.X, self.Z, self.y, transform=self.transform)
        self.dataset_no_transform = CovarDataset(self.X, self.Z, self.y, transform=None)

    def test_length(self):
        self.assertEqual(len(self.dataset), 10)

    def test_getitem_keys(self):
        sample = self.dataset[0]
        self.assertIsInstance(sample, dict)
        self.assertIn("X", sample)
        self.assertIn("Z", sample)
        self.assertIn("y", sample)

    def test_getitem_values_correct(self):
        idx = 3
        sample = self.dataset_no_transform[idx]
        torch.testing.assert_close(sample["X"], self.X[idx])
        torch.testing.assert_close(sample["Z"], self.Z[idx])
        self.assertEqual(sample["y"].item(), self.y[idx].item())

    def test_transform_applied(self):
        idx = 0
        sample = self.dataset[idx]
        expected = self.X[idx] * 2
        torch.testing.assert_close(sample["X"], expected)

    def test_transform_none(self):
        idx = 0
        sample = self.dataset_no_transform[idx]
        torch.testing.assert_close(sample["X"], self.X[idx])


if __name__ == "__main__":
    unittest.main()