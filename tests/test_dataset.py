import unittest
import torch

from cocodeel.dataset import CovarDataset

class TestCovarDataset(unittest.TestCase):

    def setUp(self):
        self.size = 4
        # Create dataset.
        self.test_image = torch.randn(self.size, 32, 32)
        self.test_covar = torch.randn(self.size, 1)
        self.test_label = torch.randn(self.size, 1)
        self.dataset = CovarDataset(self.test_image, self.test_covar, self.test_label)

    @torch.no_grad()
    def test_create_dataset(self):
        # Check dataset output.
        batch = self.dataset[0:self.size]
        self.assertEqual(batch.keys(), {'image', 'covar', 'label'})
        self.assertEqual(batch['image'].shape, self.test_image.shape)
        self.assertEqual(batch['covar'].shape, self.test_covar.shape)
        self.assertEqual(batch['label'].shape, self.test_label.shape)



if __name__ == '__main__':
    unittest.main()
