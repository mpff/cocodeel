import unittest
import torch

from tests.utils.model import TestCaseBackbone

from cocodeel.dataset import CovarDataset
from cocodeel.model import CovarNeuralNetwork
from cocodeel.posthoc_model import PostHocOrthogonalizedModel


class TestPostHocOrthoganlizedModel(unittest.TestCase):
    def setUp(self):
        params = {"backbone": TestCaseBackbone,
                  "loss_func": torch.nn.MSELoss,
                  "optimizer": torch.optim.AdamW,
                  "output_func": torch.nn.Identity,
                  "num_features": 32,
                  "num_covars": 2}
        self.size = 4
        self.pretrained_net = CovarNeuralNetwork(**params)
        for param in self.pretrained_net.parameters():
            param.data = torch.nn.parameter.Parameter(torch.ones_like(param))
        # Create dataset.
        self.train_image = torch.randn(self.size, 32)
        self.train_covar = torch.randn(self.size, 2)
        self.train_label = torch.randn(self.size, 1)
        self.train_dataset = CovarDataset(self.train_image, self.train_covar, self.train_label)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset)

    @torch.no_grad()
    def test_train_pho_model(self):
        model = PostHocOrthogonalizedModel(self.pretrained_net, self.train_loader)
        output = model(self.train_image, self.train_covar)
        self.assertEqual((self.size, 1), output.shape)


if __name__ == '__main__':
    unittest.main()
