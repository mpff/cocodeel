import unittest
import torch

from cocodeel.lightning_model import CovarNeuralNetwork
from cocodeel.lightning_model import PostHocOrthogonalizedModel

from cocodeel.dataset import CovarDataset

class TestCaseBackbone(torch.nn.Module):
    def __init__(self, num_features=16):
        super().__init__()
        self.num_features = num_features

    def forward(self, x):
        return x


class TestCovarNeuralNetwork(unittest.TestCase):
    def setUp(self):
        params = {"backbone": TestCaseBackbone,
                  "loss_func": torch.nn.MSELoss,
                  "output_func": torch.nn.Identity,
                  "num_covars": 2}
        self.batch_size = 4
        self.net = CovarNeuralNetwork(**params)
        self.test_input = torch.randn(self.batch_size, self.net.num_features)
        self.test_covar = torch.randn(self.batch_size, self.net.num_covars)

    @torch.no_grad()
    def test_output(self):
        outputs = self.net(self.test_input, self.test_covar)
        self.assertEqual((self.batch_size, 1), outputs.shape)



class TestPostHocOrthoganlizedModel(unittest.TestCase):
    def setUp(self):
        params = {"backbone": TestCaseBackbone,
                  "loss_func": torch.nn.MSELoss,
                  "output_func": torch.nn.Identity,
                  "num_covars": 2}
        self.size = 4
        self.pretrained_net = CovarNeuralNetwork(**params)
        for param in self.pretrained_net.parameters():
            param.data = torch.nn.parameter.Parameter(torch.ones_like(param))
        # Create dataset.
        self.train_image = torch.randn(self.size, 32, 32)
        self.train_covar = torch.randn(self.size, 2)
        self.train_label = torch.randn(self.size, 1)
        self.train_dataset = CovarDataset(self.train_image, self.train_covar, self.train_label)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset)

    @torch.no_grad()
    def test_train_pho_model(self):
        model = PostHocOrthogonalizedModel(self.pretrained_net, self.train_loader)
        output = model(self.train_image, self.train_covar, self.train_label)
        self.assertEqual((self.size, 1), output.shape)


if __name__ == '__main__':
    unittest.main()
