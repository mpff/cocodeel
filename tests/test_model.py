import unittest
import torch

from tests.utils.model import TestCaseBackbone

from cocodeel.model import CovarNeuralNetwork

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



if __name__ == '__main__':
    unittest.main()
