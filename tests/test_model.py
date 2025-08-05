import unittest
import torch

from tests.utils.model import TestCaseBackbone

from cocodeel.model import NeuralNetwork
from cocodeel.model import CovarNeuralNetwork


class TestCovarNeuralNetwork(unittest.TestCase):
    def setUp(self):
        params = {"backbone": TestCaseBackbone,
                  "loss_func": torch.nn.MSELoss,
                  "optimizer": torch.optim.AdamW,
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


class TestBinaryCovarNeuralNetwork(unittest.TestCase):
    def setUp(self):
        params = {"backbone": TestCaseBackbone,
                  "loss_func": torch.nn.BCEWithLogitsLoss,
                  "optimizer": torch.optim.AdamW,
                  "output_func": torch.nn.Identity,
                  "num_covars": 2}
        self.batch_size = 4
        self.net = CovarNeuralNetwork(**params)
        self.test_input = torch.randn(self.batch_size, self.net.num_features)
        self.test_covar = torch.randn(self.batch_size, self.net.num_covars)

    @torch.no_grad()
    def test_binary_response_training(self):
        test_input = torch.randn(self.batch_size, self.net.num_features)
        test_covar = torch.randn(self.batch_size, self.net.num_covars)
        y_binary = torch.randint(0, 2, (self.batch_size, 1)).float() 

        y_pred = self.net(test_input, test_covar)
        self.assertEqual(y_pred.shape, y_binary.shape)

        loss = self.net.loss_func(y_pred, y_binary)
        self.assertGreater(loss.item(), 0)


class TestMultiCLassCovarNeuralNetwork(unittest.TestCase):
    def setUp(self):
        params = {"backbone": TestCaseBackbone,
                  "loss_func": torch.nn.CrossEntropyLoss,
                  "optimizer": torch.optim.AdamW,
                  "output_func": torch.nn.Identity,
                  "num_covars": 2}
        self.batch_size = 4
        self.num_classes = 3
        self.net = CovarNeuralNetwork(**params)
        self.test_input = torch.randn(self.batch_size, self.net.num_features)
        self.test_covar = torch.randn(self.batch_size, self.net.num_covars)

    @torch.no_grad()
    def test_multiclass_response_training(self):
        test_input = torch.randn(self.batch_size, self.net.num_features)
        test_covar = torch.randn(self.batch_size, self.net.num_covars)
        y_multiclass = torch.randint(0, self.num_classes, (self.batch_size,)).float()  

        y_pred = self.net(test_input, test_covar)
        self.assertEqual(y_pred.shape, (self.batch_size, 1))

        loss = self.net.loss_func(y_pred.squeeze(), y_multiclass)
        self.assertGreater(loss.item(), 0)


if __name__ == '__main__':
    unittest.main()
