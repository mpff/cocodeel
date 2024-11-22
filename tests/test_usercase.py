import unittest
import torch
import lightning

from tests.utils.model import TestCaseBackbone

from cocodeel.dataset import CovarDataset
from cocodeel.model import CovarNeuralNetwork
from cocodeel.posthoc_model import PostHocOrthogonalizedModel


class TestUserCase(unittest.TestCase):

    def setUp(self):
        self.N = 1000
        self.inputs = torch.randn(self.N, 32)
        self.covars = torch.randn(self.N, 2)
        # Add vector of ones to covars.
        self.covars = torch.cat((torch.ones(self.N, 1), self.covars), 1)
        # Create label with coefficients [1,1,1]
        self.labels = torch.matmul(self.covars, torch.tensor([1., 1., 1.])) + torch.randn(self.N)
        self.train_set = CovarDataset(self.inputs, self.covars, self.labels)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_set, batch_size=int(self.N/10))

    def test_train_pho_model(self):
        # define model
        params = {"backbone": TestCaseBackbone,
                  "backbone_params": {"num_features": 32},
                  "loss_func": torch.nn.MSELoss,
                  "optimizer": torch.optim.AdamW,
                  "output_func": torch.nn.Identity,
                  "num_covars": 3,
                  "optimizer_params": {"lr": 0.01}}
        net = CovarNeuralNetwork(**params)
        # train model
        trainer = lightning.Trainer(max_epochs=5)
        trainer.fit(net, self.train_loader)
        # check struct part dimensions
        self.assertEqual((1, 3), net.struct_predictor.weight.shape)
        # train post-hoc orthogonalized model
        pho_net = PostHocOrthogonalizedModel(net, self.train_loader)
        # check struct part dimensions are closer to [1, 1, 1]
        self.assertEqual((1, 3), pho_net.model.struct_predictor.weight.shape)
        self.assertLessEqual(
            torch.norm(torch.tensor([1., 1., 1.]) - pho_net.model.struct_predictor.weight.squeeze()),
            torch.norm(torch.tensor([1., 1., 1.]) - net.struct_predictor.weight.squeeze())
        )


class TestBinaryResponse(unittest.TestCase):
    def setUp(self):
        self.N = 1000
        self.inputs = torch.randn(self.N, 32)
        self.covars = torch.randn(self.N, 2)
        self.covars = torch.cat((torch.ones(self.N, 1), self.covars), 1)
        # create binary labels incorporating true coefficient [1, 1, 1]
        self.labels = (torch.matmul(self.covars, torch.tensor([1., 1., 1.])) + torch.randn(self.N) > 0).float()  
        self.train_set = CovarDataset(self.inputs, self.covars, self.labels)
        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_set, batch_size=int(self.N / 10)
        )

    def test_train_pho_model_binary(self):
        # define model
        params = {
            "backbone": TestCaseBackbone,
            "backbone_params": {"num_features": 32},
            "optimizer": torch.optim.AdamW,
            "loss_func": torch.nn.BCEWithLogitsLoss,
            "output_func": torch.nn.Identity,  
            "num_covars": 3,
            "optimizer_params": {"lr": 0.01},
        }
        net = CovarNeuralNetwork(**params)
        # train model
        trainer = lightning.Trainer(max_epochs=5)
        trainer.fit(net, self.train_loader)
        # check struct predictor weight dimensions
        self.assertEqual((1, 3), net.struct_predictor.weight.shape)
        # train post-hoc orthogonalized model
        pho_net = PostHocOrthogonalizedModel(net, self.train_loader)
        # check struct predictor dimensions are closer to [1, 1, 1]
        self.assertEqual((1, 3), pho_net.model.struct_predictor.weight.shape)
        self.assertLessEqual(
            torch.norm(torch.tensor([1., 1., 1.]) - pho_net.model.struct_predictor.weight.squeeze()),
            torch.norm(torch.tensor([1., 1., 1.]) - net.struct_predictor.weight.squeeze())
        )


class TestMulticlassResponse(unittest.TestCase):
    def setUp(self):
        self.N = 1000
        self.inputs = torch.randn(self.N, 32)
        self.covars = torch.randn(self.N, 2)
        self.covars = torch.cat((torch.ones(self.N, 1), self.covars), 1)
        # create multiclass labels incorporating true coefficient [1, 1, 1]
        self.labels = torch.bucketize(torch.matmul(self.covars, torch.tensor([1., 1., 1.])) + 
                                      torch.randn(self.N), boundaries=torch.tensor([-1.0, 1.0])).float()
        self.train_set = CovarDataset(self.inputs, self.covars, self.labels)
        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_set, batch_size=int(self.N / 10)
        )

    def test_train_pho_model_multiclass(self):
        # define model
        params = {
            "backbone": TestCaseBackbone,
            "backbone_params": {"num_features": 32},
            "loss_func": torch.nn.CrossEntropyLoss,
            "optimizer": torch.optim.AdamW,
            "output_func": torch.nn.Identity,  
            "num_covars": 3,
            "optimizer_params": {"lr": 0.01},
        }
        net = CovarNeuralNetwork(**params)
        # train model
        trainer = lightning.Trainer(max_epochs=5)
        trainer.fit(net, self.train_loader)
        # check struct predictor weight dimensions
        self.assertEqual((1, 3), net.struct_predictor.weight.shape)
        # train post-hoc orthogonalized model
        pho_net = PostHocOrthogonalizedModel(net, self.train_loader)
        # check struct predictor dimensions are closer to [1, 1, 1]
        self.assertEqual((1, 3), pho_net.model.struct_predictor.weight.shape)
        self.assertLessEqual(
            torch.norm(torch.tensor([1., 1., 1.]) - pho_net.model.struct_predictor.weight.squeeze()),
            torch.norm(torch.tensor([1., 1., 1.]) - net.struct_predictor.weight.squeeze())
        )


if __name__ == '__main__':
    unittest.main()
