import unittest
import torch


from torch import nn
from torch.utils.data import DataLoader

from cocodeel.dataset import CovarDataset
from cocodeel.model import BaseNetwork, CovarNetwork
from cocodeel.trainer import covar_trainer
from cocodeel.benchmarking.posthoc_model import PostHocOrthNetwork, SemiStructuredNetwork
from tests.conftest import DummyBackbone


class TestLinearBenchmarks(unittest.TestCase):
 
    @torch.no_grad()
    def setUp(self):
        torch.manual_seed(42)
        self.n = 300
        self.out_features = 3
        self.num_covariates = 2
        self.model_params = {
            'link': 'identity',
            'backbone': DummyBackbone,
            'backbone_params': {'in_features': 3, 'out_features': self.out_features},
            'num_covariates': self.num_covariates
            }
        self.loss_fn = nn.MSELoss()
        # Data
        self.X = torch.randn(self.n, 3)
        self.Z = torch.randn(self.n, self.num_covariates)
        self.y = 2 * self.X[:, 0] + 3 * self.Z[:, 0] + 1.5  # Known linear relation
        self.y = self.y.unsqueeze(1)  # (n, 1)
        # Split into train and val
        self.train_loader = DataLoader(
            CovarDataset(self.X[:140], self.Z[:140], self.y[:140]),
            batch_size=20,
            shuffle=True
        )
        self.val_loader = DataLoader(
            CovarDataset(self.X[140:200], self.Z[140:200], self.y[140:200]),
            batch_size=20,
            shuffle=False
        )
        self.test_loader = DataLoader(
            CovarDataset(self.X[200:], self.Z[200:], self.y[200:]),
            batch_size=20,
            shuffle=False
        )

    def test_train_pho_web_model(self):
        # Fit Base model.
        base_model = covar_trainer(
            model=BaseNetwork,
            model_params=self.model_params,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device='cpu',
            loss_fn=self.loss_fn,
            epochs=2,
            lr=0.01
        ).eval()
        model = PostHocOrthNetwork(
            model=base_model,    
            num_covariates=self.num_covariates
        )
        # Before fitting.
        intercept = model.intercept.item()
        self.assertEqual(model.is_centered, False)
        self.assertTrue(torch.allclose(model.orth.weight.data, torch.zeros(self.num_covariates, 1), atol=1e-6))
        # Fit Web PHO.
        model.fit(self.train_loader)
        self.assertEqual(model.is_centered, True)
        preds = model(self.X, self.Z)
        self.assertEqual(preds.shape, self.y.shape)
        self.assertAlmostEqual(
            model.intercept.item(),
            intercept + model.fx(model.center_x.mean).item() - model.orth(model.center_z.mean).item(),
            places=6)

    def test_train_linear_covar_model(self):
        # Fit Post-hoc Covariate model.
        base_model = covar_trainer(
            model=CovarNetwork,
            model_params=self.model_params,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device='cpu',
            loss_fn=self.loss_fn,
            epochs=2,
            lr=0.01
        ).eval()
        model = SemiStructuredNetwork(
            model=base_model
        )
        # Before fitting.
        preds_before = model(self.X, self.Z)
        intercept = model.intercept.item()
        self.assertEqual(model.is_centered, False)
        self.assertTrue(torch.allclose(model.orth.weight.data, torch.zeros(self.num_covariates, 1), atol=1e-6))
        # Fit SSN.
        model.fit(self.train_loader)
        self.assertEqual(model.is_centered, True)
        preds = model(self.X, self.Z)
        self.assertEqual(preds.shape, self.y.shape)
        self.assertTrue(torch.allclose(preds_before, preds, atol=1e-4))
        self.assertAlmostEqual(
            model.intercept.item(),
            intercept + model.fx(model.center_x.mean).item() + model.fz(model.center_z.mean).item(),
            places=6)



class TestLogisticBenchmarks(unittest.TestCase):
  
    @torch.no_grad()
    def setUp(self):
        torch.manual_seed(42)
        self.n = 1000
        self.out_features = 3
        self.num_covariates = 2
        self.model_params = {
            'link': 'logit',
            'backbone': DummyBackbone,
            'backbone_params': {'in_features': 3, 'out_features': self.out_features},
            'num_covariates': self.num_covariates
            }
        self.loss_fn = nn.BCEWithLogitsLoss()
        # Data
        self.X = torch.randn(self.n, 3)
        self.Z = torch.randn(self.n, self.num_covariates)
        self.eta = 2 * self.X[:, 0] + 3 * self.Z[:, 0] + 1.5  # Known linear relation
        self.p = torch.sigmoid(self.eta.unsqueeze(1))  # (n, 1)
        self.y = torch.bernoulli(self.p) # (n, 1)
        # Split into train and val
        self.train_loader = DataLoader(
            CovarDataset(self.X[:600], self.Z[:600], self.y[:600]),
            batch_size=20,
            shuffle=True
        )
        self.val_loader = DataLoader(
            CovarDataset(self.X[600:800], self.Z[600:800], self.y[600:800]),
            batch_size=20,
            shuffle=False
        )
        self.test_loader = DataLoader(
            CovarDataset(self.X[800:], self.Z[800:], self.y[800:]),
            batch_size=20,
            shuffle=False
        )

    def test_train_pho_web_model(self):
        # Fit Base model.
        base_model = covar_trainer(
            model=BaseNetwork,
            model_params=self.model_params,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device='cpu',
            loss_fn=self.loss_fn,
            epochs=2,
            lr=0.01
        )
        model = PostHocOrthNetwork(
            model=base_model,    
            num_covariates=self.num_covariates
        )
        # Before fitting.
        intercept = model.intercept.item()
        self.assertEqual(model.is_centered, False)
        self.assertTrue(torch.allclose(model.orth.weight.data, torch.zeros(self.num_covariates, 1), atol=1e-6))
        # Fit Web PHO.
        model.fit(self.train_loader)
        self.assertEqual(model.is_centered, True)
        preds = model(self.X, self.Z)
        self.assertEqual(preds.shape, self.y.shape)
        self.assertAlmostEqual(
            model.intercept.item(),
            intercept + model.fx(model.center_x.mean).item() - model.orth(model.center_z.mean).item(),
            places=6)

    def test_train_linear_covar_model(self):
        # Fit Post-hoc Covariate model.
        base_model = covar_trainer(
            model=CovarNetwork,
            model_params=self.model_params,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device='cpu',
            loss_fn=self.loss_fn,
            epochs=2,
            lr=0.01
        )
        model = SemiStructuredNetwork(
            model=base_model
        )
        # Before fitting.
        preds_before = model(self.X, self.Z)
        intercept = model.intercept.item()
        self.assertEqual(model.is_centered, False)
        self.assertTrue(torch.allclose(model.orth.weight.data, torch.zeros(self.num_covariates, 1), atol=1e-6))
        # Fit SSN.
        model.fit(self.train_loader)
        self.assertEqual(model.is_centered, True)
        preds = model(self.X, self.Z)
        self.assertEqual(preds.shape, self.y.shape)
        self.assertTrue(torch.allclose(preds_before, preds, atol=1e-4))
        self.assertAlmostEqual(
            model.intercept.item(),
            intercept + model.fx(model.center_x.mean).item() + model.fz(model.center_z.mean).item(),
            places=6)
