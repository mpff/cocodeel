import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cocodeel.dataset import CovarDataset
from cocodeel.model import BaseNetwork, CovarNetwork


class DummyBackbone(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Flatten input to match num_features.
        return x.view(x.size(0), -1)


class TestBaseNetwork(unittest.TestCase):

    @torch.no_grad()
    def setUp(self):
        torch.manual_seed(0)
        self.out_features = 6
        self.backbone = DummyBackbone
        self.backbone_params = {'out_features': self.out_features}
        self.num_covariates = 0
        self.link = "identity"
        self.model = BaseNetwork(self.backbone, self.backbone_params, self.num_covariates, self.link)
        
    def test_initialization(self):
        self.assertIsInstance(self.model.backbone, DummyBackbone)
        self.assertEqual(self.model.backbone.out_features, self.out_features)
        self.assertEqual(self.model.num_covariates, self.num_covariates)
        self.assertEqual(self.model.link, self.link)
        self.assertTrue(hasattr(self.model, 'intercept'))
        self.assertTrue(torch.allclose(self.model.intercept, torch.zeros(1)))
        self.assertTrue(hasattr(self.model, 'center_x'))
        self.assertIsNone(self.model.center_z)
        self.assertTrue(hasattr(self.model, 'center_y'))
        self.assertFalse(self.model.is_centered)

    @torch.no_grad()
    def test_forward_shape(self):
        x = torch.randn(4, 2, 3)  # batch of 4 samples, reshaped to 6 features
        y = self.model(x)
        self.assertEqual(y.shape, (4, 1))

    @torch.no_grad()
    def test_center_features_updates_mean_and_intercept(self):
        # Create fake data: features will average to 1.0 for each dim
        X = torch.ones(10, 2, 3)  # 10 samples, flatten to shape (10, 6)
        dataset = CovarDataset(X, torch.zeros(10), torch.ones(10))
        loader = DataLoader(dataset, batch_size=5)
        # Before centering.
        predictions_before = self.model.forward(X)
        intercept_before = self.model.intercept.clone()
        # Check initial centering state.
        self.assertFalse(self.model.is_centered)
        self.assertTrue(torch.allclose(self.model.center_x.mean, torch.zeros(self.out_features), atol=1e-6))
        self.assertEqual(self.model.center_z, None)
        self.assertTrue(torch.allclose(self.model.center_y.mean, torch.tensor(0.0), atol=1e-6))
        self.assertTrue(torch.allclose(self.model.intercept, torch.tensor(0.0), atol=1e-6))
        # Center the features.
        self.model.center_effects(loader)
        # Check updated centering state.
        self.assertTrue(self.model.is_centered)
        self.assertTrue(torch.allclose(self.model.center_x.mean, torch.ones(self.out_features), atol=1e-6))
        self.assertEqual(self.model.center_z, None)
        self.assertTrue(torch.allclose(self.model.center_y.mean, torch.tensor(1.0), atol=1e-6))
        # Check that intercept is updated correctly.
        expected_intercept = intercept_before + self.model.fx(self.model.center_x.mean)
        self.assertTrue(torch.allclose(self.model.intercept, expected_intercept, atol=1e-6))
        # Check that fx is centered.
        fx_after = self.model.predict_fx(X)
        self.assertTrue(torch.allclose(fx_after.mean(), torch.tensor(0.0), atol=1e-6))
        # Check that predictions are not affected by centering.
        predictions_after = self.model.forward(X)
        self.assertTrue(torch.allclose(predictions_before, predictions_after, atol=1e-6))

    @torch.no_grad()
    def test_center_effects_is_idempotent(self):
        # Calling center_effects twice must not shift the intercept twice.
        X = torch.ones(10, 2, 3)
        dataset = CovarDataset(X, torch.zeros(10), torch.ones(10))
        loader = DataLoader(dataset, batch_size=5)

        self.model.center_effects(loader)
        intercept_after_first = self.model.intercept.clone()
        pred_after_first = self.model.forward(X).clone()

        self.model.center_effects(loader)
        self.assertTrue(torch.allclose(self.model.intercept, intercept_after_first, atol=1e-6))
        self.assertTrue(torch.allclose(self.model.forward(X), pred_after_first, atol=1e-6))


class TestCovarNetwork(unittest.TestCase):

    @torch.no_grad()
    def setUp(self):
        torch.manual_seed(0)
        self.out_features = 6
        self.backbone = DummyBackbone
        self.backbone_params = {'out_features': self.out_features}
        self.num_covariates = 2
        self.link = "identity"
        self.model = CovarNetwork(self.backbone, self.backbone_params, self.num_covariates, self.link)
       
    @torch.no_grad()
    def test_initialization(self):
        self.assertIsInstance(self.model.backbone, DummyBackbone)
        self.assertEqual(self.model.backbone.out_features, self.out_features)
        self.assertEqual(self.model.num_covariates, self.num_covariates)
        self.assertEqual(self.model.link, self.link)
        self.assertTrue(hasattr(self.model, 'intercept'))
        self.assertTrue(torch.allclose(self.model.intercept, torch.zeros(1)))
        self.assertTrue(hasattr(self.model, 'center_x'))
        self.assertTrue(hasattr(self.model, 'center_z'))
        self.assertTrue(hasattr(self.model, 'center_y'))
        self.assertFalse(self.model.is_centered)

    @torch.no_grad()
    def test_forward_shape(self):
        x = torch.randn(4, 2, 3)  # batch of 4 samples, reshaped to 6 features
        z = torch.randn(4, self.num_covariates)  # batch of 4 samples, 2 covariates
        y = self.model(x, z)
        self.assertEqual(y.shape, (4, 1))

    @torch.no_grad()
    def test_center_features_updates_mean_and_intercept(self):
        # Create fake data: features will average to 1.0 for each dim
        X = torch.ones(10, 2, 3)  # 10 samples, flatten to shape (10, 6)
        Z = torch.ones(10, self.num_covariates)  # 10 samples, 2 covariates
        dataset = CovarDataset(X, Z, torch.ones(10))
        loader = DataLoader(dataset, batch_size=5)
        # Before centering.
        predictions_before = self.model.forward(X, Z)
        intercept_before = self.model.intercept.clone()
        # Check initial centering state.
        self.assertFalse(self.model.is_centered)
        self.assertTrue(torch.allclose(self.model.center_x.mean, torch.zeros(self.out_features), atol=1e-6))
        self.assertTrue(torch.allclose(self.model.center_z.mean, torch.zeros(self.num_covariates), atol=1e-6))
        self.assertTrue(torch.allclose(self.model.center_y.mean, torch.tensor(0.0), atol=1e-6))
        self.assertTrue(torch.allclose(self.model.intercept, torch.tensor(0.0), atol=1e-6))
        # Center the features.
        self.model.center_effects(loader)
        # Check updated centering state.
        self.assertTrue(self.model.is_centered)
        self.assertTrue(torch.allclose(self.model.center_x.mean, torch.ones(self.out_features), atol=1e-6))
        self.assertTrue(torch.allclose(self.model.center_z.mean, torch.ones(self.num_covariates), atol=1e-6))
        self.assertTrue(torch.allclose(self.model.center_y.mean, torch.tensor(1.0), atol=1e-6))
        # Check that intercept is updated correctly.
        expected_intercept = intercept_before + self.model.fx(self.model.center_x.mean) + self.model.fz(self.model.center_z.mean)
        self.assertTrue(torch.allclose(self.model.intercept, expected_intercept, atol=1e-6))
        # Check that fx is centered.
        fx_after = self.model.predict_fx(X)
        self.assertTrue(torch.allclose(fx_after.mean(), torch.tensor(0.0), atol=1e-6))
        # Check that fz is centered.
        fz_after = self.model.predict_fz(Z)
        self.assertTrue(torch.allclose(fz_after.mean(), torch.tensor(0.0), atol=1e-6))
        # Check that predictions are not affected by centering.
        predictions_after = self.model.forward(X, Z)
        self.assertTrue(torch.allclose(predictions_before, predictions_after, atol=1e-6))

    @torch.no_grad()
    def test_center_effects_is_idempotent(self):
        # Calling center_effects twice must not shift the intercept twice.
        X = torch.ones(10, 2, 3)
        Z = torch.ones(10, self.num_covariates)
        dataset = CovarDataset(X, Z, torch.ones(10))
        loader = DataLoader(dataset, batch_size=5)

        self.model.center_effects(loader)
        intercept_after_first = self.model.intercept.clone()
        pred_after_first = self.model.forward(X, Z).clone()

        self.model.center_effects(loader)
        self.assertTrue(torch.allclose(self.model.intercept, intercept_after_first, atol=1e-6))
        self.assertTrue(torch.allclose(self.model.forward(X, Z), pred_after_first, atol=1e-6))


class TestGeneralizedLinkFunctions(unittest.TestCase):
    
    @torch.no_grad()   
    def test_identity_link(self):
        model = BaseNetwork(DummyBackbone, {'out_features': 4}, link="identity")
        x = torch.randn(3, 2, 2)
        output = model.forward(x)
        self.assertTrue(torch.allclose(output, model.intercept + model.predict_fx(x)))
        
    @torch.no_grad()
    def test_logit_link(self):
        model = BaseNetwork(DummyBackbone, {'out_features': 4}, link="logit")
        x = torch.randn(3, 2, 2)
        eta = model.intercept + model.predict_fx(x)
        output = model.forward(x)
        expected = torch.sigmoid(eta)
        self.assertTrue(torch.allclose(output, expected, atol=1e-6))
        
    @torch.no_grad()
    def test_log_link(self):
        model = BaseNetwork(DummyBackbone, {'out_features': 4}, link="log")
        x = torch.randn(3, 2, 2)
        eta = model.intercept + model.predict_fx(x)
        output = model.forward(x)
        self.assertTrue(torch.allclose(output, torch.exp(eta), atol=1e-6))


class TestLinkInternals(unittest.TestCase):
    # Direct math pins for forward link g(μ), derivative g'(μ), and variance
    # V(μ). Other tests exercise these only indirectly through IRLS.

    @torch.no_grad()
    def test_identity(self):
        model = BaseNetwork(DummyBackbone, {'out_features': 4}, link="identity")
        mu = torch.tensor([-1.0, 0.0, 0.5, 2.0])
        self.assertTrue(torch.allclose(model._link.forward(mu), mu, atol=1e-6))
        self.assertTrue(torch.allclose(model._link.derivative(mu), torch.ones_like(mu), atol=1e-6))
        self.assertTrue(torch.allclose(model._link.variance(mu), torch.ones_like(mu), atol=1e-6))

    @torch.no_grad()
    def test_logit(self):
        model = BaseNetwork(DummyBackbone, {'out_features': 4}, link="logit")
        mu = torch.tensor([0.2, 0.4, 0.6, 0.8])
        expected_forward = torch.log(mu / (1 - mu))     # g(μ) = log(μ / (1-μ))
        expected_grad = mu * (1 - mu)                   # g'(μ) = μ(1-μ); V(μ) = μ(1-μ)
        self.assertTrue(torch.allclose(model._link.forward(mu), expected_forward, atol=1e-5))
        self.assertTrue(torch.allclose(model._link.derivative(mu), expected_grad, atol=1e-6))
        self.assertTrue(torch.allclose(model._link.variance(mu), expected_grad, atol=1e-6))

    @torch.no_grad()
    def test_log(self):
        model = BaseNetwork(DummyBackbone, {'out_features': 4}, link="log")
        mu = torch.tensor([0.5, 1.0, 2.0, 5.0])
        self.assertTrue(torch.allclose(model._link.forward(mu), torch.log(mu), atol=1e-5))     # g(μ) = log(μ)
        self.assertTrue(torch.allclose(model._link.derivative(mu), 1 / mu, atol=1e-5))         # g'(μ) = 1/μ
        self.assertTrue(torch.allclose(model._link.variance(mu), mu, atol=1e-6))               # V(μ) = μ


if __name__ == "__main__":
    unittest.main()