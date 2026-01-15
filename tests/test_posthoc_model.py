import unittest
import torch

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

from torch import nn
from torch.utils.data import DataLoader, Dataset

from cocodeel.transform import Center
from cocodeel.dataset import CovarDataset
from cocodeel.model import BaseNetwork
from cocodeel.posthoc_model import PostHocCovarNetwork


class DummyBackbone(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.linear = nn.Linear(3, out_features)
        self.out_features = out_features

    def forward(self, x):
        return self.linear(x)


class TestPostHocLinearCovarNetwork(unittest.TestCase):

    @torch.no_grad()
    def setUp(self):
        torch.manual_seed(42)
        self.n = 100
        self.out_features = 3
        self.num_covariates = 2
        self.link = "identity"
        # Data
        self.X = torch.randn(self.n, 3)
        self.Z = torch.randn(self.n, self.num_covariates)
        self.y = 2 * self.X[:, 0] + 3 * self.Z[:, 0] + 1.5  # Known linear relation
        self.y = self.y.unsqueeze(1)  # (n, 1)
        self.dataset = CovarDataset(self.X, self.Z, self.y)
        self.dataloader = DataLoader(self.dataset, batch_size=25)
        # Pre-train BaseNetwork (to simulate backbone + fx weights)
        self.base_model = BaseNetwork(
            backbone=DummyBackbone, 
            backbone_params={'out_features': self.out_features}, 
            num_covariates=self.num_covariates,
            link=self.link
        )
        # Set backbone weights to be identity mapping.
        self.base_model.backbone.linear.weight.data = torch.eye(self.out_features)
        self.base_model.backbone.linear.bias.data.zero_()
            
    @torch.no_grad()
    def test_initialization(self):
        model = PostHocCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
            orthogonalize=False
        )
        self.assertIsInstance(model.backbone, DummyBackbone)
        self.assertEqual(model.backbone.out_features, self.out_features)
        self.assertEqual(model.num_covariates, self.num_covariates)
        self.assertEqual(model.orthogonalize, False)
        self.assertEqual(model.link, self.link)
        self.assertTrue(hasattr(model, 'fx'))
        self.assertTrue(hasattr(model, 'fz'))
        self.assertTrue(hasattr(model, 'intercept'))
        self.assertTrue(torch.allclose(model.intercept, torch.zeros(1)))
        self.assertTrue(hasattr(model, 'center_x'))
        self.assertTrue(hasattr(model, 'center_z'))
        self.assertTrue(hasattr(model, 'center_y'))
        self.assertFalse(model.is_centered)
        self.assertTrue(torch.allclose(model.orth.weight.data, torch.zeros(self.num_covariates, 1), atol=1e-6))
        self.assertTrue(hasattr(model, 'lam'))
        
    @torch.no_grad()  
    def test_forward_shape(self):
        model = PostHocCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
            orthogonalize=False
        )
        x = torch.ones(4, 3)
        z = torch.ones(4, self.num_covariates)
        y = model(x, z)
        self.assertEqual(y.shape, (4, 1))

    @torch.no_grad()
    def test_posthoc_with_lam0_gives_sensible_fit(self):
        model = PostHocCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
            orthogonalize=True
        )
        # Before fitting.
        self.assertEqual(model.is_centered, False)
        self.assertEqual(model.intercept, torch.zeros(1))
        self.assertTrue(torch.allclose(model.orth.weight.data, torch.zeros(self.num_covariates, 1), atol=1e-6))
        # Fit post-hoc model.
        model.fit(self.dataloader, lam=0.0)  # No penalty
        # Build comparison model for validation.
        val_model = LinearRegression()
        XZ = torch.hstack([model.center_x(self.X), model.center_z(self.Z)])
        val_model.fit(XZ.numpy(), self.y.numpy().squeeze())
        # After fitting.
        self.assertEqual(model.is_centered, True)
        self.assertTrue(torch.allclose(model.intercept, torch.tensor(val_model.intercept_), atol=1e-4))
        self.assertTrue(torch.allclose(model.fx.weight.data[0,0], torch.tensor(val_model.coef_[0]), atol=1e-4))
        self.assertTrue(torch.allclose(model.fz.weight.data[0,0], torch.tensor(val_model.coef_[3]), atol=1e-4))
        # Check predictions after fitting.
        fitted_predictions = model(self.X, self.Z)
        # Check that predictions are close to true y.
        self.assertTrue(torch.allclose(fitted_predictions, self.y, atol=1e-4))

    @torch.no_grad()
    def test_posthoc_gives_sensible_fit(self):
        model = PostHocCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
            orthogonalize=True
        )
        # Before fitting.
        self.assertEqual(model.is_centered, False)
        self.assertEqual(model.intercept, torch.zeros(1))
        self.assertTrue(torch.allclose(model.orth.weight.data, torch.zeros(self.num_covariates, 1), atol=1e-6))
        # Fit post-hoc model.
        model.fit(self.dataloader) # Find optimal lam via GCV.
        # Refit val_model with best lam for comparison.
        val_model = Ridge(alpha=model.lam.item())
        XZ = torch.hstack([self.X, self.Z])
        val_model.fit(XZ.numpy(), self.y.numpy().squeeze())
        # After fitting.
        self.assertEqual(model.is_centered, True)
        self.assertTrue(torch.allclose(model.intercept, self.y.mean(), atol=1e-4))
        self.assertTrue(torch.allclose(model.fx.weight.data[0,0], torch.tensor(val_model.coef_[0]), atol=1e-2))
        self.assertTrue(torch.allclose(model.fz.weight.data[0,0], torch.tensor(val_model.coef_[3]), atol=1e-2))
        # Check predictions after fitting.
        fitted_predictions = model(self.X, self.Z)
        # Check that predictions are close to true y.
        self.assertTrue(torch.allclose(fitted_predictions, self.y, atol=1e-2))

    @torch.no_grad()
    def test_effects_are_centered(self):
        model = PostHocCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
            orthogonalize=True
        )
        model.fit(self.dataloader)
        fx_new = model.predict_fx(self.X, self.Z)
        fz_new = model.predict_fz(self.Z)
        self.assertTrue(torch.allclose(fx_new.mean(dim=0), torch.zeros_like(fx_new.mean(dim=0)), atol=1e-5))
        self.assertTrue(torch.allclose(fz_new.mean(dim=0), torch.zeros_like(fz_new.mean(dim=0)), atol=1e-5))

   
class TestPostHocLogisticCovarNetwork(unittest.TestCase):

    @torch.no_grad()
    def setUp(self):
        torch.manual_seed(42)
        self.n = 4000
        self.out_features = 3
        self.num_covariates = 2
        self.link = "logit"
        # Data
        self.X = torch.randn(self.n, 3)
        self.Z = torch.randn(self.n, self.num_covariates)
        self.eta = 2 * self.X[:, 0] + 3 * self.Z[:, 0] + 1.5  # Known linear relation
        self.p = torch.sigmoid(self.eta.unsqueeze(1))  # (n, 1)
        self.y = torch.bernoulli(self.p) # (n, 1)
        self.dataset = CovarDataset(self.X, self.Z, self.y)
        self.dataloader = DataLoader(self.dataset, batch_size=25)
        # Pre-train BaseNetwork (to simulate backbone + fx weights)
        self.base_model = BaseNetwork(
            backbone=DummyBackbone, 
            backbone_params={'out_features': self.out_features}, 
            num_covariates=self.num_covariates,
            link=self.link
        )
        # Set backbone weights to simulate pre-training. Only X[0] gets a weight of 1, rest is zeroed out.
        self.base_model.backbone.linear.weight.data = torch.eye(self.out_features)
        self.base_model.backbone.linear.bias.data.zero_()
            
    @torch.no_grad()
    def test_initialization(self):
        model = PostHocCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
            orthogonalize=False
        )
        self.assertIsInstance(model.backbone, DummyBackbone)
        self.assertEqual(model.backbone.out_features, self.out_features)
        self.assertEqual(model.num_covariates, self.num_covariates)
        self.assertEqual(model.orthogonalize, False)
        self.assertEqual(model.link, self.link)
        self.assertTrue(hasattr(model, 'fx'))
        self.assertTrue(hasattr(model, 'fz'))
        self.assertTrue(hasattr(model, 'intercept'))
        self.assertTrue(torch.allclose(model.intercept, torch.zeros(1)))
        self.assertTrue(hasattr(model, 'center_x'))
        self.assertTrue(hasattr(model, 'center_z'))
        self.assertTrue(hasattr(model, 'center_y'))
        self.assertFalse(model.is_centered)
        self.assertTrue(torch.allclose(model.orth.weight.data, torch.zeros(self.num_covariates, 1), atol=1e-6))
        self.assertTrue(hasattr(model, 'lam'))
        
    @torch.no_grad()  
    def test_forward_shape(self):
        model = PostHocCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
            orthogonalize=False
        )
        x = torch.ones(4, 3)
        z = torch.ones(4, self.num_covariates)
        y = model(x, z)
        self.assertEqual(y.shape, (4, 1))
        self.assertTrue(torch.all((y >= 0) & (y <= 1)))  # Check output is in [0, 1]

    @torch.no_grad()
    def test_posthoc_with_lam0_gives_sensible_fit(self):
        model = PostHocCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
            orthogonalize=True
        )
        # Before fitting.
        self.assertEqual(model.is_centered, False)
        self.assertEqual(model.intercept, torch.zeros(1))
        self.assertTrue(torch.allclose(model.orth.weight.data, torch.zeros(self.num_covariates, 1), atol=1e-6))
        # Fit post-hoc model.
        model.fit(self.dataloader, lam=0.0) # No penalization.
        # Build comparison model for validation.
        val_model = LogisticRegression(penalty=None)
        XZ = torch.hstack([model.center_x(self.X), model.center_z(self.Z)])
        val_model.fit(XZ.numpy(), self.y.numpy().squeeze())
        # After fitting.
        self.assertEqual(model.is_centered, True)
        self.assertTrue(torch.allclose(model.intercept, torch.tensor(val_model.intercept_).float(), atol=1e-1))
        self.assertTrue(torch.allclose(model.fx.weight.data[0,0], torch.tensor(val_model.coef_[0,0]).float(), atol=1e-2))
        self.assertTrue(torch.allclose(model.fz.weight.data[0,0], torch.tensor(val_model.coef_[0,3]).float(), atol=1e-2))
        # Check predictions after fitting.
        fitted_predictions = model(self.X, self.Z)
        val_predictions = torch.tensor(val_model.predict_proba(XZ.numpy())[:,1]).unsqueeze(1)
        self.assertTrue(torch.allclose(fitted_predictions, val_predictions.float(), atol=1e-1))

    @torch.no_grad()
    def test_posthoc_gives_sensible_fit(self):
        model = PostHocCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
            orthogonalize=True
        )
        # Before fitting.
        self.assertEqual(model.is_centered, False)
        self.assertEqual(model.intercept, torch.zeros(1))
        self.assertTrue(torch.allclose(model.orth.weight.data, torch.zeros(self.num_covariates, 1), atol=1e-6))
        # Fit post-hoc model.
        model.fit(self.dataloader) # Find optimal lam via GCV.
        # Refit val_model with best lam for comparison.
        val_model = LogisticRegression(C=1.0/model.lam.item(), penalty='l2')
        XZ = torch.hstack([model.center_x(self.X), model.center_z(self.Z)])
        val_model.fit(XZ.numpy(), self.y.numpy().squeeze())
        # After fitting.
        self.assertEqual(model.is_centered, True)
        self.assertTrue(torch.allclose(model.intercept, torch.tensor(val_model.intercept_).float(), atol=1e-1))
        self.assertTrue(torch.allclose(model.fx.weight.data[0,0], torch.tensor(val_model.coef_[0,0]).float(), atol=1e-2))
        self.assertTrue(torch.allclose(model.fz.weight.data[0,0], torch.tensor(val_model.coef_[0,3]).float(), atol=1e-2))
        # Check predictions after fitting.
        fitted_predictions = model(self.X, self.Z)
        val_predictions = torch.tensor(val_model.predict_proba(XZ.numpy())[:,1]).unsqueeze(1)
        self.assertTrue(torch.allclose(fitted_predictions, val_predictions.float(), atol=1e-1))

    @torch.no_grad()
    def test_effects_are_centered(self):
        model = PostHocCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
        )
        model.fit(self.dataloader)
        fx_new = model.predict_fx(self.X, self.Z)
        fz_new = model.predict_fz(self.Z)
        self.assertTrue(torch.allclose(fx_new.mean(dim=0), torch.zeros_like(fx_new.mean(dim=0)), atol=1e-5))
        self.assertTrue(torch.allclose(fz_new.mean(dim=0), torch.zeros_like(fz_new.mean(dim=0)), atol=1e-5))


if __name__ == '__main__':
    unittest.main()
