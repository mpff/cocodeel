import unittest
import torch


from torch import nn
from torch.utils.data import DataLoader

from cocodeel.dataset import CovarDataset
from cocodeel.model import BaseNetwork
from cocodeel.trainer import covar_trainer
from cocodeel.benchmarking.model import CovarNetwork, MLPCovarNetwork
from cocodeel.benchmarking.posthoc_model import PostHocOrthNetwork, SemiStructuredNetwork
from tests.conftest import DummyBackbone


class TestCovarNetwork(unittest.TestCase):

    @torch.no_grad()
    def setUp(self):
        torch.manual_seed(0)
        self.out_features = 6
        self.backbone = DummyBackbone
        # Identity backbone: flattened (2, 3) inputs pass through unchanged.
        self.backbone_params = {'in_features': 6, 'out_features': self.out_features,
                                'identity': True}
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


class TestMLPCovarNetwork(unittest.TestCase):

    @torch.no_grad()
    def setUp(self):
        torch.manual_seed(0)
        self.out_features = 6
        self.num_covariates = 2
        self.model = MLPCovarNetwork(
            DummyBackbone,
            {'in_features': 6, 'out_features': self.out_features, 'identity': True},
            self.num_covariates,
            'identity',
        )

    @torch.no_grad()
    def test_forward_shape(self):
        x = torch.randn(4, 2, 3)
        z = torch.randn(4, self.num_covariates)
        y = self.model(x, z)
        self.assertEqual(y.shape, (4, 1))

    @torch.no_grad()
    def test_center_effects_zeroes_fz_and_preserves_predictions(self):
        X = torch.randn(50, 2, 3)
        Z = torch.randn(50, self.num_covariates)
        loader = DataLoader(CovarDataset(X, Z, torch.randn(50, 1)), batch_size=10)
        predictions_before = self.model(X, Z)
        # center
        self.model.center_effects(loader)
        self.assertTrue(self.model.is_centered)
        # fz is centered on its output, not its input
        fz_after = self.model.predict_fz(Z)
        self.assertTrue(torch.allclose(fz_after.mean(), torch.tensor(0.0), atol=1e-6))
        expected_mean = self.model.fz(Z).mean()
        self.assertTrue(torch.allclose(self.model.center_fz.mean, expected_mean, atol=1e-6))
        # predictions unchanged, second call a no-op
        predictions_after = self.model(X, Z)
        self.assertTrue(torch.allclose(predictions_before, predictions_after, atol=1e-6))
        intercept_after = self.model.intercept.clone()
        self.model.center_effects(loader)
        self.assertTrue(torch.allclose(self.model.intercept, intercept_after, atol=1e-6))

    def test_recovers_linear_covariate_effect(self):
        # y = 2*X0 + 3*Z0 + 1.5: the MLP fz must recover the linear truth
        torch.manual_seed(0)
        n = 1000
        X = torch.randn(n, 3)
        Z = torch.randn(n, self.num_covariates)
        y = (2 * X[:, 0] + 3 * Z[:, 0] + 1.5).unsqueeze(1)
        train_loader = DataLoader(CovarDataset(X[:600], Z[:600], y[:600]),
                                  batch_size=50, shuffle=True)
        val_loader = DataLoader(CovarDataset(X[600:800], Z[600:800], y[600:800]),
                                batch_size=50, shuffle=False)
        model_params = {
            'link': 'identity',
            'backbone': DummyBackbone,
            'backbone_params': {'in_features': 3, 'out_features': 3},
            'num_covariates': self.num_covariates,
        }
        model = covar_trainer(
            model=MLPCovarNetwork,
            model_params=model_params,
            train_loader=train_loader,
            val_loader=val_loader,
            device='cpu',
            epochs=200,
            lr=0.01,
            patience=20,
        )
        model = model.center_effects(train_loader).eval()
        # fz matches the centered truth on held-out covariates
        with torch.no_grad():
            fz_hat = model.predict_fz(Z[800:]).view(-1)
        fz_true = 3 * Z[800:, 0]
        fz_true = fz_true - fz_true.mean()
        rel_mse = ((fz_hat - fz_true) ** 2).mean() / (fz_true ** 2).mean()
        self.assertLess(rel_mse.item(), 0.05)


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
        # Fit Web PHO. A val loader is accepted (and ignored) for interface
        # uniformity with RefitCovarNetwork.fit.
        model.fit(self.train_loader, self.val_loader)
        self.assertEqual(model.is_centered, True)
        preds = model(self.X, self.Z)
        self.assertEqual(preds.shape, self.y.shape)
        self.assertAlmostEqual(
            model.intercept.item(),
            intercept + model.fx(model.center_x.mean).item(),
            places=6)
        # Orthogonalization is mean-zero on the fit sample: eta means match.
        with torch.no_grad():
            eta_base = base_model(self.X[:140], self.Z[:140])
            eta_web = model(self.X[:140], self.Z[:140])
        self.assertAlmostEqual(eta_base.mean().item(), eta_web.mean().item(), places=4)

    def test_wrapping_centered_base_preserves_intercept(self):
        # A base model that already ran center_effects must not be shifted
        # a second time by the wrapper's fit.
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
        base_model.center_effects(self.train_loader)
        model = PostHocOrthNetwork(
            model=base_model,
            num_covariates=self.num_covariates
        )
        self.assertEqual(model.is_centered, True)
        model.fit(self.train_loader, self.val_loader)
        self.assertAlmostEqual(model.intercept.item(), base_model.intercept.item(), places=6)

    def test_train_linear_covar_model(self):
        # Fit end-to-end covariate model, then wrap in SSN.
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
            intercept + model.fx(model.center_x.mean).item(),
            places=6)

    def test_train_linear_covar_model(self):
        # Fit end-to-end covariate model, then wrap in SSN.
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
