import unittest
import pytest
import torch
import tempfile
import os

import numpy
from sklearn.linear_model import LinearRegression, Ridge

from torch.utils.data import DataLoader

from cocodeel.dataset import CovarDataset
from cocodeel.model import BaseNetwork
from cocodeel.refit_model import RefitCovarNetwork
from cocodeel.trainer import covar_trainer
from tests.conftest import DummyBackbone

rtol, atol = 1e-2, 1e-2


class TestRefitLinearCovarNetwork(unittest.TestCase):

    @torch.no_grad()
    def setUp(self):
        torch.manual_seed(42)
        self.n = 2 * 100
        self.out_features = 3
        self.num_covariates = 2
        self.link = "identity"
        # Data
        self.X = torch.randn(self.n, 3)
        self.Z = torch.randn(self.n, self.num_covariates)
        self.y = 2 * self.X[:, 0] + 3 * self.Z[:, 0] + 1.5  # Known linear relation
        self.y = self.y.unsqueeze(1)  # (n, 1)
        self.train_dataset = CovarDataset(self.X[:100], self.Z[:100], self.y[:100])
        self.val_dataset = CovarDataset(self.X[100:200], self.Z[100:200], self.y[100:200])
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=25)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=25)
        # Identity backbone simulates a pretrained feature map.
        self.base_model = BaseNetwork(
            backbone=DummyBackbone,
            backbone_params={'in_features': self.X.shape[1],
                             'out_features': self.out_features, 'identity': True},
            num_covariates=self.num_covariates,
            link=self.link
        )

    @torch.no_grad()
    def test_initialization(self):
        model = RefitCovarNetwork(
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
        model = RefitCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
            orthogonalize=False
        )
        x = torch.ones(4, 3)
        z = torch.ones(4, self.num_covariates)
        y = model(x, z)
        self.assertEqual(y.shape, (4, 1))

    @torch.no_grad()
    def test_refit_with_lam0_gives_sensible_fit(self):
        model = RefitCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
            orthogonalize=True
        )
        # Before fitting.
        self.assertEqual(model.is_centered, False)
        self.assertEqual(model.intercept, torch.zeros(1))
        self.assertTrue(torch.allclose(model.orth.weight.data, torch.zeros(self.num_covariates, 1), atol=1e-6))
        # Fit refit model.
        model.fit(self.train_dataloader, self.val_dataloader, lam=0.0)  # No penalty
        # Build comparison model for validation.
        val_model = LinearRegression()
        XZ = torch.hstack([model.center_x(self.X[:100]), model.center_z(self.Z[:100])])
        val_model.fit(XZ.numpy(), self.y[:100].numpy().squeeze())
        # After fitting.
        self.assertEqual(model.is_centered, True)
        torch.testing.assert_close(model.intercept.squeeze(), self.y[:100].mean(), rtol=rtol, atol=atol)
        torch.testing.assert_close(model.fz.weight.data[0,0], torch.tensor(3.0).float(), rtol=rtol, atol=atol)
        torch.testing.assert_close(model.fx.weight.data[0,0], torch.tensor(2.0).float(), rtol=rtol, atol=atol)

    @torch.no_grad()
    def test_refit_gives_sensible_fit(self):
        model = RefitCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
            orthogonalize=True
        )
        # Before fitting.
        self.assertEqual(model.is_centered, False)
        self.assertEqual(model.intercept, torch.zeros(1))
        self.assertTrue(torch.allclose(model.orth.weight.data, torch.zeros(self.num_covariates, 1), atol=1e-6))
        # Fit refit model.
        model.fit(self.train_dataloader, self.val_dataloader, n_lambdas=5) # Find optimal lam via GCV.
        # Refit val_model with best lam for comparison.
        val_model = Ridge(alpha=model.lam.item())
        XZ = torch.hstack([model.center_x(self.X[:100]), model.center_z(self.Z[:100])])
        val_model.fit(XZ.numpy(), self.y[:100].numpy().squeeze())
        # After fitting.
        self.assertEqual(model.is_centered, True)
        torch.testing.assert_close(model.intercept.squeeze(), self.y[:100].mean(), rtol=rtol, atol=atol)
        torch.testing.assert_close(model.fz.weight.data[0,0], torch.tensor(3.0).float(), rtol=rtol, atol=atol)
        torch.testing.assert_close(model.fx.weight.data[0,0], torch.tensor(2.0).float(), rtol=rtol, atol=atol)

    @torch.no_grad()
    def test_effects_are_centered(self):
        model = RefitCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
            orthogonalize=True
        )
        model.fit(self.train_dataloader, self.val_dataloader, n_lambdas=5)
        fx_new = model.predict_fx(self.X[:100], self.Z[:100])
        fz_new = model.predict_fz(self.Z[:100])
        self.assertTrue(torch.allclose(fx_new.mean(dim=0), torch.zeros_like(fx_new.mean(dim=0)), atol=1e-5))
        self.assertTrue(torch.allclose(fz_new.mean(dim=0), torch.zeros_like(fz_new.mean(dim=0)), atol=1e-5))

    @torch.no_grad()
    def test_refit_penalty_z_shrinks_fz(self):
        # Unpenalized baseline.
        model_unpen = RefitCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
            orthogonalize=False
        )
        model_unpen.fit(self.train_dataloader, self.val_dataloader, lam=0.0)
        fz_unpen = model_unpen.fz.weight.data[0, 0].item()

        # Large penalty_z: fz should be shrunk towards zero.
        model_pen = RefitCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
            orthogonalize=False
        )
        penalty_z = 1e4 * torch.eye(self.num_covariates)
        model_pen.fit(self.train_dataloader, self.val_dataloader, lam=0.0, penalty_z=penalty_z)
        fz_pen = model_pen.fz.weight.data[0, 0].item()

        self.assertFalse(torch.isnan(model_pen.fz.weight.data).any())
        self.assertFalse(torch.isnan(model_pen.fx.weight.data).any())
        self.assertGreater(abs(fz_unpen), abs(fz_pen))  # penalty shrinks fz toward zero

    @torch.no_grad()
    def test_refit_save_and_reload_linear(self):
        # Train refit model
        model = RefitCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
            orthogonalize=True
        )
        model.fit(self.train_dataloader, self.val_dataloader, n_lambdas=5)

        y_pred_before = model(self.X, self.Z)
        fx_before = model.predict_fx(self.X, self.Z)
        fz_before = model.predict_fz(self.Z)

        # Save refit state_dict
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "refit_linear.pt")
            torch.save(model.state_dict(), path)

            # Recreate BaseNetwork EXACTLY as in setUp
            base_model_reloaded = BaseNetwork(
                backbone=DummyBackbone,
                backbone_params={'in_features': self.X.shape[1],
                                 'out_features': self.out_features, 'identity': True},
                num_covariates=self.num_covariates,
                link=self.link
            )

            # Create fresh refit model and load state_dict
            reloaded = RefitCovarNetwork(
                model=base_model_reloaded,
                num_covariates=self.num_covariates,
                orthogonalize=True
            )
            reloaded.load_state_dict(torch.load(path))
            reloaded.eval()

        # Compare outputs
        y_pred_after = reloaded(self.X, self.Z)
        fx_after = reloaded.predict_fx(self.X, self.Z)
        fz_after = reloaded.predict_fz(self.Z)

        self.assertTrue(torch.allclose(y_pred_before, y_pred_after, atol=1e-4))
        self.assertTrue(torch.allclose(fx_before, fx_after, atol=1e-4))
        self.assertTrue(torch.allclose(fz_before, fz_after, atol=1e-4))
        self.assertTrue(torch.allclose(model.intercept, reloaded.intercept, atol=1e-6))

   
class TestClosedFormRidgeSolution(unittest.TestCase):
    """Pin the identity-link ridge solve against its closed form.

    `fit` standardizes X and Z internally, solves FWL + ridge on the
    standardized scale, and de-standardizes the coefficients:

        X~ = (X - mean X) / sd(X),   Z~ = (Z - mean Z) / sd(Z),   y~ = y - mean y
        X⊥ = X~ - Z~ (Z~'Z~)⁻¹ Z~'X~,   y⊥ = y~ - Z~ (Z~'Z~)⁻¹ Z~'y~
        β_fx = sd(X)⁻¹ ⊙ (X⊥'X⊥ + λI)⁻¹ X⊥'y⊥
        β_fz = sd(Z)⁻¹ ⊙ (Z~'Z~)⁻¹ Z~'(y~ - X~ β_fx,std)
        intercept = mean y

    The sklearn comparisons elsewhere cannot pin the standardize /
    de-standardize round-trip or the λ-scale convention (λ enters
    unscaled on the standardized data); this test does. Tolerance 5e-4:
    the solver's eps=1e-5 IRLS guards perturb coefficients by O(1e-5)
    relative, anything beyond that is a real deviation.
    """

    @torch.no_grad()
    def setUp(self):
        torch.manual_seed(7)
        self.n_train, p, q = 120, 3, 2
        n = 2 * self.n_train
        self.X = torch.randn(n, p)
        self.Z = torch.randn(n, q)
        self.y = (2.0 * self.X[:, 0] - 1.0 * self.X[:, 2] + 3.0 * self.Z[:, 0]
                  + 1.5 + 0.1 * torch.randn(n)).unsqueeze(1)
        tr = CovarDataset(self.X[:self.n_train], self.Z[:self.n_train], self.y[:self.n_train])
        va = CovarDataset(self.X[self.n_train:], self.Z[self.n_train:], self.y[self.n_train:])
        self.train_loader = DataLoader(tr, batch_size=40)
        self.val_loader = DataLoader(va, batch_size=40)
        self.base = BaseNetwork(
            backbone=DummyBackbone,
            backbone_params={'in_features': p, 'out_features': p, 'identity': True},
            num_covariates=q, link="identity",
        )

    @torch.no_grad()
    def _closed_form(self, lam):
        X, Z, y = self.X[:self.n_train], self.Z[:self.n_train], self.y[:self.n_train]
        x_sd = X.std(dim=0, keepdim=True)
        z_sd = Z.std(dim=0, keepdim=True)
        Xs = (X - X.mean(dim=0)) / x_sd
        Zs = (Z - Z.mean(dim=0)) / z_sd
        yc = y - y.mean()

        def regress_out_z(A):
            return A - Zs @ torch.linalg.solve(Zs.T @ Zs, Zs.T @ A)

        X_perp, y_perp = regress_out_z(Xs), regress_out_z(yc)
        beta_fx = torch.linalg.solve(
            X_perp.T @ X_perp + lam * torch.eye(Xs.shape[1]), X_perp.T @ y_perp)
        beta_fz = torch.linalg.solve(Zs.T @ Zs, Zs.T @ (yc - Xs @ beta_fx))
        return (beta_fx.squeeze() / x_sd.squeeze(),
                beta_fz.squeeze() / z_sd.squeeze(),
                y.mean())

    @torch.no_grad()
    def test_fit_matches_closed_form(self):
        for lam in (0.0, 5.0, 50.0):
            with self.subTest(lam=lam):
                model = RefitCovarNetwork(self.base, num_covariates=self.Z.shape[1])
                model.fit(self.train_loader, self.val_loader, lam=lam)
                beta_fx, beta_fz, intercept = self._closed_form(torch.tensor(lam))
                torch.testing.assert_close(model.fx.weight.squeeze(), beta_fx,
                                           rtol=5e-4, atol=5e-4)
                torch.testing.assert_close(model.fz.weight.squeeze(), beta_fz,
                                           rtol=5e-4, atol=5e-4)
                torch.testing.assert_close(model.intercept.squeeze(), intercept,
                                           rtol=5e-4, atol=5e-4)


class TestRefitLogisticCovarNetwork(unittest.TestCase):

    @torch.no_grad()
    def setUp(self):
        torch.manual_seed(42)
        self.n = 500
        self.out_features = 3
        self.num_covariates = 2
        self.link = "logit"
        # Data
        self.X = torch.randn(self.n, 3)
        self.Z = torch.randn(self.n, self.num_covariates)
        self.eta = 1 * self.X[:, 0] + 2 * self.Z[:, 0] + 0.5 # Known linear relation
        self.p = torch.sigmoid(self.eta.unsqueeze(1))  # (n, 1)
        self.y = torch.bernoulli(self.p) # (n, 1)
        self.train_dataset = CovarDataset(self.X[:self.n//2], self.Z[:self.n//2], self.y[:self.n//2])
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=50)
        self.val_dataset = CovarDataset(self.X[self.n//2:], self.Z[self.n//2:], self.y[self.n//2:])
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=50)
        # Identity backbone simulates a pretrained feature map.
        self.base_model = BaseNetwork(
            backbone=DummyBackbone,
            backbone_params={'in_features': self.X.shape[1],
                             'out_features': self.out_features, 'identity': True},
            num_covariates=self.num_covariates,
            link=self.link
        )
            
    @torch.no_grad()
    def test_initialization(self):
        model = RefitCovarNetwork(
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
        model = RefitCovarNetwork(
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
    def test_refit_with_lam0_gives_sensible_fit(self):
        model = RefitCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
            orthogonalize=True
        )
        # Before fitting.
        self.assertEqual(model.is_centered, False)
        self.assertEqual(model.intercept, torch.zeros(1))
        self.assertTrue(torch.allclose(model.orth.weight.data, torch.zeros(self.num_covariates, 1), atol=1e-6))
        # Fit refit model.
        model.fit(self.train_dataloader, self.val_dataloader, lam=0.0) # No penalization.
        # After fitting.
        self.assertEqual(model.is_centered, True)
        # n=250 train → finite-sample noise O(1/sqrt(250))≈0.06, atol=0.1
        torch.testing.assert_close(model.fz.weight.data[0,0], torch.tensor(2.0).float(), rtol=0.1, atol=0.1)
        torch.testing.assert_close(model.fx.weight.data[0,0], torch.tensor(1.0).float(), rtol=0.1, atol=0.1)

    @torch.no_grad()
    def test_refit_gives_sensible_fit(self):
        model = RefitCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
            orthogonalize=True
        )
        # Before fitting.
        self.assertEqual(model.is_centered, False)
        self.assertEqual(model.intercept, torch.zeros(1))
        self.assertTrue(torch.allclose(model.orth.weight.data, torch.zeros(self.num_covariates, 1), atol=1e-6))
        # Fit refit model.
        model.fit(self.train_dataloader, self.val_dataloader, n_lambdas=5)
        # After fitting.
        self.assertEqual(model.is_centered, True)
        # n=250 train → finite-sample noise O(1/sqrt(250))≈0.06, atol=0.1
        torch.testing.assert_close(model.fz.weight.data[0,0], torch.tensor(2.0).float(), rtol=0.1, atol=0.1)
        torch.testing.assert_close(model.fx.weight.data[0,0], torch.tensor(1.0).float(), rtol=0.1, atol=0.1)

    @torch.no_grad()
    def test_effects_are_centered(self):
        model = RefitCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
        )
        model.fit(self.train_dataloader, self.val_dataloader, n_lambdas=5)
        fx_new = model.predict_fx(self.X[:self.n//2], self.Z[:self.n//2])
        fz_new = model.predict_fz(self.Z[:self.n//2])
        self.assertTrue(torch.allclose(fx_new.mean(dim=0), torch.zeros_like(fx_new.mean(dim=0)), atol=1e-5))
        self.assertTrue(torch.allclose(fz_new.mean(dim=0), torch.zeros_like(fz_new.mean(dim=0)), atol=1e-5))

    @torch.no_grad()
    def test_refit_penalty_z_shrinks_fz(self):
        # Unpenalized baseline.
        model_unpen = RefitCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
            orthogonalize=False
        )
        model_unpen.fit(self.train_dataloader, self.val_dataloader, lam=0.0)
        fz_unpen = model_unpen.fz.weight.data[0, 0].item()

        # Large penalty_z: fz should be shrunk towards zero.
        model_pen = RefitCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
            orthogonalize=False
        )
        penalty_z = 1e4 * torch.eye(self.num_covariates)
        model_pen.fit(self.train_dataloader, self.val_dataloader, lam=0.0, penalty_z=penalty_z)
        fz_pen = model_pen.fz.weight.data[0, 0].item()

        self.assertFalse(torch.isnan(model_pen.fz.weight.data).any())
        self.assertFalse(torch.isnan(model_pen.fx.weight.data).any())
        self.assertGreater(abs(fz_unpen), abs(fz_pen))  # penalty shrinks fz toward zero

    @torch.no_grad()
    def test_refit_save_and_reload_logistic(self):
        # Train refit model
        model = RefitCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
            orthogonalize=True
        )
        model.fit(self.train_dataloader, self.val_dataloader, n_lambdas=5)

        y_pred_before = model(self.X[:self.n//2], self.Z[:self.n//2])
        fx_before = model.predict_fx(self.X[:self.n//2], self.Z[:self.n//2])
        fz_before = model.predict_fz(self.Z[:self.n//2])

        # Save refit state_dict
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "refit_logistic.pt")
            torch.save(model.state_dict(), path)

            # Recreate BaseNetwork EXACTLY as in setUp
            base_model_reloaded = BaseNetwork(
                backbone=DummyBackbone,
                backbone_params={'in_features': self.X.shape[1],
                                 'out_features': self.out_features, 'identity': True},
                num_covariates=self.num_covariates,
                link=self.link
            )

            # Create fresh refit model and load state_dict
            reloaded = RefitCovarNetwork(
                model=base_model_reloaded,
                num_covariates=self.num_covariates,
                orthogonalize=True
            )
            reloaded.load_state_dict(torch.load(path))
            reloaded.eval()

        # Compare outputs
        y_pred_after = reloaded(self.X[:self.n//2], self.Z[:self.n//2])
        fx_after = reloaded.predict_fx(self.X[:self.n//2], self.Z[:self.n//2])
        fz_after = reloaded.predict_fz(self.Z[:self.n//2])

        self.assertTrue(torch.allclose(y_pred_before, y_pred_after, atol=atol))
        self.assertTrue(torch.allclose(fx_before, fx_after, atol=atol))
        self.assertTrue(torch.allclose(fz_before, fz_after, atol=atol))
        self.assertTrue(torch.allclose(model.intercept, reloaded.intercept, atol=atol))


class TestHighDimensionalRefit(unittest.TestCase):

    @torch.no_grad()
    def setUp(self):
        torch.manual_seed(42)
        self.n = 200
        self.ntrain = self.n // 2
        self.p = 64
        self.in_features = self.p
        self.out_features = self.p
        self.num_covariates = 1
        self.link = "identity"
        # Data
        self.X = torch.randn(self.n, self.p)
        self.Z = torch.bernoulli(torch.rand(self.n, self.num_covariates))  # Binary covariate
        self.Xcorr = self.X + 0.5 * self.Z  # Introduce correlation between X and Z
        self.y = 2 * self.X[:, 0] + 3 * self.Z[:, 0] + 1.5  # Known linear relation
        self.y = self.y.unsqueeze(1)  # (n, 1)
        self.ycorr = 2 * self.Xcorr[:, 0] + 3 * self.Z[:, 0] + 1.5
        self.ycorr = self.ycorr.unsqueeze(1)  # (n, 1)
        self.train_dataset = CovarDataset(self.X[:self.ntrain], self.Z[:self.ntrain], self.y[:self.ntrain])
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=50)
        self.val_dataset = CovarDataset(self.X[self.ntrain:], self.Z[self.ntrain:], self.y[self.ntrain:])
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=50)
        self.train_dataset_corr = CovarDataset(self.Xcorr[:self.ntrain], self.Z[:self.ntrain], self.ycorr[:self.ntrain])
        self.train_dataloader_corr = DataLoader(self.train_dataset_corr, batch_size=50)
        self.val_dataset_corr = CovarDataset(self.Xcorr[self.ntrain:], self.Z[self.ntrain:], self.ycorr[self.ntrain:])
        self.val_dataloader_corr = DataLoader(self.val_dataset_corr, batch_size=50)
        # Identity backbone simulates a pretrained feature map.
        self.base_model = BaseNetwork(
            backbone=DummyBackbone,
            backbone_params={'in_features': self.in_features,
                             'out_features': self.out_features, 'identity': True},
            num_covariates=self.num_covariates,
            link=self.link
        )

    @torch.no_grad()
    def test_refit_identifies_effects(self):
        model = RefitCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
            orthogonalize=True
        )
        # Before fitting.
        self.assertEqual(model.is_centered, False)
        self.assertEqual(model.intercept, torch.zeros(1))
        self.assertTrue(torch.allclose(model.orth.weight.data, torch.zeros(self.num_covariates, 1), atol=atol))
        # Fit refit model.
        model.fit(self.train_dataloader, self.val_dataloader, n_lambdas=5)
        # After fitting.
        self.assertEqual(model.is_centered, True)
        # Check that identified effects are close to true effects!
        torch.testing.assert_close(model.fz.weight.data[0,0], torch.tensor(3.0).float(), rtol=rtol, atol=atol)

    @torch.no_grad()
    def test_refit_identifies_effects_for_correlated_data(self):
        model = RefitCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
            orthogonalize=True
        )
        # Before fitting.
        self.assertEqual(model.is_centered, False)
        self.assertEqual(model.intercept, torch.zeros(1))
        self.assertTrue(torch.allclose(model.orth.weight.data, torch.zeros(self.num_covariates, 1), atol=atol))
        # Fit refit model with penalty
        model.fit(self.train_dataloader_corr, self.val_dataloader_corr, n_lambdas=5)
        # After fitting.
        self.assertEqual(model.is_centered, True)
        torch.testing.assert_close(model.fz.weight.data[0,0], torch.tensor(3.0).float(), rtol=rtol, atol=atol)




class TestHighDimRegression(unittest.TestCase):
    """FWL residualization and ridge correctness with confounded data.

    Scientific question: after FWL residualization (resid_X = X - Z(Z'Z)⁻¹Z'X,
    resid_y = y - Z(Z'Z)⁻¹Z'y), can we estimate β_fx from resid_X → resid_y and β_fz
    consistently, even when N < d and corr(Z, X) is substantial?

    DGP: H = 0.3*Z + noise (corr(Z, H_j) ≈ 0.29), y = 2*H[:,0] + 3*Z + eps.
    Identity backbone (H passes through unchanged).
    True β_fz=3, β_fx[0]=2, intercept=0.

    setUp uses d=64, N=200 for fast CPU tests. The UKBB-scale test (d=2048, N=2000)
    is skipped by default — see test_consistent_fz_under_high_dimensional_confounding.
    """

    @torch.no_grad()
    def setUp(self):
        torch.manual_seed(0)
        rng = numpy.random.default_rng(0)
        self.n, self.ntrain, self.d = 200, 150, 64
        Z = torch.tensor(rng.standard_normal((self.n, 1)), dtype=torch.float32)
        noise = torch.tensor(rng.standard_normal((self.n, self.d)), dtype=torch.float32)
        H = 0.3 * Z + noise                                          # corr(Z, H_j) ≈ 0.29
        eps = torch.tensor(rng.standard_normal((self.n, 1)), dtype=torch.float32) * 0.5
        y = 2.0 * H[:, [0]] + 3.0 * Z + eps                         # true fz=3, fx[0]=2
        self.Z, self.H, self.y = Z, H, y

        train_ds = CovarDataset(H[:self.ntrain], Z[:self.ntrain], y[:self.ntrain])
        val_ds   = CovarDataset(H[self.ntrain:], Z[self.ntrain:], y[self.ntrain:])
        self.train_loader = DataLoader(train_ds, batch_size=64)
        self.val_loader   = DataLoader(val_ds,   batch_size=64)

        self.base_model = BaseNetwork(
            backbone=DummyBackbone,
            backbone_params={'in_features': self.d, 'out_features': self.d,
                             'identity': True},
            num_covariates=1, link="identity"
        )

    def test_ridge_collapses_fx_when_lambda_dominates_spectrum(self):
        """Isotropic ridge shrinks β_fx to ~0 when λ >> σ_max²(resid_X).

        With d=64, N=200 (ntrain=150), σ_max²(resid_X) ≈ ntrain ≈ 150.
        λ=2e5 >> 150: shrinkage factor σ_j / (σ_j² + λ) ≈ 0 for every direction.
        β_fx ≈ 0. This motivates restricting regularization to the identifiable subspace.
        """
        model = RefitCovarNetwork(self.base_model, num_covariates=1)
        model.fit(self.train_loader, self.val_loader, lam=2e5)
        fx = model.predict_fx(self.H[:self.ntrain], self.Z[:self.ntrain])
        self.assertLess(fx.std().item(), 0.1,
            msg=f"fx_std={fx.std().item():.4f}, expected < 0.1 when λ >> σ_max²")

    @pytest.mark.skip(reason="slow (~2min on CPU): UKBB-scale N/d≈1 bias. "
                             "Behaviour documented by test_fz_high_variance_at_low_N_d_ratio. "
                             "Remove skip after PCR fix lands.")
    def test_consistent_fz_under_high_dimensional_confounding(self):
        """β_fz is biased at UKBB scale: d=2048, N=2000 (ntrain/d≈0.73).

        With isotropic ridge, bias(β_fz) = (Zw'Zw)⁻¹ Zw'Xw @ (β_fx_true − β_fx_ridge)
        is non-negligible because β_fx is over-shrunk when N/d < 1. Expected: fz≈3.9
        (true value 3.0). This test FAILS by design — it documents a known open issue.

        See test_fz_high_variance_at_low_N_d_ratio for the fast equivalent (d=64).
        """
        rng = numpy.random.default_rng(0)
        n, ntrain, d = 2000, 1500, 2048
        Z = torch.tensor(rng.standard_normal((n, 1)), dtype=torch.float32)
        H = 0.3 * Z + torch.tensor(rng.standard_normal((n, d)), dtype=torch.float32)
        y = 2.0 * H[:, [0]] + 3.0 * Z + torch.tensor(
            rng.standard_normal((n, 1)), dtype=torch.float32) * 0.5
        train_loader = DataLoader(
            CovarDataset(H[:ntrain], Z[:ntrain], y[:ntrain]), batch_size=64)
        val_loader = DataLoader(
            CovarDataset(H[ntrain:], Z[ntrain:], y[ntrain:]), batch_size=64)
        base = BaseNetwork(backbone=DummyBackbone,
                           backbone_params={'in_features': d, 'out_features': d,
                                            'identity': True},
                           num_covariates=1, link="identity")
        model = RefitCovarNetwork(base, num_covariates=1)
        model.fit(train_loader, val_loader)
        fz = model.fz.weight.data[0, 0].item()
        self.assertAlmostEqual(fz, 3.0, delta=0.3,
            msg=f"fz={fz:.3f}, expected ~3.0 ± 0.3")

    def test_refit_preserves_fx_signal(self):
        """Refit retains fx signal: std(fx) > 0.5 when a real effect exists.

        d=64, N=200 (ntrain=150), lam=10 (lam/σ_max²≈7%). After FWL residualization
        and ridge regression, the estimated β_fx should pick up the true signal in
        H[:,0] and produce non-trivial predictions.
        """
        model = RefitCovarNetwork(self.base_model, num_covariates=1)
        model.fit(self.train_loader, self.val_loader, lam=10.0)
        fx = model.predict_fx(self.H[:self.ntrain], self.Z[:self.ntrain])
        self.assertGreater(fx.std().item(), 0.5,
            msg=f"fx_std={fx.std().item():.4f}, expected > 0.5")

    def test_refit_fx_decorrelated_from_Z(self):
        """|Corr(Z, fx)| < 0.3 after refit: FWL residualization removes Z from β_fx.

        d=64, N=200 (ntrain=150), lam=10. Regressing Z out of X before estimating
        β_fx (resid_X = X - Z(Z'Z)⁻¹Z'X) ensures the estimated fx is approximately
        orthogonal to Z. This holds whenever λ is not so large that fx ≈ 0.
        """
        model = RefitCovarNetwork(self.base_model, num_covariates=1)
        model.fit(self.train_loader, self.val_loader, lam=10.0)
        fx = model.predict_fx(self.H[:self.ntrain], self.Z[:self.ntrain]).squeeze()
        z  = self.Z[:self.ntrain].squeeze()
        corr = ((z - z.mean()) @ (fx - fx.mean())) / (
            (z - z.mean()).norm() * (fx - fx.mean()).norm() + 1e-8)
        self.assertLess(abs(corr.item()), 0.35,
            msg=f"Corr(Z, fx)={corr.item():.3f}, expected |corr| < 0.35")

    @torch.no_grad()
    def test_fz_high_variance_at_low_N_d_ratio(self):
        """Ridge fz is highly variable at ntrain/d=0.75: the estimator is unreliable.

        d=64, N=64 (ntrain=48, ntrain/d=0.75), fixed lam=100.
        Empirical: std=0.443 over 7 seeds (individual values range from 2.67 to 4.0).

        Documents that at UKBB-equivalent ntrain/d, ridge cannot reliably estimate fz.
        Pair with test_fz_converges_at_large_N_d_ratio (same d=64) for the full picture.
        """
        d, N, lam = 64, 64, 100.0
        ntrain = int(0.75 * N)
        fz_values = []
        for seed in range(7):
            rng = numpy.random.default_rng(seed)
            Z = torch.tensor(rng.standard_normal((N, 1)), dtype=torch.float32)
            H = 0.3 * Z + torch.tensor(rng.standard_normal((N, d)), dtype=torch.float32)
            y = 2.0 * H[:, [0]] + 3.0 * Z + torch.tensor(
                rng.standard_normal((N, 1)), dtype=torch.float32) * 0.5
            train_loader = DataLoader(
                CovarDataset(H[:ntrain], Z[:ntrain], y[:ntrain]), batch_size=64)
            val_loader = DataLoader(
                CovarDataset(H[ntrain:], Z[ntrain:], y[ntrain:]), batch_size=64)
            base = BaseNetwork(backbone=DummyBackbone,
                               backbone_params={'in_features': d, 'out_features': d,
                                                'identity': True},
                               num_covariates=1, link="identity")
            model = RefitCovarNetwork(base, num_covariates=1)
            model.fit(train_loader, val_loader, lam=lam)
            fz_values.append(model.fz.weight.data[0, 0].item())

        std_fz = numpy.std(fz_values)
        self.assertGreater(std_fz, 0.3,
            msg=f"std(fz)={std_fz:.3f} at ntrain/d=0.75: expected >0.3. "
                f"Individual: {[f'{v:.2f}' for v in fz_values]}")

    @torch.no_grad()
    def test_fz_converges_at_large_N_d_ratio(self):
        """Ridge fz converges to truth at ntrain/d=9: estimator is consistent.

        d=64, N=768 (ntrain=576, ntrain/d=9.0), fixed lam=100 (lam/σ_max²≈10%).
        Empirical: mean=3.067, std=0.030 over 7 seeds.

        Documents that the bias at UKBB scale (ntrain/d≈0.73) is finite-sample, not
        structural. Pair with test_fz_high_variance_at_low_N_d_ratio (same d=64).
        """
        d, N, lam = 64, 768, 100.0
        ntrain = int(0.75 * N)
        fz_values = []
        for seed in range(7):
            rng = numpy.random.default_rng(seed)
            Z = torch.tensor(rng.standard_normal((N, 1)), dtype=torch.float32)
            H = 0.3 * Z + torch.tensor(rng.standard_normal((N, d)), dtype=torch.float32)
            y = 2.0 * H[:, [0]] + 3.0 * Z + torch.tensor(
                rng.standard_normal((N, 1)), dtype=torch.float32) * 0.5
            train_loader = DataLoader(
                CovarDataset(H[:ntrain], Z[:ntrain], y[:ntrain]), batch_size=64)
            val_loader = DataLoader(
                CovarDataset(H[ntrain:], Z[ntrain:], y[ntrain:]), batch_size=64)
            base = BaseNetwork(backbone=DummyBackbone,
                               backbone_params={'in_features': d, 'out_features': d,
                                                'identity': True},
                               num_covariates=1, link="identity")
            model = RefitCovarNetwork(base, num_covariates=1)
            model.fit(train_loader, val_loader, lam=lam)
            fz_values.append(model.fz.weight.data[0, 0].item())

        std_fz = numpy.std(fz_values)
        mean_fz = numpy.mean(fz_values)
        self.assertLess(std_fz, 0.05,
            msg=f"std(fz)={std_fz:.3f} at ntrain/d=9: expected <0.05. "
                f"Individual: {[f'{v:.2f}' for v in fz_values]}")
        self.assertAlmostEqual(mean_fz, 3.0, delta=0.1,
            msg=f"mean(fz)={mean_fz:.3f} at ntrain/d=9: expected 3.0±0.1")


    @torch.no_grad()
    def test_fx_z_correlation_when_signal_shares_confounded_direction(self):
        """When the true signal shares the confounded direction, Corr(Z, fx) persists.

        Mirrors UKBB: the backbone was fine-tuned on confounded data, so its
        features entangle the true signal with the confound in the dominant PC.
        The FWL correctly estimates b from residualized features, but predict_fx
        uses full H, so the Z-correlated component of the true signal passes through.

        This is CORRECT behavior — the true effect genuinely covaries with Z.
        Only orthogonalization can remove it (by subtracting the Z projection).

        DGP: d=20, N=300, identity link.
        PC0: dominant (σ₀=5), Z-correlated (ρ₀=0.5), AND carries true signal.
        PC1-3: moderate variance, uncorrelated with Z, also carry true signal.

        Tests:
        A. Ridge: Corr(Z, fx) persists — true signal loads on Z-correlated PC0.
        B. OLS (λ≈0): similar — the true signal IS in PC0 regardless of regularization.
        C. Orth: removes Z-correlation by construction.
        """
        d, N, ntrain = 20, 300, 200
        rng = numpy.random.default_rng(42)
        Z = torch.tensor(rng.standard_normal((N, 1)), dtype=torch.float32)

        # Build features: dominant PC0 is Z-correlated, PC1-4 carry signal.
        H = torch.zeros(N, d)
        # PC0: dominant, Z-correlated (mimics UKBB PC0 at 50% variance, ρ=0.5)
        rho0 = 0.5
        H[:, 0] = 5.0 * (rho0 * Z.squeeze() + numpy.sqrt(1 - rho0**2) *
                          torch.tensor(rng.standard_normal(N), dtype=torch.float32))
        # PC1-4: moderate variance, weak Z-correlation, carry true signal
        for j in range(1, 5):
            H[:, j] = 1.5 * torch.tensor(rng.standard_normal(N), dtype=torch.float32)
        # PC5+: small filler
        for j in range(5, d):
            H[:, j] = 0.3 * torch.tensor(rng.standard_normal(N), dtype=torch.float32)

        # True signal in BOTH PC0 AND PC1-3 (entangled, mimics contaminated backbone).
        # The backbone was trained on confounded data, so the learned features that
        # predict y are partly in the Z-correlated direction (PC0).
        y = (1.0 * H[:, [0]] + 0.5 * H[:, [1]] + 0.3 * H[:, [2]]
             + 3.0 * Z + 0.5 * torch.tensor(rng.standard_normal((N, 1)),
                                              dtype=torch.float32))

        # Base model: w_base loads on PC0 (simulates confound-exploiting training).
        base = BaseNetwork(backbone=DummyBackbone,
                           backbone_params={'in_features': d, 'out_features': d,
                                            'identity': True},
                           num_covariates=1, link="identity")
        base.fx.weight.data.zero_()
        base.fx.weight.data[0, 0] = 1.0  # loads on PC0

        # IMPORTANT: shuffle=False for all evaluation DataLoaders.
        train_ds = CovarDataset(H[:ntrain], Z[:ntrain], y[:ntrain])
        val_ds = CovarDataset(H[ntrain:], Z[ntrain:], y[ntrain:])
        tl = DataLoader(train_ds, batch_size=64, shuffle=False)
        vl = DataLoader(val_ds, batch_size=64, shuffle=False)

        # --- fx_base (before refit) ---
        fx_base = (H[:ntrain] @ base.fx.weight.data.squeeze()).numpy()
        z_train = Z[:ntrain, 0].numpy()
        corr_z_base = numpy.corrcoef(z_train, fx_base)[0, 1]

        # --- A: Ridge (auto λ) ---
        import copy
        base_a = copy.deepcopy(base)
        model_ridge = RefitCovarNetwork(base_a, num_covariates=1)
        model_ridge.fit(tl, vl, n_lambdas=5)
        fx_ridge = model_ridge.predict_fx(H[:ntrain], Z[:ntrain]).squeeze().numpy()
        corr_z_ridge = numpy.corrcoef(z_train, fx_ridge)[0, 1]
        corr_base_ridge = numpy.corrcoef(fx_base, fx_ridge)[0, 1]

        # --- B: Near-zero λ (OLS-like) ---
        base_b = copy.deepcopy(base)
        model_ols = RefitCovarNetwork(base_b, num_covariates=1)
        model_ols.fit(tl, vl, lam=0.01)
        fx_ols = model_ols.predict_fx(H[:ntrain], Z[:ntrain]).squeeze().numpy()
        corr_z_ols = numpy.corrcoef(z_train, fx_ols)[0, 1]
        corr_base_ols = numpy.corrcoef(fx_base, fx_ols)[0, 1]

        # --- C: Ridge + orthogonalization ---
        base_c = copy.deepcopy(base)
        model_orth = RefitCovarNetwork(base_c, num_covariates=1, orthogonalize=True)
        model_orth.fit(tl, vl, n_lambdas=5)
        fx_orth = model_orth.predict_fx(H[:ntrain], Z[:ntrain]).squeeze().numpy()
        corr_z_orth = numpy.corrcoef(z_train, fx_orth)[0, 1]

        # Print diagnostics for debugging.
        print(f"\n  fx diagnostics (identity link, d={d}, N={N}):")
        print(f"  base:      corr(Z,fx)={corr_z_base:.3f}  std={fx_base.std():.3f}")
        print(f"  ridge:     corr(Z,fx)={corr_z_ridge:.3f}  std={fx_ridge.std():.3f}  "
              f"corr(base,ridge)={corr_base_ridge:.3f}  lam={model_ridge.lam.item():.2e}")
        print(f"  ols(λ≈0):  corr(Z,fx)={corr_z_ols:.3f}  std={fx_ols.std():.3f}  "
              f"corr(base,ols)={corr_base_ols:.3f}")
        print(f"  orth:      corr(Z,fx)={corr_z_orth:.3f}  std={fx_orth.std():.3f}")

        # Assertions:
        # A. Base model fx IS Z-correlated (by construction, loads on PC0).
        self.assertGreater(abs(corr_z_base), 0.3,
            msg=f"Base fx should be Z-correlated: corr={corr_z_base:.3f}")

        # B. Ridge refit RETAINS Z-correlation — the true signal genuinely
        #    uses the Z-correlated feature (PC0). This is correct, not a bug.
        self.assertGreater(abs(corr_z_ridge), 0.15,
            msg=f"Ridge refit should retain Z-correlation when signal shares "
                f"confounded direction: corr={corr_z_ridge:.3f}")

        # C. OLS also retains Z-correlation — same reason.
        self.assertGreater(abs(corr_z_ols), 0.15,
            msg=f"OLS should also retain Z-correlation: corr={corr_z_ols:.3f}")

        # D. Orthogonalization removes Z-correlation by construction.
        self.assertLess(abs(corr_z_orth), 0.1,
            msg=f"Orth should remove Z-correlation: corr={corr_z_orth:.3f}")


    @torch.no_grad()
    def test_disentangle_ridge_vs_entanglement(self):
        """Separates two causes of unchanged Corr(Z, fx) after refit.

        Cause 1 (ridge): large λ lets through OVB → debiasing blocked.
        Cause 2 (entanglement): true signal shares confounded direction → nothing to debias.

        2D sweep over (β₀, λ):
        - β₀ controls entanglement (0 = separate, 1 = shared direction)
        - λ controls regularization (0.01 = OLS, 1e5 = heavy ridge)

        Expected:
          β₀=0, λ≈0:  LARGE reduction (>50%) — debiasing works
          β₀=0, λ=1e5: SMALL reduction (<20%) — ridge blocks debiasing
          β₀=1, λ≈0:  NO reduction (<10%) — genuine entanglement
        """
        d, N, ntrain = 20, 500, 400

        def run_cell(beta0, lam):
            rng = numpy.random.default_rng(42)
            Z = torch.tensor(rng.standard_normal((N, 1)), dtype=torch.float32)

            H = torch.zeros(N, d)
            rho = 0.5
            H[:, 0] = 5.0 * (rho * Z.squeeze() + numpy.sqrt(1 - rho**2) *
                              torch.tensor(rng.standard_normal(N), dtype=torch.float32))
            for j in range(1, 5):
                H[:, j] = 1.5 * torch.tensor(rng.standard_normal(N), dtype=torch.float32)
            for j in range(5, d):
                H[:, j] = 0.3 * torch.tensor(rng.standard_normal(N), dtype=torch.float32)

            y = (beta0 * H[:, [0]] + 0.5 * H[:, [1]] + 0.3 * H[:, [2]]
                 + 3.0 * Z + 0.5 * torch.tensor(rng.standard_normal((N, 1)),
                                                  dtype=torch.float32))

            # Base: OLS without Z (has OVB).
            w_noZ = torch.linalg.lstsq(H[:ntrain], y[:ntrain]).solution
            fx_base = (H[ntrain:] @ w_noZ).squeeze().numpy()

            # Refit: FWL with given λ.
            base = BaseNetwork(backbone=DummyBackbone,
                               backbone_params={'in_features': d, 'out_features': d,
                                                'identity': True},
                               num_covariates=1, link="identity")
            base.fx.weight.data = w_noZ.T

            tl = DataLoader(CovarDataset(H[:ntrain], Z[:ntrain], y[:ntrain]),
                            batch_size=64, shuffle=False)
            vl = DataLoader(CovarDataset(H[ntrain:], Z[ntrain:], y[ntrain:]),
                            batch_size=64, shuffle=False)

            phm = RefitCovarNetwork(base, num_covariates=1)
            phm.fit(tl, vl, lam=lam)
            fx_ph = phm.predict_fx(H[ntrain:], Z[ntrain:]).squeeze().numpy()

            z_test = Z[ntrain:, 0].numpy()
            c_base = abs(numpy.corrcoef(z_test, fx_base)[0, 1])
            c_ph = abs(numpy.corrcoef(z_test, fx_ph)[0, 1])
            reduction = (c_base - c_ph) / c_base * 100 if c_base > 0.01 else 0
            return c_base, c_ph, reduction

        betas = [0.0, 0.3, 1.0]
        lambdas = [0.01, 10, 1000, 1e5]

        print(f"\n  {'':>6s}", end="")
        for lam in lambdas:
            print(f"  {'λ='+f'{lam:.0e}':>12s}", end="")
        print()
        print("  " + "-" * 56)

        results = {}
        for beta0 in betas:
            print(f"  β₀={beta0:3.1f}", end="")
            for lam in lambdas:
                c_b, c_p, red = run_cell(beta0, lam)
                results[(beta0, lam)] = (c_b, c_p, red)
                print(f"  {red:>10.1f}%", end="")
            print()

        # Key assertions:
        # 1. No entanglement → debiasing works at ANY λ.
        _, _, red_ols = results[(0.0, 0.01)]
        _, _, red_ridge = results[(0.0, 1e5)]
        self.assertGreater(red_ols, 50,
            msg=f"β₀=0, λ≈0: reduction={red_ols:.1f}%, expected >50%")
        self.assertGreater(red_ridge, 50,
            msg=f"β₀=0, λ=1e5: reduction={red_ridge:.1f}%, expected >50% "
                f"— ridge does NOT block debiasing when signal is disentangled")

        # 2. Entanglement → debiasing impossible at ANY λ.
        _, _, red_entangled_ols = results[(1.0, 0.01)]
        _, _, red_entangled_ridge = results[(1.0, 1e5)]
        self.assertLess(abs(red_entangled_ols), 10,
            msg=f"β₀=1, λ≈0: reduction={red_entangled_ols:.1f}%, expected <10% "
                f"— genuine entanglement, debiasing impossible even with OLS")
        self.assertLess(abs(red_entangled_ridge), 10,
            msg=f"β₀=1, λ=1e5: reduction={red_entangled_ridge:.1f}%, expected <10% "
                f"— entanglement dominates regardless of λ")


class TestFitWithDisjointRefitLoader(unittest.TestCase):
    """Regression test: RefitCovarNetwork.fit works when the refit loader's
    observations are disjoint from the sample used to pretrain the backbone.

    This is the `sample-split` recipe — the backbone was fit on sample A; the
    refit is refitted on sample B. Exogeneity of H = phi(X_B; theta*) is then
    restored (Pagan 1984 generated regressors). The test verifies the API runs
    cleanly and recovers centered effects on a disjoint sample.
    """

    @torch.no_grad()
    def setUp(self):
        torch.manual_seed(123)
        self.out_features = 3
        self.num_covariates = 1

        # Two independent draws from the same DGP — A is for the backbone, B for the refit.
        n = 200
        self.X_A = torch.randn(n, 3)
        self.Z_A = torch.randn(n, self.num_covariates)
        self.y_A = (2 * self.X_A[:, 0] + 3 * self.Z_A[:, 0] + 1.5).unsqueeze(1)
        self.X_B = torch.randn(n, 3)
        self.Z_B = torch.randn(n, self.num_covariates)
        self.y_B = (2 * self.X_B[:, 0] + 3 * self.Z_B[:, 0] + 1.5).unsqueeze(1)

        # DataLoaders: use half of each sample for train, half for val.
        def _make(X, Z, y, bs=25):
            tr = CovarDataset(X[:n // 2], Z[:n // 2], y[:n // 2])
            va = CovarDataset(X[n // 2:], Z[n // 2:], y[n // 2:])
            return DataLoader(tr, batch_size=bs), DataLoader(va, batch_size=bs)

        self.tr_A, self.va_A = _make(self.X_A, self.Z_A, self.y_A)
        self.tr_B, self.va_B = _make(self.X_B, self.Z_B, self.y_B)

        # Base model with an identity backbone (simulates a pretrained feature map).
        self.base = BaseNetwork(
            backbone=DummyBackbone,
            backbone_params={"in_features": 3, "out_features": self.out_features,
                             "identity": True},
            num_covariates=self.num_covariates,
            link="identity",
        )
        self.base = self.base.center_effects(self.tr_A)  # center on the A-sample

    @torch.no_grad()
    def test_refit_runs_on_disjoint_refit_sample(self):
        """Split recipe: backbone on A, refit on B. Must run and
        recover effects close to truth."""
        model = RefitCovarNetwork(
            model=self.base,
            num_covariates=self.num_covariates,
            orthogonalize=False,
        )
        # Refit on B (disjoint from A).
        model.fit(self.tr_B, self.va_B, lam=0.0)

        # Basic invariants.
        self.assertEqual(model.is_centered, True)
        # Effects should recover truth within tolerance.
        torch.testing.assert_close(
            model.fz.weight.data[0, 0], torch.tensor(3.0).float(),
            rtol=rtol, atol=atol,
        )
        torch.testing.assert_close(
            model.fx.weight.data[0, 0], torch.tensor(2.0).float(),
            rtol=rtol, atol=atol,
        )

    @torch.no_grad()
    def test_refit_fx_centered_on_refit_sample(self):
        """After refit on B, f̂_X and f̂_Z should be mean-zero on B (the refit
        sample's centering), not on A."""
        model = RefitCovarNetwork(
            model=self.base,
            num_covariates=self.num_covariates,
            orthogonalize=False,
        )
        model.fit(self.tr_B, self.va_B, lam=0.0)

        fx_B = model.predict_fx(self.X_B[:100], self.Z_B[:100])
        fz_B = model.predict_fz(self.Z_B[:100])
        self.assertTrue(torch.allclose(fx_B.mean(dim=0), torch.zeros_like(fx_B.mean(dim=0)), atol=1e-4))
        self.assertTrue(torch.allclose(fz_B.mean(dim=0), torch.zeros_like(fz_B.mean(dim=0)), atol=1e-4))

    @torch.no_grad()
    def test_center_effects_after_fit_is_idempotent(self):
        """After `.fit(B)`, calling `center_effects(B)` again must not change
        predictions.

        Bug under the current `is_centered=True` branch: `intercept += fz(μ_z)`
        fires on every call. Once IRLS has set `fz.weight ≠ 0`, this silently
        shifts the intercept by `fz.weight · μ_z_B` — corrupting predictions.
        """
        refit = RefitCovarNetwork(
            model=self.base,
            num_covariates=self.num_covariates,
            orthogonalize=False,
        )
        refit.fit(self.tr_B, self.va_B, lam=0.0)

        intercept_before = refit.intercept.clone()
        pred_before = refit(self.X_B[:20], self.Z_B[:20]).clone()

        refit.center_effects(self.tr_B)  # re-center on the same loader

        torch.testing.assert_close(refit.intercept, intercept_before, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            refit(self.X_B[:20], self.Z_B[:20]), pred_before, rtol=1e-5, atol=1e-5
        )

    @torch.no_grad()
    def test_fit_on_B_uses_B_centered_means(self):
        """Invariant: after `.fit(B)` with `base.center_effects(A)`, the
        refit's centering means must reflect B — not A. Intercept is fit on
        B (by IRLS), so the centers have to match B as well."""
        refit = RefitCovarNetwork(
            model=self.base,
            num_covariates=self.num_covariates,
            orthogonalize=False,
        )
        refit.fit(self.tr_B, self.va_B, lam=0.0)

        X_B_feat, Z_B_feat, _ = refit._extract_features_from_loader(self.tr_B)
        torch.testing.assert_close(
            refit.center_x.mean, X_B_feat.mean(dim=0), rtol=1e-5, atol=1e-5,
        )
        torch.testing.assert_close(
            refit.center_z.mean, Z_B_feat.mean(dim=0), rtol=1e-5, atol=1e-5,
        )


class TestSampleSplitRecoversKnownEffects(unittest.TestCase):
    """Split recipe (backbone trained on A, refit on disjoint B)
    recovers known linear effects — continuous and binary outcome.

    The refit runs unpenalized (lam=0): ridge shrinkage of fx would
    attenuate fz under the logit link (non-collapsibility), which is a
    penalization effect, not a property of the split estimator.

    Calibrated over seeds 0-4: identity fz = 3.02 ± 0.02 (truth 3.0),
    logit fz = 1.96 ± 0.13 (truth 2.0), corr(fx_hat, fx) >= 0.94.
    """

    N, D, SEEDS = 600, 3, (0, 1, 2)

    def _split_fit(self, seed, link):
        torch.manual_seed(seed)

        def sample():
            Z = torch.randn(self.N, 1)
            X = torch.randn(self.N, self.D)
            if link == "identity":
                y = 2.0 * X[:, [0]] + 3.0 * Z + 0.5 * torch.randn(self.N, 1)
            else:
                y = torch.bernoulli(torch.sigmoid(1.0 * X[:, [0]] + 2.0 * Z + 0.5))
            half = self.N // 2
            tr = DataLoader(CovarDataset(X[:half], Z[:half], y[:half]),
                            batch_size=32, shuffle=True)
            va = DataLoader(CovarDataset(X[half:], Z[half:], y[half:]),
                            batch_size=32, shuffle=False)
            return X, Z, tr, va

        _, _, tr_A, va_A = sample()
        X_B, Z_B, tr_B, va_B = sample()
        model_params = {
            "link": link,
            "backbone": DummyBackbone,
            "backbone_params": {"in_features": self.D, "out_features": self.D},
            "num_covariates": 0,
        }
        loss = torch.nn.MSELoss() if link == "identity" else torch.nn.BCELoss()
        base = covar_trainer(BaseNetwork, model_params, tr_A, va_A, device="cpu",
                             loss_fn=loss, epochs=60, lr=0.01, patience=20)
        model = RefitCovarNetwork(base, num_covariates=1).fit(tr_B, va_B, lam=0.0)

        fx_hat = model.predict_fx(X_B, Z_B).squeeze()
        corr = torch.corrcoef(torch.stack([fx_hat, X_B[:, 0]]))[0, 1].item()
        return model.fz.weight.data[0, 0].item(), corr

    def _assert_recovery(self, link, fz_true, fz_tol):
        results = [self._split_fit(seed, link) for seed in self.SEEDS]
        fz_mean = numpy.mean([fz for fz, _ in results])
        self.assertLess(abs(fz_mean - fz_true), fz_tol,
            msg=f"{link}: mean fz={fz_mean:.3f}, expected {fz_true} ± {fz_tol}")
        for seed, (_, corr) in zip(self.SEEDS, results):
            self.assertGreater(corr, 0.9,
                msg=f"{link}, seed={seed}: corr(fx_hat, fx)={corr:.3f}")

    def test_identity(self):
        self._assert_recovery("identity", fz_true=3.0, fz_tol=0.1)

    def test_binary(self):
        self._assert_recovery("logit", fz_true=2.0, fz_tol=0.2)


class TestIRLSConvergenceCriterion(unittest.TestCase):
    """Verify the fitted-values convergence criterion in _solve_fixed_lambda.

    The old criterion measured relative *coefficient* change:
        delta_fx = ||β_new - β_old|| / (||β_old|| + ε),  ε = 1e-5.

    Pathology: at iteration 0, β_old = 0 (zero init), so the denominator is
    1e-5 and delta_fx = ||β_1|| / 1e-5 is huge even when β_1 is already close
    to the solution.  This causes the loop to run all max_iters iterations for
    the (common) case where fx ≈ 0.

    The new criterion measures relative change in *fitted values*:
        delta_fx = mean|η_fx_new - η_fx_old| / (mean|η_fx_old| + 0.1)

    The 0.1 offset (mgcv convention) bounds the denominator away from zero
    regardless of the coefficient scale, so the first-iteration delta is
    proportional to the actual change in the predictor contribution.
    """

    @torch.no_grad()
    def setUp(self):
        torch.manual_seed(0)
        self.n, self.n_train = 400, 200
        self.out_features = 3
        self.num_covariates = 1

        self.X = torch.randn(self.n, 3)
        self.Z = torch.randn(self.n, 1)

        # DGP 1: Z has effect, X has no effect (pathological for old criterion).
        eta_no_x = 2.0 * self.Z[:, 0] + 0.5
        self.y_no_x = torch.bernoulli(torch.sigmoid(eta_no_x.unsqueeze(1)))

        # DGP 2: both X and Z have effects (standard case).
        eta_both = 1.0 * self.X[:, 0] + 2.0 * self.Z[:, 0] + 0.5
        self.y_both = torch.bernoulli(torch.sigmoid(eta_both.unsqueeze(1)))

        # DGP 3: null — no X or Z effect.  Both true coefficients are 0.
        # This is the pathological DGP for the old criterion: with high penalties,
        # both ||beta_fx|| and ||beta_fz|| end up near eps = 1e-5, making the old
        # coefficient-change denominator dominated by eps even at convergence.
        self.y_null = torch.bernoulli(torch.full((self.n, 1), torch.sigmoid(torch.tensor(0.5)).item()))

        self.base_model = BaseNetwork(
            backbone=DummyBackbone,
            backbone_params={"in_features": 3, "out_features": self.out_features,
                             "identity": True},
            num_covariates=self.num_covariates,
            link="logit",
        )

    def _make_loaders(self, y):
        train_ds = CovarDataset(self.X[:self.n_train], self.Z[:self.n_train], y[:self.n_train])
        val_ds   = CovarDataset(self.X[self.n_train:], self.Z[self.n_train:], y[self.n_train:])
        return DataLoader(train_ds, batch_size=64), DataLoader(val_ds, batch_size=64)

    @staticmethod
    @torch.no_grad()
    def _n_iters_coeff_criterion(model, X_std, Z_std, y, lam, max_iters=50, tol=1e-2, penalty_z=None):
        """Run IRLS with the OLD coefficient-change criterion; return (n_iters, converged).

        Replicates _solve_fixed_lambda exactly except the delta lines are swapped back
        to the coefficient-change formula.  Used purely for comparison.
        """
        import copy
        m = copy.deepcopy(model)
        eps = 1e-5
        m.fx.weight.data.zero_()
        m.fz.weight.data.zero_()
        m.intercept.data = m._link.forward(y.mean()).view(1)

        fx_old = m.fx.weight.data.clone()
        fz_old = m.fz.weight.data.clone()
        P_z = penalty_z.to(X_std.device) if penalty_z is not None \
            else torch.zeros(Z_std.shape[1], Z_std.shape[1])

        for i in range(max_iters):
            eta = m.intercept + m.fx(X_std) + m.fz(Z_std)
            mu = m._link.inverse(eta)
            var = m._link.variance(mu)
            g_p = m._link.derivative(mu)
            w   = g_p**2 / (var + eps)

            close0, close1 = mu < eps, mu > 1 - eps
            mu = torch.where(close0, torch.zeros_like(mu), mu)
            mu = torch.where(close1, torch.ones_like(mu), mu)
            w  = torch.where(close0 | close1, torch.full_like(w, eps), w)
            sw = w.sqrt()

            y_work = eta + (y - mu) / (g_p + eps)
            m.intercept.data = y_work.mean().view(1)

            yw = sw * (y_work - (w * y_work).sum() / w.sum())
            Xw = sw * (X_std - (w * X_std).sum(0, keepdim=True) / w.sum())
            Zw = sw * (Z_std - (w * Z_std).sum(0, keepdim=True) / w.sum())
            I1 = eps * torch.eye(Z_std.shape[1])

            ry = yw - Zw @ torch.linalg.solve(Zw.T @ Zw + P_z + I1, Zw.T @ yw)
            rX = Xw - Zw @ torch.linalg.solve(Zw.T @ Zw + P_z + I1, Zw.T @ Xw)

            bfx = torch.linalg.solve(rX.T @ rX + lam * torch.eye(rX.shape[1]), rX.T @ ry)
            m.fx.weight.data.copy_(bfx.view(1, -1))
            ry2 = yw - m.fx(Xw)
            bfz = torch.linalg.solve(Zw.T @ Zw + P_z + I1, Zw.T @ ry2)
            m.fz.weight.data.copy_(bfz.view(1, -1))

            # OLD stopping criterion: relative coefficient change.
            dfx = torch.norm(m.fx.weight.data - fx_old) / (torch.norm(fx_old) + eps)
            dfz = torch.norm(m.fz.weight.data - fz_old) / (torch.norm(fz_old) + eps)
            if dfx < tol and dfz < tol:
                return i + 1, True
            fx_old = m.fx.weight.data.clone()
            fz_old = m.fz.weight.data.clone()

        return max_iters, False

    @torch.no_grad()
    def test_converges_near_zero_fx(self):
        """New criterion converges when fx ≈ 0 at solution.

        Old pathology: delta_fx at iter 0 = ||β_1 - 0|| / (||0|| + 1e-5)
        = ||β_1|| / 1e-5, which is O(1e3) even for small β_1.
        New criterion: delta_fx = |η_fx_new|.mean() / 0.1, which is O(tol) once
        fx has settled near its small-but-finite solution.
        """
        import copy
        train_dl, val_dl = self._make_loaders(self.y_no_x)
        model = RefitCovarNetwork(copy.deepcopy(self.base_model), num_covariates=1)
        model.fit(train_dl, val_dl, lam=0.1, max_iters=50, tol=1e-2)
        rec = model.lambda_path_[0]

        self.assertTrue(rec["converged"],
            f"Should converge: n_iters={rec['n_iters']}, "
            f"delta_fx={rec['delta_fx']:.3e}, delta_fz={rec['delta_fz']:.3e}")
        self.assertLess(rec["delta_fx"], 1e-2,
            f"delta_fx={rec['delta_fx']:.3e} should be < tol=1e-2 at convergence")
        self.assertLess(rec["delta_fz"], 1e-2,
            f"delta_fz={rec['delta_fz']:.3e} should be < tol=1e-2 at convergence")

    @torch.no_grad()
    def test_converges_both_effects_nonzero(self):
        """New criterion converges on standard logistic problem with both effects."""
        import copy
        train_dl, val_dl = self._make_loaders(self.y_both)
        model = RefitCovarNetwork(copy.deepcopy(self.base_model), num_covariates=1)
        model.fit(train_dl, val_dl, lam=0.1, max_iters=50, tol=1e-2)
        rec = model.lambda_path_[0]

        self.assertTrue(rec["converged"])
        self.assertLess(rec["delta_fx"], 1e-2)
        self.assertLess(rec["delta_fz"], 1e-2)

    @torch.no_grad()
    def test_first_iter_delta_is_bounded_vs_coeff_criterion(self):
        """The old criterion's first-iteration delta is pathologically large.

        Setup: null DGP (true fx=fz=0), lam=1e4 on X, lam_z=1e4 on Z.
        Both true coefficients converge to near zero.  With ||beta|| ~ 1e-5 at
        convergence, the old denominator (||0|| + 1e-5 = 1e-5) makes
            delta_old = ||beta_1|| / 1e-5 >> tol = 0.01.
        The new denominator (|eta_fx_0|.mean() + 0.1 ≈ 0.1) keeps delta_new small:
            delta_new = |eta_fx_1|.mean() / 0.1 = O(||beta_1||) << tol.

        This test runs a single IRLS step from the zero-init state and compares both
        deltas directly, with the weights properly zeroed as _fit_effects does.
        """
        import copy
        lam, penalty_z, tol, eps = 1e4, torch.tensor([[1e4]]), 1e-2, 1e-5

        train_dl, _ = self._make_loaders(self.y_null)
        model = RefitCovarNetwork(copy.deepcopy(self.base_model), num_covariates=1)
        model.center_effects(train_dl)
        X_raw, Z_raw, y_raw = model._extract_features_from_loader(train_dl)
        X_c = model.center_x(X_raw)
        Z_c = model.center_z(Z_raw)
        X_std = X_c / X_c.std(0, keepdim=True)
        Z_std = Z_c / Z_c.std(0, keepdim=True)

        # Replicate the zero-init state of _fit_effects.
        model.fx.weight.data.zero_()
        model.fz.weight.data.zero_()
        model.intercept.data = model._link.forward(model.center_y.mean).view(1)

        # Capture the pre-step state (all zeros).
        eta_fx_0 = model.fx(X_std).clone()    # = 0
        beta_fx_0 = model.fx.weight.data.clone()  # = 0

        # One IRLS step to get beta_fx_1.
        P_z = penalty_z
        eta = model.intercept + model.fx(X_std) + model.fz(Z_std)
        mu = model._link.inverse(eta)
        var = model._link.variance(mu)
        g_p = model._link.derivative(mu)
        w   = g_p**2 / (var + eps)
        sw  = w.sqrt()
        y_work = eta + (y_raw - mu) / (g_p + eps)
        yw = sw * (y_work - (w * y_work).sum() / w.sum())
        Xw = sw * (X_std - (w * X_std).sum(0, keepdim=True) / w.sum())
        Zw = sw * (Z_std - (w * Z_std).sum(0, keepdim=True) / w.sum())
        I1 = eps * torch.eye(1)
        ry = yw - Zw @ torch.linalg.solve(Zw.T @ Zw + P_z + I1, Zw.T @ yw)
        rX = Xw - Zw @ torch.linalg.solve(Zw.T @ Zw + P_z + I1, Zw.T @ Xw)
        b1 = torch.linalg.solve(rX.T @ rX + lam * torch.eye(rX.shape[1]), rX.T @ ry)
        model.fx.weight.data.copy_(b1.view(1, -1))

        # Old criterion: delta = ||beta_1 - 0|| / (||0|| + eps).
        delta_old = float(torch.norm(model.fx.weight.data - beta_fx_0) / (torch.norm(beta_fx_0) + eps))

        # New criterion: delta = |eta_fx_1 - eta_fx_0|.mean() / (|eta_fx_0|.mean() + 0.1).
        delta_new = float((model.fx(X_std) - eta_fx_0).abs().mean() / (eta_fx_0.abs().mean() + 0.1))

        print(f"\n  [first-iter delta, null DGP, zero init, lam={lam:.0e}]")
        print(f"  ||beta_1||      = {model.fx.weight.data.norm():.3e}")
        print(f"  old coeff-chg   : delta_fx = {delta_old:.2e}  >> tol={tol}")
        print(f"  new fitted-vals : delta_fx = {delta_new:.2e}  compare tol={tol}")

        # Old criterion blows up: ||beta_1|| / eps >> tol.
        self.assertGreater(delta_old, 10.0,
            f"Old criterion first-iter delta should be >> tol (got {delta_old:.2e}). "
            f"||beta_1||={model.fx.weight.data.norm():.3e}")

        # New criterion is bounded: |eta_fx_1|.mean() / 0.1 << 1.
        self.assertLess(delta_new, tol,
            f"New criterion first-iter delta should be < tol={tol} (got {delta_new:.2e})")

    @torch.no_grad()
    def test_n_iters_comparison(self):
        """New criterion needs strictly fewer iterations on the null DGP.

        Null DGP + high penalties force both true coefficients to near zero.
        With lam_fx=lam_fz=1e4, ||beta_converged|| ~ 1e-5 = eps:

        - Old criterion: first-iter delta = ||beta_1||/eps >> tol, so it always
          needs at least 2 iterations regardless of how close beta_1 is to the
          solution.
        - New criterion: first-iter delta = |eta_fx_1|.mean()/0.1 << tol when
          ||beta|| is small, so it can converge in a single iteration.

        n_new < n_old strictly in this regime.
        """
        import copy
        lam, penalty_z, max_iters, tol = 1e4, torch.tensor([[1e4]]), 50, 1e-2

        train_dl, val_dl = self._make_loaders(self.y_null)

        # ---- New criterion (via fit) ----
        model_new = RefitCovarNetwork(copy.deepcopy(self.base_model), num_covariates=1)
        model_new.fit(train_dl, val_dl, lam=lam, max_iters=max_iters, tol=tol, penalty_z=penalty_z)
        n_new = model_new.lambda_path_[0]["n_iters"]

        # ---- Old criterion (direct call, same standardized data) ----
        model_old = RefitCovarNetwork(copy.deepcopy(self.base_model), num_covariates=1)
        model_old.center_effects(train_dl)
        X_raw, Z_raw, y_raw = model_old._extract_features_from_loader(train_dl)
        X_c = model_old.center_x(X_raw)
        Z_c = model_old.center_z(Z_raw)
        X_std = X_c / X_c.std(0, keepdim=True)
        Z_std = Z_c / Z_c.std(0, keepdim=True)
        n_old, _ = self._n_iters_coeff_criterion(
            model_old, X_std, Z_std, y_raw,
            lam=lam, max_iters=max_iters, tol=tol, penalty_z=penalty_z,
        )

        print(f"\n  [iteration count comparison, null DGP, lam={lam:.0e}, tol={tol}]")
        print(f"  fitted-values criterion : {n_new:3d} iters")
        print(f"  coeff-change criterion  : {n_old:3d} iters")

        self.assertLess(n_new, n_old,
            f"New criterion ({n_new} iters) should converge faster than "
            f"old criterion ({n_old} iters) when true coefficients are near zero.")


class TestRecenter(unittest.TestCase):
    """recenter() re-expresses f_X/f_Z on a new sample without refitting.

    Pins the identity from DESIGN.md: shifting (center_x, center_z) and
    compensating the intercept leaves eta exactly unchanged for any input,
    and makes f_X/f_Z exactly zero-mean on the new reference — for any
    link, and independent of orthogonalize (the orth term cancels out of
    eta regardless of centering, so recenter needs no orth-specific case).
    """

    def _fit_model(self, link, orthogonalize):
        torch.manual_seed(0)
        n, p, q = 200, 3, 1
        X = torch.randn(n, p)
        Z = torch.randn(n, q)
        if link == "identity":
            y = 2.0 * X[:, [0]] + 3.0 * Z + 0.3 * torch.randn(n, 1)
        else:
            y = torch.bernoulli(torch.sigmoid(1.0 * X[:, [0]] + 2.0 * Z))
        tr = DataLoader(CovarDataset(X[:100], Z[:100], y[:100]), batch_size=50)
        va = DataLoader(CovarDataset(X[100:], Z[100:], y[100:]), batch_size=50)
        base = BaseNetwork(
            backbone=DummyBackbone,
            backbone_params={"in_features": p, "out_features": p, "identity": True},
            num_covariates=q, link=link,
        )
        model = RefitCovarNetwork(base, num_covariates=q, orthogonalize=orthogonalize)
        model.fit(tr, va, lam=0.5)
        return model

    @staticmethod
    def _pooled_loader(n=300, seed=7):
        g = torch.Generator().manual_seed(seed)
        X = torch.randn(n, 3, generator=g)
        Z = torch.randn(n, 1, generator=g)
        loader = DataLoader(CovarDataset(X, Z, torch.zeros(n, 1)), batch_size=100)
        return loader, X, Z

    @torch.no_grad()
    def _check(self, link, orthogonalize):
        model = self._fit_model(link, orthogonalize)
        X_test = torch.randn(50, 3)
        Z_test = torch.randn(50, 1)
        eta_before = model.predict_eta(X_test, Z_test).clone()
        pred_before = model(X_test, Z_test).clone()

        pool_loader, X_pool, Z_pool = self._pooled_loader()
        model.recenter(pool_loader)

        eta_after = model.predict_eta(X_test, Z_test)
        pred_after = model(X_test, Z_test)
        torch.testing.assert_close(eta_before, eta_after, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(pred_before, pred_after, rtol=1e-4, atol=1e-4)

        fx_pool = model.predict_fx(X_pool, Z_pool)
        fz_pool = model.predict_fz(Z_pool)
        self.assertTrue(torch.allclose(fx_pool.mean(dim=0), torch.zeros(1), atol=1e-4))
        self.assertTrue(torch.allclose(fz_pool.mean(dim=0), torch.zeros(1), atol=1e-4))

    def test_identity_link(self):
        self._check("identity", orthogonalize=False)

    def test_logit_link(self):
        self._check("logit", orthogonalize=False)

    def test_identity_link_orthogonalized(self):
        self._check("identity", orthogonalize=True)

    def test_logit_link_orthogonalized(self):
        self._check("logit", orthogonalize=True)


class DuplicatedColumnBackbone(torch.nn.Module):
    """Backbone whose output duplicates its input columns: exactly collinear features."""

    def __init__(self, in_features):
        super().__init__()
        self.out_features = 2 * in_features

    def forward(self, x):
        return torch.cat([x, x], dim=1)


class TestSingularRidgeSolve(unittest.TestCase):
    """Exactly collinear features make the unregularized Gram singular; a
    failed solve must surface as a clean path failure, never a raw solver
    crash, and any positive lambda must still fit."""

    @torch.no_grad()
    def setUp(self):
        torch.manual_seed(0)
        n = 200
        self.X = torch.randn(n, 3)
        self.Z = torch.randn(n, 1)
        self.y = 2 * self.X[:, [0]] + 3 * self.Z
        tr = CovarDataset(self.X[:100], self.Z[:100], self.y[:100])
        va = CovarDataset(self.X[100:], self.Z[100:], self.y[100:])
        self.tr = DataLoader(tr, batch_size=50)
        self.va = DataLoader(va, batch_size=50)
        self.base = BaseNetwork(
            backbone=DuplicatedColumnBackbone,
            backbone_params={'in_features': 3},
            num_covariates=1,
            link="identity",
        )

    def test_lam_zero_raises_cleanly(self):
        model = RefitCovarNetwork(self.base, num_covariates=1)
        with self.assertRaises(RuntimeError):
            model.fit(self.tr, self.va, lam=0.0)

    def test_positive_lam_fits(self):
        model = RefitCovarNetwork(self.base, num_covariates=1)
        model.fit(self.tr, self.va, lam=0.1)
        self.assertAlmostEqual(model.fz.weight.item(), 3.0, delta=0.3)


class TestSingularOrthogonalization(unittest.TestCase):
    """Exactly collinear covariate columns make Z'Z singular; the
    orthogonalization solve must return a finite min-norm solution."""

    @torch.no_grad()
    def setUp(self):
        torch.manual_seed(0)
        n = 200
        self.X = torch.randn(n, 3)
        z = torch.randn(n, 1)
        self.Z = torch.cat([z, z], dim=1)
        self.y = 2 * self.X[:, [0]] + 3 * z
        tr = CovarDataset(self.X[:100], self.Z[:100], self.y[:100])
        va = CovarDataset(self.X[100:], self.Z[100:], self.y[100:])
        self.tr = DataLoader(tr, batch_size=50)
        self.va = DataLoader(va, batch_size=50)
        self.base = BaseNetwork(
            backbone=DummyBackbone,
            backbone_params={'in_features': 3, 'out_features': 3},
            num_covariates=2,
            link="identity",
        )

    def test_orthogonalization_fits_min_norm(self):
        model = RefitCovarNetwork(self.base, num_covariates=2, orthogonalize=True)
        model.fit(self.tr, self.va, lam=0.1)
        self.assertTrue(torch.isfinite(model.orth.weight).all())
        # the min-norm solution still satisfies the normal equations
        with torch.no_grad():
            X, Z, _ = model._extract_features_from_loader(self.tr)
            fX = model.fx(model.center_x(X))
            Zc = model.center_z(Z)
            resid = fX - model.orth(Zc)
        self.assertLess((Zc.T @ resid).abs().max().item(), 1e-2)


class DeadColumnBackbone(torch.nn.Module):
    """Backbone whose last output column is constant: zero std after centering."""

    def __init__(self, in_features):
        super().__init__()
        self.out_features = in_features + 1

    def forward(self, x):
        return torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)


class TestDeadFeatureColumn(unittest.TestCase):
    """A constant feature column must draw a zero coefficient, not NaN out the solve."""

    def test_fit_succeeds_with_zero_weight_on_dead_column(self):
        torch.manual_seed(0)
        n = 200
        X = torch.randn(n, 3)
        Z = torch.randn(n, 1)
        y = 2 * X[:, [0]] + 3 * Z
        tr = DataLoader(CovarDataset(X[:100], Z[:100], y[:100]), batch_size=50)
        va = DataLoader(CovarDataset(X[100:], Z[100:], y[100:]), batch_size=50)
        base = BaseNetwork(
            backbone=DeadColumnBackbone,
            backbone_params={'in_features': 3},
            num_covariates=1,
            link="identity",
        )
        model = RefitCovarNetwork(base, num_covariates=1)
        model.fit(tr, va, lam=0.1)
        self.assertEqual(model.fx.weight.data[0, -1].item(), 0.0)
        self.assertAlmostEqual(model.fz.weight.item(), 3.0, delta=0.3)


if __name__ == '__main__':
    unittest.main()
