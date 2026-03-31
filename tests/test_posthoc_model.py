import unittest
import torch
import tempfile
import os

import numpy
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

from torch import nn
from torch.utils.data import DataLoader, Dataset

from cocodeel.transform import Center
from cocodeel.dataset import CovarDataset
from cocodeel.model import BaseNetwork
from cocodeel.posthoc_model import PostHocCovarNetwork

rtol, atol = 1e-2, 1e-2

class DummyBackbone(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.out_features = out_features
        self.in_features = in_features

    def forward(self, x):
        return self.linear(x)


class TestPostHocLinearCovarNetwork(unittest.TestCase):

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
        # Pre-train BaseNetwork (to simulate backbone + fx weights)
        self.base_model = BaseNetwork(
            backbone=DummyBackbone, 
            backbone_params={'in_features': self.X.shape[1], 'out_features': self.out_features}, 
            num_covariates=self.num_covariates,
            link=self.link
        )
        # Set backbone weights to be identity mapping.
        self.base_model.backbone.linear.weight.data = torch.eye(self.out_features, self.X.shape[1])
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
        model.fit(self.train_dataloader, self.val_dataloader) # Find optimal lam via GCV.
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
        model = PostHocCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
            orthogonalize=True
        )
        model.fit(self.train_dataloader, self.val_dataloader)
        fx_new = model.predict_fx(self.X[:100], self.Z[:100])
        fz_new = model.predict_fz(self.Z[:100])
        self.assertTrue(torch.allclose(fx_new.mean(dim=0), torch.zeros_like(fx_new.mean(dim=0)), atol=1e-5))
        self.assertTrue(torch.allclose(fz_new.mean(dim=0), torch.zeros_like(fz_new.mean(dim=0)), atol=1e-5))

    @torch.no_grad()
    def test_posthoc_penalty_z_shrinks_fz(self):
        # Unpenalized baseline.
        model_unpen = PostHocCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
            orthogonalize=False
        )
        model_unpen.fit(self.train_dataloader, self.val_dataloader, lam=0.0)
        fz_unpen = model_unpen.fz.weight.data[0, 0].item()

        # Large penalty_z: fz should be shrunk towards zero.
        model_pen = PostHocCovarNetwork(
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
    def test_posthoc_save_and_reload_linear(self):
        # Train posthoc model
        model = PostHocCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
            orthogonalize=True
        )
        model.fit(self.train_dataloader, self.val_dataloader)

        y_pred_before = model(self.X, self.Z)
        fx_before = model.predict_fx(self.X, self.Z)
        fz_before = model.predict_fz(self.Z)

        # Save posthoc state_dict
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "posthoc_linear.pt")
            torch.save(model.state_dict(), path)

            # Recreate BaseNetwork EXACTLY as in setUp
            base_model_reloaded = BaseNetwork(
                backbone=DummyBackbone,
                backbone_params={'in_features': self.X.shape[1], 'out_features': self.out_features},
                num_covariates=self.num_covariates,
                link=self.link
            )
            base_model_reloaded.backbone.linear.weight.data = torch.eye(self.out_features)
            base_model_reloaded.backbone.linear.bias.data.zero_()

            # Create fresh posthoc model and load state_dict
            reloaded = PostHocCovarNetwork(
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

   
class TestPostHocLogisticCovarNetwork(unittest.TestCase):

    @torch.no_grad()
    def setUp(self):
        torch.manual_seed(42)
        self.n = 5000
        self.out_features = 3
        self.num_covariates = 2
        self.link = "logit"
        # Data
        self.X = torch.randn(self.n, 3)
        self.Z = torch.randn(self.n, self.num_covariates)
        self.eta = 1 * self.X[:, 0] + 2 * self.Z[:, 0] + 0.5 # Known linear relation
        self.p = torch.sigmoid(self.eta.unsqueeze(1))  # (n, 1)
        self.y = torch.bernoulli(self.p) # (n, 1)
        self.train_dataset = CovarDataset(self.X[:2500], self.Z[:2500], self.y[:2500])
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=25)
        self.val_dataset = CovarDataset(self.X[2500:5000], self.Z[2500:5000], self.y[2500:5000])
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=25)
        # Pre-train BaseNetwork (to simulate backbone + fx weights)
        self.base_model = BaseNetwork(
            backbone=DummyBackbone, 
            backbone_params={'in_features': self.X.shape[1], 'out_features': self.out_features}, 
            num_covariates=self.num_covariates,
            link=self.link
        )
        # Set backbone weights to simulate pre-training. Only X[0] gets a weight of 1, rest is zeroed out.
        self.base_model.backbone.linear.weight.data = torch.eye(self.out_features, self.X.shape[1])
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
        model.fit(self.train_dataloader, self.val_dataloader, lam=0.0) # No penalization.
        # After fitting.
        self.assertEqual(model.is_centered, True)
        torch.testing.assert_close(model.fz.weight.data[0,0], torch.tensor(2.0).float(), rtol=rtol, atol=atol)
        torch.testing.assert_close(model.fx.weight.data[0,0], torch.tensor(1.0).float(), rtol=rtol, atol=atol)

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
        model.fit(self.train_dataloader, self.val_dataloader)
        # After fitting.
        self.assertEqual(model.is_centered, True)
        torch.testing.assert_close(model.fz.weight.data[0,0], torch.tensor(2.0).float(), rtol=rtol, atol=atol)
        torch.testing.assert_close(model.fx.weight.data[0,0], torch.tensor(1.0).float(), rtol=rtol, atol=atol)

    @torch.no_grad()
    def test_effects_are_centered(self):
        model = PostHocCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
        )
        model.fit(self.train_dataloader, self.val_dataloader)
        fx_new = model.predict_fx(self.X[:2500], self.Z[:2500])
        fz_new = model.predict_fz(self.Z[:2500])
        self.assertTrue(torch.allclose(fx_new.mean(dim=0), torch.zeros_like(fx_new.mean(dim=0)), atol=1e-5))
        self.assertTrue(torch.allclose(fz_new.mean(dim=0), torch.zeros_like(fz_new.mean(dim=0)), atol=1e-5))

    @torch.no_grad()
    def test_posthoc_penalty_z_shrinks_fz(self):
        # Unpenalized baseline.
        model_unpen = PostHocCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
            orthogonalize=False
        )
        model_unpen.fit(self.train_dataloader, self.val_dataloader, lam=0.0)
        fz_unpen = model_unpen.fz.weight.data[0, 0].item()

        # Large penalty_z: fz should be shrunk towards zero.
        model_pen = PostHocCovarNetwork(
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
    def test_posthoc_save_and_reload_logistic(self):
        # Train posthoc model
        model = PostHocCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
            orthogonalize=True
        )
        model.fit(self.train_dataloader, self.val_dataloader)

        y_pred_before = model(self.X[:2500], self.Z[:2500])
        fx_before = model.predict_fx(self.X[:2500], self.Z[:2500])
        fz_before = model.predict_fz(self.Z[:2500])

        # Save posthoc state_dict
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "posthoc_logistic.pt")
            torch.save(model.state_dict(), path)

            # Recreate BaseNetwork EXACTLY as in setUp
            base_model_reloaded = BaseNetwork(
            backbone=DummyBackbone, 
            backbone_params={'in_features': self.X.shape[1], 'out_features': self.out_features}, 
            num_covariates=self.num_covariates,
            link=self.link
        )
            base_model_reloaded.backbone.linear.weight.data = torch.eye(self.out_features)
            base_model_reloaded.backbone.linear.bias.data.zero_()

            # Create fresh posthoc model and load state_dict
            reloaded = PostHocCovarNetwork(
                model=base_model_reloaded,
                num_covariates=self.num_covariates,
                orthogonalize=True
            )
            reloaded.load_state_dict(torch.load(path))
            reloaded.eval()

        # Compare outputs
        y_pred_after = reloaded(self.X[:2500], self.Z[:2500])
        fx_after = reloaded.predict_fx(self.X[:2500], self.Z[:2500])
        fz_after = reloaded.predict_fz(self.Z[:2500])

        self.assertTrue(torch.allclose(y_pred_before, y_pred_after, atol=atol))
        self.assertTrue(torch.allclose(fx_before, fx_after, atol=atol))
        self.assertTrue(torch.allclose(fz_before, fz_after, atol=atol))
        self.assertTrue(torch.allclose(model.intercept, reloaded.intercept, atol=atol))


class TestHighDimensionalPostHocFit(unittest.TestCase):

    @torch.no_grad()
    def setUp(self):
        torch.manual_seed(42)
        self.n = 2000
        self.ntrain = self.n // 2
        self.p = 512
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
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=25)
        self.val_dataset = CovarDataset(self.X[self.ntrain:], self.Z[self.ntrain:], self.y[self.ntrain:])
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=25)
        self.train_dataset_corr = CovarDataset(self.Xcorr[:self.ntrain], self.Z[:self.ntrain], self.ycorr[:self.ntrain])
        self.train_dataloader_corr = DataLoader(self.train_dataset_corr, batch_size=25)
        self.val_dataset_corr = CovarDataset(self.Xcorr[self.ntrain:], self.Z[self.ntrain:], self.ycorr[self.ntrain:])
        self.val_dataloader_corr = DataLoader(self.val_dataset_corr, batch_size=25)
        # Pre-train BaseNetwork (to simulate backbone + fx weights)
        self.base_model = BaseNetwork(
            backbone=DummyBackbone, 
            backbone_params={'in_features': self.in_features, 'out_features': self.out_features}, 
            num_covariates=self.num_covariates,
            link=self.link
        )
        # Set backbone weights to be identity mapping.
        self.base_model.backbone.linear.weight.data = torch.eye(self.out_features, self.in_features)
        self.base_model.backbone.linear.bias.data.zero_()

    @torch.no_grad()
    def test_posthoc_identifies_effects(self):
        model = PostHocCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
            orthogonalize=True
        )
        # Before fitting.
        self.assertEqual(model.is_centered, False)
        self.assertEqual(model.intercept, torch.zeros(1))
        self.assertTrue(torch.allclose(model.orth.weight.data, torch.zeros(self.num_covariates, 1), atol=atol))
        # Fit post-hoc model.
        model.fit(self.train_dataloader, self.val_dataloader)
        # After fitting.
        self.assertEqual(model.is_centered, True)
        # Check that identified effects are close to true effects!
        torch.testing.assert_close(model.fz.weight.data[0,0], torch.tensor(3.0).float(), rtol=rtol, atol=atol)

    @torch.no_grad()
    def test_posthoc_identifies_effects_for_correlated_data(self):
        model = PostHocCovarNetwork(
            model=self.base_model,
            num_covariates=self.num_covariates,
            orthogonalize=True
        )
        # Before fitting.
        self.assertEqual(model.is_centered, False)
        self.assertEqual(model.intercept, torch.zeros(1))
        self.assertTrue(torch.allclose(model.orth.weight.data, torch.zeros(self.num_covariates, 1), atol=atol))
        # Fit post-hoc model with penalty
        model.fit(self.train_dataloader_corr, self.val_dataloader_corr)
        # After fitting.
        self.assertEqual(model.is_centered, True)
        torch.testing.assert_close(model.fz.weight.data[0,0], torch.tensor(3.0).float(), rtol=rtol, atol=atol)




class TestHighDimRegression(unittest.TestCase):
    """Consistent estimation of β_fx and β_fz when N/d ≈ 1 and confounding is present.

    Scientific question: after FWL residualization (resid_X = X - Z(Z'Z)⁻¹Z'X,
    resid_y = y - Z(Z'Z)⁻¹Z'y), can we estimate β_fx from resid_X → resid_y and β_fz
    consistently, even when N < d and corr(Z, X) is substantial?

    DGP: H = 0.3*Z + noise (corr(Z, H_j) ≈ 0.29), y = 2*H[:,0] + 3*Z + eps.
    Identity backbone (H passes through unchanged), N=2000, d=2048, N/d≈0.73.
    True β_fz=3, β_fx[0]=2, intercept=0.
    """

    @torch.no_grad()
    def setUp(self):
        torch.manual_seed(0)
        rng = numpy.random.default_rng(0)
        self.n, self.ntrain, self.d = 2000, 1500, 2048
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
            backbone_params={'in_features': self.d, 'out_features': self.d},
            num_covariates=1, link="identity"
        )
        self.base_model.backbone.linear.weight.data = torch.eye(self.d)
        self.base_model.backbone.linear.bias.data.zero_()

    def test_ridge_collapses_fx_when_lambda_dominates_spectrum(self):
        """Isotropic ridge shrinks β_fx to ~0 when λ >> σ_max²(resid_X).

        With λ=2e5 >> σ_max²(resid_X) ≈ 4096, the shrinkage factor
        σ_j / (σ_j² + λ) ≈ 0 for every singular direction. β_fx ≈ 0.
        This motivates restricting regularization to the identifiable subspace.
        """
        model = PostHocCovarNetwork(self.base_model, num_covariates=1)
        model.fit(self.train_loader, self.val_loader, lam=2e5)
        fx = model.predict_fx(self.H[:self.ntrain], self.Z[:self.ntrain])
        self.assertLess(fx.std().item(), 0.1,
            msg=f"fx_std={fx.std().item():.4f}, expected < 0.1 when λ >> σ_max²")

    def test_consistent_fz_under_high_dimensional_confounding(self):
        """β_fz should recover the true value 3.0 despite N/d ≈ 1 and corr(Z, X) ≈ 0.29.

        Any regularization bias in β_fx propagates to β_fz via the OVB mechanism:
            bias(β_fz) = (Zw'Zw)⁻¹ Zw'Xw @ (β_fx_true − β_fx_estimated)
        A consistent estimator for β_fx is required to make β_fz consistent.
        """
        model = PostHocCovarNetwork(self.base_model, num_covariates=1)
        model.fit(self.train_loader, self.val_loader)
        fz = model.fz.weight.data[0, 0].item()
        self.assertAlmostEqual(fz, 3.0, delta=0.3,
            msg=f"fz={fz:.3f}, expected ~3.0 ± 0.3")

    def test_refit_preserves_fx_signal(self):
        """Post-hoc refit retains fx signal: std(fx) > 0.5 when a real effect exists.

        After FWL residualization and ridge regression, the estimated β_fx should
        pick up the true signal in H[:,0] and produce non-trivial predictions.
        """
        model = PostHocCovarNetwork(self.base_model, num_covariates=1)
        model.fit(self.train_loader, self.val_loader)
        fx = model.predict_fx(self.H[:self.ntrain], self.Z[:self.ntrain])
        self.assertGreater(fx.std().item(), 0.5,
            msg=f"fx_std={fx.std().item():.4f}, expected > 0.5")

    def test_refit_fx_decorrelated_from_Z(self):
        """|Corr(Z, fx)| < 0.3 after refit: FWL residualization removes Z from β_fx.

        Regressing Z out of X before estimating β_fx (resid_X = X - Z(Z'Z)⁻¹Z'X)
        ensures the estimated fx is approximately orthogonal to Z. This holds whenever
        λ is not so large that fx ≈ 0.
        """
        model = PostHocCovarNetwork(self.base_model, num_covariates=1)
        model.fit(self.train_loader, self.val_loader)
        fx = model.predict_fx(self.H[:self.ntrain], self.Z[:self.ntrain]).squeeze()
        z  = self.Z[:self.ntrain].squeeze()
        corr = ((z - z.mean()) @ (fx - fx.mean())) / (
            (z - z.mean()).norm() * (fx - fx.mean()).norm() + 1e-8)
        self.assertLess(abs(corr.item()), 0.3,
            msg=f"Corr(Z, fx)={corr.item():.3f}, expected |corr| < 0.3")


if __name__ == '__main__':
    unittest.main()
