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




class TestPostHocFitWithDisjointRefitLoader(unittest.TestCase):
    """Regression test: PostHocCovarNetwork.fit works when the refit loader's
    observations are disjoint from the sample used to pretrain the backbone.

    This is the `sample-split` recipe — the backbone was fit on sample A; the
    posthoc is refitted on sample B. Exogeneity of H = phi(X_B; theta*) is then
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
            backbone_params={"in_features": 3, "out_features": self.out_features},
            num_covariates=self.num_covariates,
            link="identity",
        )
        self.base.backbone.linear.weight.data = torch.eye(self.out_features, 3)
        self.base.backbone.linear.bias.data.zero_()
        self.base = self.base.center_effects(self.tr_A)  # center on the A-sample

    @torch.no_grad()
    def test_posthoc_runs_on_disjoint_refit_sample(self):
        """Split recipe: backbone on A, posthoc refit on B. Must run and
        recover effects close to truth."""
        model = PostHocCovarNetwork(
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
    def test_posthoc_fx_centered_on_refit_sample(self):
        """After refit on B, f̂_X and f̂_Z should be mean-zero on B (the refit
        sample's centering), not on A."""
        model = PostHocCovarNetwork(
            model=self.base,
            num_covariates=self.num_covariates,
            orthogonalize=False,
        )
        model.fit(self.tr_B, self.va_B, lam=0.0)

        fx_B = model.predict_fx(self.X_B[:100], self.Z_B[:100])
        fz_B = model.predict_fz(self.Z_B[:100])
        self.assertTrue(torch.allclose(fx_B.mean(dim=0), torch.zeros_like(fx_B.mean(dim=0)), atol=1e-4))
        self.assertTrue(torch.allclose(fz_B.mean(dim=0), torch.zeros_like(fz_B.mean(dim=0)), atol=1e-4))


if __name__ == '__main__':
    unittest.main()
