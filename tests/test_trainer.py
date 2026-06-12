import unittest
import torch


from torch import nn
from torch.utils.data import DataLoader

from cocodeel.dataset import CovarDataset
from cocodeel.model import BaseNetwork
from cocodeel.posthoc_model import PostHocCovarNetwork
from cocodeel.benchmarking.model import CovarNetwork
from cocodeel.benchmarking.posthoc_model import SemiStructuredNetwork
from cocodeel.trainer import covar_trainer
from tests.conftest import DummyBackbone


class TestLinearTraining(unittest.TestCase):

    @torch.no_grad()
    def setUp(self):
        torch.manual_seed(42)
        self.n = 300  # per sample
        self.out_features = 3
        self.num_covariates = 2
        self.model_params = {
            'link': 'identity',
            'backbone': DummyBackbone,
            'backbone_params': {'in_features': 3, 'out_features': self.out_features},
            'num_covariates': self.num_covariates
            }
        self.loss_fn = nn.MSELoss()

        # Two disjoint samples: A trains the backbone, B hosts the posthoc
        # refit (the recommended sample-split recipe).
        def sample():
            X = torch.randn(self.n, 3)
            Z = torch.randn(self.n, self.num_covariates)
            y = (2 * X[:, 0] + 3 * Z[:, 0] + 1.5).unsqueeze(1)
            tr = DataLoader(CovarDataset(X[:150], Z[:150], y[:150]),
                            batch_size=20, shuffle=True)
            va = DataLoader(CovarDataset(X[150:], Z[150:], y[150:]),
                            batch_size=20, shuffle=False)
            return X, Z, y, tr, va

        self.X, self.Z, self.y, self.train_loader, self.val_loader = sample()
        self.X_B, self.Z_B, self.y_B, self.tr_B, self.va_B = sample()

    def test_train_linear_base_model(self):
        # Fit Base model on sample A.
        model = covar_trainer(
            model=BaseNetwork,
            model_params=self.model_params,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device='cpu',
            loss_fn=self.loss_fn,
            epochs=2,
            lr=0.01
        ).eval()
        # Check prediction shape
        preds = model(self.X, self.Z)
        intercept = model.intercept.item()
        self.assertEqual(preds.shape, self.y.shape)
        # Check if preds change after centering.
        self.assertEqual(model.is_centered, torch.tensor(False))
        model.center_effects(self.train_loader)
        preds_centered = model(self.X, self.Z)
        self.assertTrue(torch.allclose(preds, preds_centered))
        self.assertEqual(model.is_centered, torch.tensor(True))
        self.assertAlmostEqual(model.intercept.item(), intercept + model.fx(model.center_x.mean))
        # Posthoc refit on the disjoint sample B: intercept centres on B and
        # the linear effects are recovered (noiseless DGP).
        posthoc_model = PostHocCovarNetwork(
            model=model,
            num_covariates=self.num_covariates
        )
        posthoc_model.fit(self.tr_B, self.va_B)
        self.assertAlmostEqual(posthoc_model.intercept.item(), self.y_B[:150].mean().item(), places=4)
        self.assertAlmostEqual(posthoc_model.fz.weight.data[0, 0].item(), 3.0, places=2)

    def test_train_linear_model(self):
        # Fit end-to-end covariate model (single sample: no posthoc refit here).
        model = covar_trainer(
            model=CovarNetwork,
            model_params=self.model_params,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device='cpu',
            loss_fn=self.loss_fn,
            epochs=2,
            lr=0.01
        ).eval()
        # Check prediction shape
        intercept = model.intercept.item()
        preds = model(self.X, self.Z)
        self.assertEqual(preds.shape, self.y.shape)
        # Check if preds change after centering.
        self.assertEqual(model.is_centered, torch.tensor(False))
        model.center_effects(self.train_loader)
        preds_centered = model(self.X, self.Z)
        self.assertTrue(torch.allclose(preds, preds_centered))
        self.assertEqual(model.is_centered, torch.tensor(True))
        self.assertAlmostEqual(model.intercept.item(), intercept + model.fx(model.center_x.mean) + model.fz(model.center_z.mean), places=4)
        # Fit posthoc model and check centering equal to mean of y.
        posthoc_model = SemiStructuredNetwork(
            model=model
        )
        posthoc_model.fit(self.train_loader)
        preds_posthoc = posthoc_model(self.X, self.Z)
        self.assertAlmostEqual(posthoc_model.intercept.item(), model.intercept.item(), places=4)
        self.assertEqual(preds_posthoc.shape, self.y.shape)
        self.assertTrue(torch.allclose(preds_posthoc, preds_centered, atol=1e-6))




class TestLogisticTraining(unittest.TestCase):

    @torch.no_grad()
    def setUp(self):
        torch.manual_seed(42)
        self.n = 1000  # per sample
        self.out_features = 3
        self.num_covariates = 2
        self.model_params = {
            'link': 'logit',
            'backbone': DummyBackbone,
            'backbone_params': {'in_features': 3, 'out_features': self.out_features},
            'num_covariates': self.num_covariates
            }
        self.loss_fn = nn.BCELoss()

        # Two disjoint samples: A trains the backbone, B hosts the posthoc
        # refit (the recommended sample-split recipe).
        def sample():
            X = torch.randn(self.n, 3)
            Z = torch.randn(self.n, self.num_covariates)
            eta = 2 * X[:, 0] + 3 * Z[:, 0] + 1.5
            y = torch.bernoulli(torch.sigmoid(eta.unsqueeze(1)))
            tr = DataLoader(CovarDataset(X[:600], Z[:600], y[:600]),
                            batch_size=20, shuffle=True)
            va = DataLoader(CovarDataset(X[600:], Z[600:], y[600:]),
                            batch_size=20, shuffle=False)
            return X, Z, y, tr, va

        self.X, self.Z, self.y, self.train_loader, self.val_loader = sample()
        self.X_B, self.Z_B, self.y_B, self.tr_B, self.va_B = sample()

    def test_train_logistic_base_model(self):
        # Fit Base model on sample A.
        model = covar_trainer(
            model=BaseNetwork,
            model_params=self.model_params,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device='cpu',
            loss_fn=self.loss_fn,
            epochs=50,
            lr=0.01
        ).eval()
        # Check prediction shape
        preds = model(self.X, self.Z)
        intercept = model.intercept.item()
        self.assertEqual(preds.shape, self.y.shape)
        # Check if preds change after centering.
        self.assertEqual(model.is_centered, torch.tensor(False))
        model.center_effects(self.train_loader)
        preds_centered = model(self.X, self.Z)
        self.assertTrue(torch.allclose(preds, preds_centered))
        self.assertEqual(model.is_centered, torch.tensor(True))
        self.assertAlmostEqual(model.intercept.item(), intercept + model.fx(model.center_x.mean))
        # Posthoc refit on the disjoint sample B: IRLS runs and recovers the
        # Z effect within finite-sample noise (600 refit obs).
        posthoc_model = PostHocCovarNetwork(
            model=model,
            num_covariates=self.num_covariates
        )
        posthoc_model.fit(self.tr_B, self.va_B)
        preds_posthoc = posthoc_model(self.X_B, self.Z_B)
        self.assertEqual(preds_posthoc.shape, self.y_B.shape)
        torch.testing.assert_close(posthoc_model.fz.weight.data[0, 0],
                                   torch.tensor(3.0), rtol=0.2, atol=0.2)

    def test_train_logistic_covar_model(self):
        # Fit Post-hoc Covariate model.
        model = covar_trainer(
            model=CovarNetwork,
            model_params=self.model_params,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device='cpu',
            loss_fn=self.loss_fn,
            epochs=2,
            lr=0.01
        ).eval()
        # Check prediction shape
        preds = model(self.X, self.Z)
        intercept = model.intercept.item()
        self.assertEqual(preds.shape, self.y.shape)
        # Check if preds change after centering.
        self.assertEqual(model.is_centered, torch.tensor(False))
        model.center_effects(self.train_loader)
        preds_centered = model(self.X, self.Z)
        self.assertTrue(torch.allclose(preds, preds_centered))
        self.assertEqual(model.is_centered, torch.tensor(True))
        self.assertAlmostEqual(model.intercept.item(), intercept + model.fx(model.center_x.mean) + model.fz(model.center_z.mean), places=4)
        # Fit posthoc model and check centering equal to mean of y.
        posthoc_model = SemiStructuredNetwork(
            model=model
        )
        posthoc_model.fit(self.train_loader)
        preds_posthoc = posthoc_model(self.X, self.Z)
        self.assertAlmostEqual(posthoc_model.intercept.item(), model.intercept.item(), places=4)
        self.assertEqual(preds_posthoc.shape, self.y.shape)
        self.assertTrue(torch.allclose(preds_posthoc, preds_centered, atol=1e-6))

class TestSchedulerParam(unittest.TestCase):

    @torch.no_grad()
    def setUp(self):
        torch.manual_seed(0)
        n = 200
        X = torch.randn(n, 3)
        Z = torch.randn(n, 2)
        y = (2 * X[:, 0] + Z[:, 0]).unsqueeze(1)
        ds = CovarDataset(X, Z, y)
        self.train_loader = DataLoader(ds, batch_size=20, shuffle=True)
        self.val_loader = DataLoader(ds, batch_size=20, shuffle=False)
        self.model_params = {
            "link": "identity",
            "backbone": DummyBackbone,
            "backbone_params": {"in_features": 3, "out_features": 2},
            "num_covariates": 2,
        }

    def test_step_lr(self):
        """Non-ReduceLROnPlateau scheduler is accepted and does not error."""
        model = covar_trainer(
            model=BaseNetwork,
            model_params=self.model_params,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device="cpu",
            loss_fn=nn.MSELoss(),
            epochs=3,
            lr=0.01,
            scheduler=torch.optim.lr_scheduler.StepLR,
            scheduler_kwargs={"step_size": 1, "gamma": 0.5},
        )
        self.assertIsNotNone(model)

    def test_reduce_lr_custom(self):
        """Custom ReduceLROnPlateau kwargs override the default."""
        model = covar_trainer(
            model=BaseNetwork,
            model_params=self.model_params,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device="cpu",
            loss_fn=nn.MSELoss(),
            epochs=3,
            lr=0.01,
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
            scheduler_kwargs={"patience": 1, "factor": 0.8},
        )
        self.assertIsNotNone(model)

    def test_default_behaviour_unchanged(self):
        """Omitting scheduler keeps the original ReduceLROnPlateau behaviour."""
        model = covar_trainer(
            model=BaseNetwork,
            model_params=self.model_params,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device="cpu",
            loss_fn=nn.MSELoss(),
            epochs=3,
            lr=0.01,
        )
        self.assertIsNotNone(model)


class TestDeterminism(unittest.TestCase):
    """Same seed ⇒ bit-identical pipeline on CPU.

    Covers every stochastic channel of the standard workflow: data draw,
    weight init, train-loader shuffling (trainer), and the posthoc refit.
    GPU runs are NOT claimed to be bit-exact (CUDA nondeterminism).
    """

    def _run_pipeline(self, seed):
        torch.manual_seed(seed)
        n = 240
        X = torch.randn(n, 3)
        Z = torch.randn(n, 2)
        y = (2 * X[:, 0] + 3 * Z[:, 0] + 0.5 * torch.randn(n)).unsqueeze(1)
        tr = DataLoader(CovarDataset(X[:120], Z[:120], y[:120]),
                        batch_size=20, shuffle=True)
        va = DataLoader(CovarDataset(X[120:], Z[120:], y[120:]),
                        batch_size=20, shuffle=False)
        model_params = {
            "link": "identity",
            "backbone": DummyBackbone,
            "backbone_params": {"in_features": 3, "out_features": 3},
            "num_covariates": 2,
        }
        base = covar_trainer(BaseNetwork, model_params, tr, va, device="cpu",
                             loss_fn=nn.MSELoss(), epochs=3, lr=0.01)
        posthoc = PostHocCovarNetwork(base, num_covariates=2)
        posthoc.fit(tr, va, n_lambdas=3)
        return posthoc.state_dict()

    def test_same_seed_same_state_dict(self):
        sd1 = self._run_pipeline(seed=123)
        sd2 = self._run_pipeline(seed=123)
        self.assertEqual(sd1.keys(), sd2.keys())
        for key in sd1:
            self.assertTrue(
                torch.equal(sd1[key], sd2[key]),
                f"state_dict['{key}'] differs across identically seeded runs",
            )


if __name__ == '__main__':
    unittest.main()
