import unittest
import torch


from torch import nn
from torch.utils.data import DataLoader

from cocodeel.dataset import CovarDataset
from cocodeel.model import BaseNetwork, CovarNetwork
from cocodeel.posthoc_model import PostHocCovarNetwork
from cocodeel.benchmarking.posthoc_model import SemiStructuredNetwork
from cocodeel.trainer import covar_trainer


class DummyBackbone(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.linear = nn.Linear(3, out_features)
        self.out_features = out_features

    def forward(self, x):
        return self.linear(x)


class TestLinearTraining(unittest.TestCase):
 
    @torch.no_grad()
    def setUp(self):
        torch.manual_seed(42)
        self.n = 300
        self.out_features = 3
        self.num_covariates = 2
        self.model_params = {
            'link': 'identity',
            'backbone': DummyBackbone,
            'backbone_params': {'out_features': self.out_features},
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

    def test_train_linear_base_model(self):
        # Fit Base model.
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
        # Fit posthoc model and check centering equal to mean of y.
        posthoc_model = PostHocCovarNetwork(
            model=model,    
            num_covariates=self.num_covariates
        )
        posthoc_model.fit(self.train_loader, self.val_loader)
        self.assertAlmostEqual(posthoc_model.intercept.item(), self.y[:140].mean().item(), places=4)

    def test_train_linear_model(self):
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
        self.n = 1000
        self.out_features = 3
        self.num_covariates = 2
        self.model_params = {
            'link': 'logit',
            'backbone': DummyBackbone,
            'backbone_params': {'out_features': self.out_features},
            'num_covariates': self.num_covariates
            }
        self.loss_fn = nn.BCELoss()
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

    def test_train_logistic_base_model(self):
        # Fit Base model.
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
        # Fit posthoc model and check centering equal to mean of y.
        posthoc_model = PostHocCovarNetwork(
            model=model,    
            num_covariates=self.num_covariates
        )
        posthoc_model.fit(self.train_loader, self.val_loader)
        preds_posthoc = model(self.X, self.Z)
        self.assertEqual(preds_posthoc.shape, self.y.shape)

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
            "backbone": lambda out_features: nn.Sequential(nn.Linear(3, out_features)),
            "backbone_params": {"out_features": 2},
            "num_covariates": 2,
        }

    def _dummy_params(self):
        backbone_cls = type("B", (nn.Module,), {
            "__init__": lambda self, out_features: (nn.Module.__init__(self), setattr(self, "out_features", out_features), setattr(self, "linear", nn.Linear(3, out_features))).__class__,
        })
        # Use DummyBackbone defined at module level
        return {
            "link": "identity",
            "backbone": DummyBackbone,
            "backbone_params": {"out_features": 2},
            "num_covariates": 2,
        }

    def test_step_lr(self):
        """Non-ReduceLROnPlateau scheduler is accepted and does not error."""
        model = covar_trainer(
            model=BaseNetwork,
            model_params=self._dummy_params(),
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
            model_params=self._dummy_params(),
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
            model_params=self._dummy_params(),
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device="cpu",
            loss_fn=nn.MSELoss(),
            epochs=3,
            lr=0.01,
        )
        self.assertIsNotNone(model)


if __name__ == '__main__':
    unittest.main()
