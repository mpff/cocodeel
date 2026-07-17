import unittest
import torch

from torch import nn
from torch.utils.data import DataLoader

from cocodeel.dataset import CovarDataset
from cocodeel.model import BaseNetwork
from cocodeel.trainer import covar_trainer
from cocodeel.benchmarking.adversarial_trainer import adversarial_trainer, ConfoundPredictor, _squared_corr
from tests.conftest import DummyBackbone


class TestConfoundPredictor(unittest.TestCase):

    def test_output_shape(self):
        head = ConfoundPredictor(in_features=6, num_covariates=2, hidden=4)
        features = torch.randn(5, 6)
        z_pred = head(features)
        self.assertEqual(z_pred.shape, (5, 2))


class TestSquaredCorr(unittest.TestCase):

    def test_perfect_correlation(self):
        z = torch.randn(50, 1)
        self.assertAlmostEqual(_squared_corr(z, z).item(), 1.0, places=4)

    def test_zero_correlation(self):
        torch.manual_seed(0)
        n = 5000
        z_true = torch.randn(n, 1)
        z_pred = torch.randn(n, 1)  # independent of z_true
        self.assertAlmostEqual(_squared_corr(z_true, z_pred).item(), 0.0, delta=0.02)

    def test_sums_over_covariates(self):
        z = torch.randn(50, 3)
        self.assertAlmostEqual(_squared_corr(z, z).item(), 3.0, places=4)


class TestAdversarialTrainer(unittest.TestCase):

    @torch.no_grad()
    def setUp(self):
        torch.manual_seed(0)
        self.n = 400
        self.model_params = {
            'link': 'identity',
            'backbone': DummyBackbone,
            'backbone_params': {'in_features': 3, 'out_features': 3, 'identity': True},
        }
        self.loss_fn = nn.MSELoss()

        # Concurvity DGP: Z correlates with X[:, 0], so a Z-free model's
        # f_X(X) absorbs part of f_Z(Z) (Theorem 1) and, in turn, correlates
        # with Z. This is the correlation the adversarial step should reduce.
        X = torch.randn(self.n, 3)
        Z = (0.8 * X[:, 0] + 0.6 * torch.randn(self.n)).unsqueeze(1)
        y = (2 * X[:, 0] + 3 * Z[:, 0] + 0.1 * torch.randn(self.n)).unsqueeze(1)

        self.train_loader = DataLoader(
            CovarDataset(X[:280], Z[:280], y[:280]), batch_size=40, shuffle=True
        )
        self.val_loader = DataLoader(
            CovarDataset(X[280:340], Z[280:340], y[280:340]), batch_size=40, shuffle=False
        )
        self.X_test, self.Z_test = X[340:], Z[340:]

    def test_reduces_correlation_with_covariate(self):
        # Match the task learning rate to the baseline's so both models reach
        # a comparable task-loss convergence state — otherwise a slower
        # adversarial task step alone (independent of the adversarial
        # mechanism) could explain a lower correlation. lr_adv/lr_cp raised
        # well above lr_task (source ratio: adversary faster than task) so
        # the adversarial step isn't left trailing a well-converged task
        # network; verified against 13 seeds, not just the one fixed here.
        torch.manual_seed(0)
        baseline = covar_trainer(
            model=BaseNetwork, model_params=self.model_params,
            train_loader=self.train_loader, val_loader=self.val_loader,
            device='cpu', loss_fn=self.loss_fn, epochs=200, lr=0.01, patience=200,
        ).eval()

        torch.manual_seed(0)
        adversarial = adversarial_trainer(
            model=BaseNetwork, model_params=self.model_params, num_covariates=1,
            train_loader=self.train_loader, val_loader=self.val_loader,
            device='cpu', loss_fn=self.loss_fn, epochs=200,
            lr_task=0.01, lr_adv=0.1, lr_cp=0.1, patience=200,
        ).eval()

        with torch.no_grad():
            fx_base = baseline.predict_fx(self.X_test)
            fx_adv = adversarial.predict_fx(self.X_test)
        corr_base = _squared_corr(self.Z_test, fx_base).item()
        corr_adv = _squared_corr(self.Z_test, fx_adv).item()

        self.assertLess(corr_adv, corr_base)

    def test_center_effects_after_training(self):
        # adversarial_trainer returns an uncentered BaseNetwork, same
        # two-step contract as covar_trainer — center_effects is a separate,
        # explicit call so every end-to-end benchmark is centered on the
        # same reference set before predictions are compared across methods.
        model = adversarial_trainer(
            model=BaseNetwork, model_params=self.model_params, num_covariates=1,
            train_loader=self.train_loader, val_loader=self.val_loader,
            device='cpu', loss_fn=self.loss_fn, epochs=10,
        )
        self.assertFalse(model.is_centered)
        preds_before = model(self.X_test, self.Z_test)

        model.center_effects(self.train_loader)

        self.assertTrue(model.is_centered)
        preds_after = model(self.X_test, self.Z_test)
        self.assertTrue(torch.allclose(preds_before, preds_after, atol=1e-5))
        # fx is exactly zero-mean on the set centering was fit on (train), not
        # on the held-out split, which is a different sample from the DGP.
        X_train = self.train_loader.dataset.X
        self.assertAlmostEqual(model.predict_fx(X_train).mean().item(), 0.0, places=5)

    def test_attaches_fit_history_and_confound_head(self):
        model = adversarial_trainer(
            model=BaseNetwork, model_params=self.model_params, num_covariates=1,
            train_loader=self.train_loader, val_loader=self.val_loader,
            device='cpu', loss_fn=self.loss_fn, epochs=3,
        )
        self.assertTrue(hasattr(model, 'val_losses_'))
        self.assertTrue(hasattr(model, 'best_epoch_'))
        self.assertTrue(hasattr(model, 'n_epochs_run_'))
        self.assertEqual(model.lr_history_.keys(), {"task", "adv", "cp"})
        self.assertIsInstance(model.confound_head_, ConfoundPredictor)


class TestAdversarialTrainerLogitLink(unittest.TestCase):

    @torch.no_grad()
    def setUp(self):
        torch.manual_seed(1)
        n = 300
        self.model_params = {
            'link': 'logit',
            'backbone': DummyBackbone,
            'backbone_params': {'in_features': 3, 'out_features': 3, 'identity': True},
        }
        X = torch.randn(n, 3)
        Z = torch.randn(n, 1)
        eta = 2 * X[:, 0] + 3 * Z[:, 0]
        y = torch.bernoulli(torch.sigmoid(eta.unsqueeze(1)))
        self.train_loader = DataLoader(CovarDataset(X[:200], Z[:200], y[:200]), batch_size=32, shuffle=True)
        self.val_loader = DataLoader(CovarDataset(X[200:], Z[200:], y[200:]), batch_size=32, shuffle=False)

    def test_control_cohort_runs_without_error(self):
        # Exercises the y == control_label filtering path on binary batches.
        model = adversarial_trainer(
            model=BaseNetwork, model_params=self.model_params, num_covariates=1,
            train_loader=self.train_loader, val_loader=self.val_loader,
            device='cpu', loss_fn=nn.BCELoss(), epochs=3,
        )
        preds = model(self.train_loader.dataset.X, self.train_loader.dataset.Z)
        self.assertEqual(preds.shape, self.train_loader.dataset.y.shape)

    def test_batch_too_small_for_control_cohort_is_skipped_not_errored(self):
        # batch_size=1 guarantees the y == control_label filter always leaves
        # 0 or 1 samples, forcing the `x_ctrl.size(0) >= 2` branch to skip the
        # confound-predictor/adversarial steps every batch. The task step
        # must still run and training must not error.
        tiny_loader = DataLoader(self.train_loader.dataset, batch_size=1, shuffle=True)
        model = adversarial_trainer(
            model=BaseNetwork, model_params=self.model_params, num_covariates=1,
            train_loader=tiny_loader, val_loader=self.val_loader,
            device='cpu', loss_fn=nn.BCELoss(), epochs=1,
        )
        preds = model(self.train_loader.dataset.X, self.train_loader.dataset.Z)
        self.assertEqual(preds.shape, self.train_loader.dataset.y.shape)


if __name__ == '__main__':
    unittest.main()
