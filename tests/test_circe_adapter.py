import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

# importing the adapter puts external/circe on sys.path
from cocodeel.benchmarking.circe_adapter import (
    circe_fit, circe_predict, CirceRosterModel, _featurizer_arch,
)
from model.network import Network
from utils import losses

from experiments.simulation.common.backbone import TrafficBackbone


class TestFeaturizerConfig(unittest.TestCase):

    def test_matches_traffic_backbone(self):
        # the vendored Network built from our config must be architecture-
        # identical to TrafficBackbone: same parameter shapes, same output
        net = Network(_featurizer_arch(q=32))
        ref = TrafficBackbone(out_features=32)
        net_shapes = sorted(tuple(p.shape) for p in net.parameters())
        ref_shapes = sorted(tuple(p.shape) for p in ref.parameters())
        self.assertEqual(net_shapes, ref_shapes)

        x = torch.randn(2, 1, 20, 60)
        self.assertEqual(net(x).shape, (2, 32))


class TestVendoredLosses(unittest.TestCase):

    def test_gaussian_kernel_exact(self):
        X = torch.tensor([[0.0], [1.0], [3.0]])
        K = losses.gaussian_kernel(X, sigma2=2.0)
        expected = torch.exp(-torch.tensor([
            [0.0, 1.0, 9.0], [1.0, 0.0, 4.0], [9.0, 4.0, 0.0]]) / 2.0)
        self.assertTrue(torch.allclose(K, expected, atol=1e-6))

    def test_unbiased_hsic_matches_direct_sums(self):
        # vendored vectorized estimator vs the Song et al. (2012) formula
        # written as explicit sums
        torch.manual_seed(0)
        n = 12
        Kx = losses.gaussian_kernel(torch.randn(n, 3), sigma2=1.0)
        Ky = losses.gaussian_kernel(torch.randn(n, 2), sigma2=1.0)
        Kxt = Kx - torch.diag(torch.diag(Kx))
        Kyt = Ky - torch.diag(torch.diag(Ky))
        term1 = sum(Kxt[i, j] * Kyt[i, j] for i in range(n) for j in range(n))
        term2 = sum(Kxt[i, :].sum() * Kyt[i, :].sum() for i in range(n))
        term3 = Kxt.sum() * Kyt.sum()
        direct = (term1 - 2.0 / (n - 2) * term2
                  + term3 / ((n - 1) * (n - 2))) / (n * (n - 3))
        self.assertAlmostEqual(losses.hsic_matrices(Kx, Ky).item(), direct.item(), places=5)

    def test_circe_estimate_near_zero_under_independence(self):
        # features independent of (Z, Y): the unbiased estimate fluctuates
        # around zero instead of picking up spurious dependence
        torch.manual_seed(0)
        m, n = 64, 256
        Y_h, Z_h = torch.rand(m, 1), torch.rand(m, 1)
        Ky = losses.gaussian_kernel(Y_h, sigma2=1.0)
        Kz = losses.gaussian_kernel(Z_h, sigma2=1.0)
        W_all = torch.linalg.solve(Ky + 1.0 * torch.eye(m), torch.cat([torch.eye(m), Kz], dim=1))
        W_1, W_2 = W_all[:, :m], W_all[:, m:]
        feats = torch.randn(n, 4)
        Y, Z = torch.rand(n, 1), torch.rand(n, 1)
        val = losses.circe_estimate(
            feats, Z, Z_h, Y, Y_h, W_1, W_2,
            kernelX='gaussian', kernelX_params={'sigma2': 1.0},
            kernelZ='gaussian', kernelZ_params={'sigma2': 1.0},
            kernelY='gaussian', kernelY_params={'sigma2': 1.0},
            biased=False, cond_cov=False)
        self.assertLess(abs(val.item()), 0.05)


class TestCirceFit(unittest.TestCase):

    @torch.no_grad()
    def setUp(self):
        torch.manual_seed(0)
        n_tr, n_va = 96, 60
        self.X_tr = torch.rand(n_tr, 1, 20, 60)
        self.Z_tr = torch.rand(n_tr, 1)
        self.y_tr = self.X_tr.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1) + 0.5 * self.Z_tr
        self.X_va = torch.rand(n_va, 1, 20, 60)
        self.Z_va = torch.rand(n_va, 1)
        self.y_va = self.X_va.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1) + 0.5 * self.Z_va
        self.workdir = tempfile.mkdtemp(prefix="circe_test_")

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def test_fit_and_predict(self):
        # keep the test off the GPUs: the vendored BaseTrainer hardcodes
        # cuda-if-available
        with mock.patch("torch.cuda.is_available", return_value=False):
            trainer = circe_fit(
                self.X_tr, self.Z_tr, self.y_tr, self.X_va, self.Z_va, self.y_va,
                lam=1.0, epochs=3, heldout_ratio=0.1, workdir=self.workdir)

        # heldout carve: 10% of 96 -> m=10, train shrinks to 86
        self.assertEqual(trainer.Y_heldout.shape, (10, 1))
        self.assertEqual(len(trainer.dataloaders["train"].dataset), 86)

        # KRR precompute solves (Ky + lambda I) W_1 = I on the heldout set,
        # with sigma2/lambda as selected by the vendored LOO routine
        Ky = losses.gaussian_kernel(trainer.Y_heldout.cpu(), **trainer.kernel_y_args)
        lhs = (Ky + trainer.model_cfg.ridge_lambda * torch.eye(10)) @ trainer.W_1.cpu()
        self.assertTrue(torch.allclose(lhs, torch.eye(10), atol=1e-3))

        # best checkpoint was written and predictions have the right shape
        self.assertTrue((__import__("pathlib").Path(self.workdir) / "best.pth").exists())
        preds = circe_predict(trainer, self.X_va)
        self.assertEqual(preds.shape, (60, 1))
        self.assertTrue(torch.isfinite(preds).all())

        # roster interface: fx is the output centered on the reference
        # sample, fz is identically zero, eta is unchanged
        from torch.utils.data import DataLoader
        from cocodeel.dataset import CovarDataset
        ref = DataLoader(CovarDataset(self.X_tr, self.Z_tr, self.y_tr), batch_size=32)
        model = CirceRosterModel(trainer, ref).eval()
        with torch.no_grad():
            fx_tr = model.predict_fx(self.X_tr)
            self.assertAlmostEqual(fx_tr.mean().item(), 0.0, places=5)
            self.assertTrue(torch.equal(model.predict_fz(self.Z_va),
                                        torch.zeros(60, 1)))
            self.assertTrue(torch.allclose(model(self.X_va),
                                           model.predict_fx(self.X_va) + model.offset))

    def test_early_stopping_exit_is_caught(self):
        # patience=0 forces BaseTrainer.save's sys.exit at the first
        # non-improving epoch; circe_fit must survive and return
        with mock.patch("torch.cuda.is_available", return_value=False):
            trainer = circe_fit(
                self.X_tr, self.Z_tr, self.y_tr, self.X_va, self.Z_va, self.y_va,
                lam=1.0, epochs=30, patience=0, heldout_ratio=0.1, workdir=self.workdir)
        self.assertIsNotNone(circe_predict(trainer, self.X_va[:4]))


if __name__ == "__main__":
    unittest.main()
