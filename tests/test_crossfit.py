import unittest

import torch
from torch.utils.data import DataLoader

from cocodeel.crossfit import CrossFitEnsemble
from cocodeel.dataset import CovarDataset
from cocodeel.model import BaseNetwork
from cocodeel.posthoc_model import PostHocCovarNetwork
from cocodeel.trainer import covar_trainer
from tests.conftest import DummyBackbone


def _fold(seed_backbone, seed_refit, link, n=200):
    """Sample-split recipe for one fold: backbone on A, posthoc refit on B."""
    def sample(seed):
        g = torch.Generator().manual_seed(seed)
        X = torch.randn(n, 3, generator=g)
        Z = torch.randn(n, 1, generator=g)
        if link == "identity":
            y = 2.0 * X[:, [0]] + 3.0 * Z + 0.3 * torch.randn(n, 1, generator=g)
        else:
            y = torch.bernoulli(torch.sigmoid(1.0 * X[:, [0]] + 2.0 * Z))
        return X, Z, y

    XA, ZA, yA = sample(seed_backbone)
    XB, ZB, yB = sample(seed_refit)
    half = n // 2
    trA = DataLoader(CovarDataset(XA[:half], ZA[:half], yA[:half]), batch_size=50, shuffle=True)
    vaA = DataLoader(CovarDataset(XA[half:], ZA[half:], yA[half:]), batch_size=50)
    trB = DataLoader(CovarDataset(XB[:half], ZB[:half], yB[:half]), batch_size=50, shuffle=True)
    vaB = DataLoader(CovarDataset(XB[half:], ZB[half:], yB[half:]), batch_size=50)

    loss = torch.nn.MSELoss() if link == "identity" else torch.nn.BCELoss()
    base = covar_trainer(
        BaseNetwork,
        dict(backbone=DummyBackbone, backbone_params=dict(in_features=3, out_features=3), link=link),
        trA, vaA, device="cpu", loss_fn=loss, epochs=40, lr=0.01, patience=15,
    )
    model = PostHocCovarNetwork(base, num_covariates=1, orthogonalize=False)
    model.fit(trB, vaB, lam=0.1)
    return model, torch.cat([XA, XB]), torch.cat([ZA, ZB])


class TestRecenterThenAverageIdentity(unittest.TestCase):
    """The core correctness result: averaging RECENTERED, then-summed
    (intercept + f_X + f_Z) exactly reproduces averaging the ORIGINAL,
    unmodified per-fold eta — for both links. Not a design choice, a proof:
    recenter() is a lossless reparameterization, so recentering every fold
    before averaging can only change how the ensemble is decomposed, never
    what it predicts."""

    def _check(self, link):
        torch.manual_seed(0)
        m1, X1, Z1 = _fold(10, 11, link)
        m2, X2, Z2 = _fold(20, 21, link)

        X_test = torch.randn(60, 3)
        Z_test = torch.randn(60, 1)
        eta_direct = 0.5 * (m1.predict_eta(X_test, Z_test) + m2.predict_eta(X_test, Z_test))

        ensemble = CrossFitEnsemble([m1, m2])
        X_pool, Z_pool = torch.cat([X1, X2]), torch.cat([Z1, Z2])
        pool_loader = DataLoader(
            CovarDataset(X_pool, Z_pool, torch.zeros(len(X_pool), 1)), batch_size=200,
        )
        ensemble.recenter(pool_loader)
        eta_ensemble = ensemble.predict_eta(X_test, Z_test)

        torch.testing.assert_close(eta_direct, eta_ensemble, rtol=1e-4, atol=1e-4)

    def test_identity_link(self):
        self._check("identity")

    def test_logit_link(self):
        self._check("logit")


class TestEtaSpaceAveraging(unittest.TestCase):
    """Under a nonlinear link, the ensemble must average eta then apply the
    link once — not average each fold's post-link prediction, since
    mean_k(sigmoid(eta_k)) != sigmoid(mean_k(eta_k)) in general (Jensen)."""

    def test_logit_matches_sigmoid_of_mean_eta_not_mean_of_sigmoid(self):
        torch.manual_seed(0)
        m1, _, _ = _fold(30, 31, "logit")
        m2, _, _ = _fold(40, 41, "logit")
        ensemble = CrossFitEnsemble([m1, m2])

        X_test = torch.randn(40, 3)
        Z_test = torch.randn(40, 1)
        eta1 = m1.predict_eta(X_test, Z_test)
        eta2 = m2.predict_eta(X_test, Z_test)

        correct = torch.sigmoid(0.5 * (eta1 + eta2))
        wrong = 0.5 * (torch.sigmoid(eta1) + torch.sigmoid(eta2))
        pred = ensemble(X_test, Z_test)

        torch.testing.assert_close(pred, correct, rtol=1e-5, atol=1e-5)
        self.assertGreater(
            (pred - wrong).abs().max().item(), 1e-3,
            "sigmoid(mean(eta)) and mean(sigmoid(eta)) should actually differ on "
            "this batch — otherwise this test isn't exercising the distinction "
            "it's meant to check.",
        )


class TestCrossFitRecoversKnownEffects(unittest.TestCase):
    """K=2 sample-split-and-cross-fit recovers a known linear b_Z, for both
    links — parameter recovery extended from a single fold to the ensemble."""

    def _check(self, link, b_z_true, tol):
        torch.manual_seed(1)
        m1, X1, Z1 = _fold(50, 51, link)
        m2, X2, Z2 = _fold(60, 61, link)
        ensemble = CrossFitEnsemble([m1, m2])
        X_pool, Z_pool = torch.cat([X1, X2]), torch.cat([Z1, Z2])
        pool_loader = DataLoader(
            CovarDataset(X_pool, Z_pool, torch.zeros(len(X_pool), 1)), batch_size=400,
        )
        ensemble.recenter(pool_loader)

        b_z = 0.5 * (m1.fz.weight.item() + m2.fz.weight.item())
        self.assertAlmostEqual(b_z, b_z_true, delta=tol)

    def test_identity_link(self):
        self._check("identity", b_z_true=3.0, tol=0.3)

    def test_logit_link(self):
        self._check("logit", b_z_true=2.0, tol=0.5)


if __name__ == "__main__":
    unittest.main()
