import unittest
import torch

from torch import nn
from torch.utils.data import DataLoader

from cocodeel.dataset import CovarDataset
from cocodeel.model import BaseNetwork
from cocodeel.posthoc_model import PostHocCovarNetwork


# Dummy backbone with trainable parameters
class DummyBackbone(nn.Module):
    def __init__(self, in_features=4, out_features=4):
        super().__init__()
        self.layer = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        return self.layer(x)


class TestSimpleTraining(unittest.TestCase):

    def setUp(self):
        self.num_features = 4
        self.num_covariates = 2
        self.n = 1000
        self.X = torch.randn(self.n, self.num_features)
        self.Z = torch.randn(self.n, self.num_covariates)
        self.y = self.X.sum(dim=1, keepdim=True) + self.Z.sum(dim=1, keepdim=True) + 0.1 * torch.randn(self.n, 1)
        self.dataset = CovarDataset(X=self.X, Z=self.Z, y=self.y)
        self.loader = DataLoader(self.dataset, batch_size=16, shuffle=True)

    def test_training_and_posthoc(self):

        model = BaseNetwork(backbone=DummyBackbone,
                            backbone_params={"in_features": self.num_features, "out_features": self.num_features})
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        model.train()
        for epoch in range(10):
            for batch in self.loader:
                x, y = batch["X"], batch["y"]
                optimizer.zero_grad()
                preds = model(x)
                loss = loss_fn(preds, y)
                loss.backward()
                optimizer.step()
        model.eval()

        posthoc_model = PostHocCovarNetwork(model, num_covariates=self.num_covariates, train_dataloader=self.loader)

        preds = posthoc_model(self.X, self.Z)

        self.assertEqual(preds.shape, self.y.shape)

        # Check intercept close to y mean
        y_mean = self.y.mean()
        intercept = posthoc_model.intercept.squeeze()
        torch.testing.assert_close(intercept, y_mean, atol=1e-4, rtol=0)

        # Check centering of features
        centered_X = posthoc_model.center_x(posthoc_model.backbone(self.X))
        centered_Z = posthoc_model.center_z(self.Z)

        torch.testing.assert_close(
            centered_X.mean(dim=0),
            torch.zeros_like(centered_X[0]),
            atol=1e-4, rtol=0,
        )

        torch.testing.assert_close(
            centered_Z.mean(dim=0),
            torch.zeros_like(centered_Z[0]),
            atol=1e-4, rtol=0
        )

        # Check fz weights close to true values.
        fz = posthoc_model.fz.weight.squeeze()
        torch.testing.assert_close(fz, torch.ones_like(fz), atol=1e-2, rtol=0)

        # Check fx prediction close to true values.
        fx_pred = posthoc_model.fx(centered_X)
        fx_true = self.X.sum(dim=1, keepdim=True)
        fx_erro = (fx_pred - fx_true).abs().mean()
        torch.testing.assert_close(fx_erro, torch.zeros_like(fx_erro), atol=1e-1, rtol=0)



if __name__ == '__main__':
    unittest.main()
