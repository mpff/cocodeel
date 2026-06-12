import torch
import torch.nn as nn


class DummyBackbone(nn.Module):
    """Minimal backbone for tests: flatten input, apply one linear map.

    `identity=True` initialises the linear map to the (rectangular)
    identity with zero bias, making the backbone an exact pass-through of
    the first `out_features` flattened input features — the analytical
    setting used to validate the posthoc ridge solve against known
    coefficients.
    """

    def __init__(self, in_features, out_features, identity=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
        if identity:
            with torch.no_grad():
                self.linear.weight.copy_(torch.eye(out_features, in_features))
                self.linear.bias.zero_()

    def forward(self, x):
        return self.linear(x.view(x.size(0), -1))
