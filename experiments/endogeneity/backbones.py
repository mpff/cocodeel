"""Backbone architectures for the endogeneity test.

Two backbones are used:
- MLPBackbone: two-layer MLP with ReLU, the workhorse for joint-training approaches.
- IdentityBackbone: linear-only, used by the pretrained/exogenous baseline.
"""
import torch.nn as nn


class MLPBackbone(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32), nn.ReLU(), nn.Linear(32, out_features))
        self.out_features = out_features
        self.in_features = in_features

    def forward(self, x):
        return self.net(x)


class IdentityBackbone(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.out_features = out_features
        self.in_features = in_features

    def forward(self, x):
        return self.linear(x)
