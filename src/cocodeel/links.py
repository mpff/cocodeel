"""Link functions for cocodeel models.

A `Link` is a 4-tuple of pure functions describing the GLM link:

- `inverse(eta) -> mu`: g⁻¹(η). Forward pass: linear predictor → predicted mean.
- `forward(mu) -> eta`: g(μ). IRLS init: initial intercept η₀ from ȳ.
- `derivative(mu) -> g'(μ)`: IRLS reweighting (working-response weight).
- `variance(mu) -> V(μ)`: GLM variance function, also IRLS reweighting.

Numerical stabilisation constants (`+ 1e-6`) are preserved bit-for-bit from
the pre-refactor implementation so sha256 baselines of paper-relevant CSVs
do not drift.
"""
from typing import Callable, NamedTuple

import torch


class Link(NamedTuple):
    inverse:    Callable[[torch.Tensor], torch.Tensor]
    forward:    Callable[[torch.Tensor], torch.Tensor]
    derivative: Callable[[torch.Tensor], torch.Tensor]
    variance:   Callable[[torch.Tensor], torch.Tensor]


LINKS: dict[str, Link] = {
    "identity": Link(
        inverse=lambda eta: eta,
        forward=lambda mu: mu,
        derivative=lambda mu: torch.ones_like(mu),
        variance=lambda mu: torch.ones_like(mu),
    ),
    "logit": Link(
        inverse=torch.sigmoid,
        forward=lambda mu: torch.log(mu / (1 - mu + 1e-6) + 1e-6),
        derivative=lambda mu: mu * (1 - mu),
        variance=lambda mu: mu * (1 - mu),
    ),
    "log": Link(
        inverse=torch.exp,
        forward=lambda mu: torch.log(mu + 1e-6),
        derivative=lambda mu: 1 / (mu + 1e-6),
        variance=lambda mu: mu,
    ),
}
