"""GLM link functions registered by name.

Each `Link` carries `inverse` (g⁻¹), `forward` (g), `derivative` (g'(μ)),
and `variance` (V(μ)).
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
