"""Count parameters of the DNN part (backbone + fx head + intercept).

The DNN part is the BaseNetwork: TrafficBackbone (conv-conv-pool-fc → q
features) followed by a linear fx head (q → 1, no bias) and a scalar
intercept. Centering modules carry buffers (not parameters) and are
ignored.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cocodeel.model import BaseNetwork
from experiments.simulation.common.backbone import TrafficBackbone


def count(module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def main(q: int = 32) -> None:
    net = BaseNetwork(
        backbone=TrafficBackbone,
        backbone_params={"out_features": q},
        num_covariates=0,
        link="identity",
    )

    total = count(net)
    backbone = count(net.backbone)
    fx = count(net.fx)
    intercept = net.intercept.numel()

    print(f"q = {q}")
    print(f"  TrafficBackbone:          {backbone:>7,}")
    print(f"    conv1 (1→16, 3×3):      {count(net.backbone.conv1):>7,}")
    print(f"    conv2 (16→32, 3×3):     {count(net.backbone.conv2):>7,}")
    print(f"    fc (32·4·4 → q):        {count(net.backbone.fc):>7,}")
    print(f"  fx (Linear q→1, no bias): {fx:>7,}")
    print(f"  intercept (scalar):       {intercept:>7,}")
    print("  ────────────────────────  ───────")
    print(f"  TOTAL (backbone + fx):    {total:>7,}")


if __name__ == "__main__":
    main()
