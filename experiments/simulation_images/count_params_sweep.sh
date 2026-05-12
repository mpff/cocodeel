#!/usr/bin/env bash
source activate dl-mri
cd "$(dirname "$0")"
for q in 2 4 8 16 32 64 128 256 512 1024; do
    python -c "from count_params import BaseNetwork, TrafficBackbone, count; net = BaseNetwork(backbone=TrafficBackbone, backbone_params={'out_features': $q}, num_covariates=0, link='identity'); print(f'q={$q:>4}: {count(net):>8,} params')"
done
