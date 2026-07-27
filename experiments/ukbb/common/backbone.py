"""Pretrained 3D ResNet50 backbone for UKBB T1 MRI (input 1x96x114x96, output 2048)."""
import torch
import torch.nn as nn


# ── resnet building blocks (inlined Ritter Lab 3D-ResNet) ─────────────────────
def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class ResNet(nn.Module):
    """3D ResNet feature extractor mapping (B,1,96,114,96) to a (B,C) global-pooled vector."""

    def __init__(self, block, layers, block_inplanes, n_input_channels=1,
                 conv1_t_size=7, conv1_t_stride=1, no_max_pool=False, shortcut_type='B'):
        super().__init__()
        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.conv1 = nn.Conv3d(n_input_channels, self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], shortcut_type, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion))
        layers = [block(in_planes=self.in_planes, planes=planes, stride=stride, downsample=downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        if not self.no_max_pool:
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)


# ── pretrained wrapper ────────────────────────────────────────────────────────
class ResNet50(nn.Module):
    """ResNet50 feature extractor; loads Kinetics-pretrained weights and sums RGB conv1 to one channel."""

    def __init__(self, freeze_feature_extractor=False, pretrained_model=""):
        super().__init__()
        self.num_features = 2048
        self.out_features = 2048
        self.feature_extractor = ResNet(Bottleneck, [3, 4, 6, 3], [64, 128, 256, 512])
        if pretrained_model:
            sd = torch.load(pretrained_model, weights_only=False)
            if "state_dict" in sd:
                sd = sd["state_dict"]
            if sd["conv1.weight"].shape[1] == 3:
                sd["conv1.weight"] = sd["conv1.weight"].sum(1, keepdim=True)
            # strict=False: the Kinetics checkpoint carries a classification head we drop
            self.feature_extractor.load_state_dict(sd, strict=False)
        if freeze_feature_extractor:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.feature_extractor(x)


def default_model_params(pretrained=""):
    """BaseNetwork kwargs for the UKBB ResNet50; leave pretrained empty when restoring a checkpoint."""
    return dict(backbone=ResNet50, backbone_params={"pretrained_model": pretrained},
                num_covariates=0, link="logit")
