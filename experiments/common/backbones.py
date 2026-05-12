"""Shared CNN backbones for ADNI and UKBB experiments.

Contains:
  - ADNISixtyFourBackbone: 3D-CNN for ADNI (input: 1×182×218×182 or 1×96×114×96)
  - ResNet50: Pretrained 3D ResNet50 for UKBB (input: 1×96×114×96, output: 2048)

The ResNet building blocks (conv3x3x3, Bottleneck, ResNet) are from the
Ritter Lab 3D-ResNet implementation (RoshanRane), inlined here to avoid
a separate module.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# 3D ResNet building blocks
# ═══════════════════════════════════════════════════════════════════════════════

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1,
                     stride=stride, bias=False)


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
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, block_inplanes, out_classes=1,
                 n_input_channels=1, conv1_t_size=7, conv1_t_stride=1,
                 no_max_pool=False, shortcut_type='B', widen_factor=1.0):
        super().__init__()
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels, self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
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
        layers = []
        layers.append(block(in_planes=self.in_planes, planes=planes,
                            stride=stride, downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# UKBB: Pretrained 3D ResNet50 wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class ResNet50(nn.Module):
    """3D ResNet50 backbone for UKBB (pretrained on Kinetics-200).

    Input shape:  (B, 1, 96, 114, 96)
    Output shape: (B, 2048)
    """
    def __init__(self, freeze_feature_extractor=False,
                 pretrained_model='', debug_print=False):
        super().__init__()
        self.num_features = 2048
        self.block_inplanes = [64, 128, 256, 512]
        self.block = Bottleneck
        self.feature_extractor = ResNet(
            block=self.block, layers=[3, 4, 6, 3],
            block_inplanes=self.block_inplanes)
        self.pretrained_model = pretrained_model

        if self.pretrained_model:
            state_dict = torch.load(self.pretrained_model)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            n_in_chns = state_dict['conv1.weight'].shape[1]
            if n_in_chns == 3:
                state_dict['conv1.weight'] = state_dict['conv1.weight'].sum(dim=1, keepdim=True)
            log = self.feature_extractor.load_state_dict(state_dict, strict=False)
            if debug_print:
                if log.missing_keys:
                    print(f"[WARN] missing_keys: {log.missing_keys}")
                if log.unexpected_keys and len([l for l in log.unexpected_keys if 'fc' not in l]) > 0:
                    print(f"[WARN] unexpected_keys: {log.unexpected_keys}")

        if freeze_feature_extractor:
            for name, layer in self.feature_extractor.named_parameters():
                layer.requires_grad = False
                if debug_print and "bias" not in name and "bn" not in name:
                    print(f"layer {name.replace('.weight','')}({list(layer.shape)}) was frozen")

    def forward(self, x):
        return self.feature_extractor(x)


# ═══════════════════════════════════════════════════════════════════════════════
# ADNI: Custom 3D-CNN backbone
# ═══════════════════════════════════════════════════════════════════════════════

class ADNISixtyFourBackbone(nn.Module):
    """3D-CNN backbone for ADNI.

    Input shape:  (B, 1, 182, 218, 182) or (B, 1, 96, 114, 96) if downsampled
    Output shape: (B, num_features)
    """
    def __init__(self, num_features=32, drp_rate=0.3, downsampled=False):
        super().__init__()
        self.num_features = num_features
        self.drp_rate = drp_rate
        self.downsampled = downsampled
        if self.downsampled:
            self.conv_1 = nn.Sequential(
                nn.Dropout3d(p=self.drp_rate),
                nn.Conv3d(1, 16, kernel_size=5, stride=1, padding=0),
                nn.BatchNorm3d(16), nn.ELU(),
                nn.MaxPool3d(kernel_size=3, stride=3, padding=0))
        else:
            self.conv_1 = nn.Sequential(
                nn.Dropout3d(p=self.drp_rate),
                nn.Conv3d(1, 16, kernel_size=10, stride=2, padding=0),
                nn.BatchNorm3d(16), nn.ELU(),
                nn.MaxPool3d(kernel_size=3, stride=3, padding=0))
        self.conv_2 = nn.Sequential(
            nn.Dropout3d(p=self.drp_rate),
            nn.Conv3d(16, 32, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm3d(32), nn.ELU(),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=0))
        self.conv_3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(64), nn.ELU())
        self.conv_4 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(128), nn.ELU())
        self.conv_5 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(64), nn.ELU())
        self.conv_6 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(64), nn.ELU(),
            nn.MaxPool3d(kernel_size=4, stride=2, padding=0))
        self.fc = nn.Sequential(
            nn.Linear(128, self.num_features), nn.ELU())

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
