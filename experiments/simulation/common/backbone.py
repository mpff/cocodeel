import torch.nn as nn
import torch.nn.functional as F


class TrafficBackbone(nn.Module):
    """Conv-conv-pool-fc CNN mapping a traffic-light image to q features."""

    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(32 * 4 * 4, self.out_features)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
