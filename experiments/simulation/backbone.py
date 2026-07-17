from torch import nn as nn
from torch.nn import functional as F


class TrafficBackbone(nn.Module):
    def __init__(self, out_features):
        super(TrafficBackbone, self).__init__()
        self.out_features = out_features
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # fixed simulation_images size 32 * 4 * 4
        self.fc = nn.Linear(32 * 4 * 4, self.out_features)

    def forward(self, X):
        # X: (batch, 1, h, w)
        x = F.relu(self.conv1(X))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # flatten to (batch, 32 * 4 * 4)
        x = self.fc(x)
        return x
