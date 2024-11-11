import torch

class TestCaseBackbone(torch.nn.Module):
    def __init__(self, num_features=16):
        super().__init__()
        self.num_features = num_features

    def forward(self, x):
        return x
