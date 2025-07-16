import torch

class IdentityBackbone(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.model = torch.nn.Identity()
    def forward(self, x):
        return self.model(x)

class ShallowBackbone(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features, out_features),
            torch.nn.ReLU()
        )
    def forward(self, x):
        return self.model(x)

class DeepBackbone(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_features, 2 * in_features),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * in_features, 2 * out_features),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * out_features, 2 * out_features),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * out_features, out_features),
            torch.nn.ReLU(),
            torch.nn.Linear(out_features, out_features),
            torch.nn.ReLU()
        )
    def forward(self, x):
        return self.model(x)