import torch
import torch.nn as nn

from cocodeel.transform import Center

class BaseNetwork(nn.Module):
    def __init__(self, backbone, backbone_params={}):
        """ Base Network class for a model with CENTRED features and INTERCEPT.
        Parameters:
            backbone (nn.Module): the CNN backbone for feature extraction.
            backbone_params (dict): parameters to give to the backbone model.
        Methods:
            forward: defines the forward computation at every call.
        """
        super(BaseNetwork, self).__init__()
        self.backbone = backbone(**backbone_params)
        self.backbone_params = backbone_params
        self.center_x = Center(self.backbone.out_features)
        self.is_centered = False
        self.fx = nn.Linear(self.backbone.out_features, 1, bias=False)
        self.intercept = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x, z=None):
        x = self.backbone(x)
        x = self.center_x(x)
        y = self.intercept + self.fx(x)
        return y

    def center_features(self, dataloader):
        self.center_x.fit_from_loader(dataloader, self.backbone, 'X', device=next(self.parameters()).device)
        with torch.no_grad():
            self.intercept += self.fx.weight @ self.center_x.mean
        self.is_centered = True