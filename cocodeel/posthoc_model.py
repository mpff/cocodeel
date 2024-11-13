import copy
import torch
import lightning

class PostHocOrthogonalizedModel(lightning.LightningModule):

    def __init__(self, model, train_dataloader):
        """ Orthogonalizes a pre-trained model over the training data set. """
        super().__init__()
        self.model = copy.deepcopy(model)
        # Update last layer with orthogonalization.
        self.model.ortho_parameters = torch.nn.Parameter(torch.zeros((self.model.num_covars, 1)), requires_grad=False)
        with torch.no_grad():
            # See Rügamer (2023) ... Algorithm ...
            xtx = torch.zeros((self.model.num_covars, self.model.num_covars), device=self.model.device)
            xte = torch.zeros((self.model.num_covars, 1), device=self.model.device)
            for batch in train_dataloader:
                u = batch["image"].to(self.model.device)
                x = batch["covar"].to(self.model.device)
                h = self.model.backbone(u)
                eta_h = self.model.deep_predictor(h)
                xtx += x.T @ x
                xte += x.T @ eta_h
            self.model.ortho_parameters.data = torch.linalg.solve(xtx, xte)
            self.model.struct_predictor.weight.data += self.model.ortho_parameters.T

    def forward(self, u, x):
        h = self.model.backbone(u)
        eta_deep = self.model.deep_predictor(h) - x @ self.model.ortho_parameters
        eta_struct = self.model.struct_predictor(x)
        eta = eta_deep + eta_struct
        return self.model.output_func(eta)
