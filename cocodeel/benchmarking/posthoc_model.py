import copy
import torch
import lightning

class PHORuegamer2023(lightning.LightningModule):

    def __init__(self, model, train_dataloader):
        """ Orthogonalizes a pre-trained model over the training data set. """
        super().__init__()
        self.model = copy.deepcopy(model)
        self.loss_func = model.loss_func
        self.output_func = model.output_func
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

    def test_step(self, batch, batch_idx):
        u, x, y = batch["image"], batch["covar"], batch["label"]
        yhat = self(u, x)
        loss = self.model.loss_func(yhat.squeeze(), y)
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        u, x, y = batch["image"], batch["covar"], batch["label"]
        yhat = self(u, x)
        return yhat

    def predict_deep(self, batch, batch_idx=None):
        u, x = batch["image"], batch["covar"]
        h = self.model.backbone(u)
        return self.model.deep_predictor(h) - x @ self.model.ortho_parameters

    def predict_struct(self, batch, batch_idx=None):
        x = batch["covar"]
        return self.model.struct_predictor(x)

    def struct_coefs(self):
        return self.model.struct_predictor.weight.detach().numpy().squeeze()


class PHOWeber2024(lightning.LightningModule):

    def __init__(self, model, num_covars, train_dataloader):
        """ Orthogonalizes a pre-trained model over the training data set. """
        super().__init__()
        self.model = copy.deepcopy(model)
        self.model.num_covars = num_covars
        self.loss_func = model.loss_func
        self.output_func = model.output_func
        # Update last layer with orthogonalization.
        self.model.ortho_parameters = torch.nn.Parameter(torch.zeros((self.model.num_covars, 1)), requires_grad=False)
        self.intercept = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
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
            self.intercept.data = self.model.deep_predictor.bias.data + self.model.ortho_parameters[0]
        self.model.deep_predictor.bias.data = torch.zeros(1)

    def forward(self, u, x):
        h = self.model.backbone(u)
        eta_deep = self.model.deep_predictor(h) - x @ self.model.ortho_parameters
        return self.model.output_func(self.intercept + eta_deep)

    def test_step(self, batch, batch_idx):
        u, x, y = batch["image"], batch["covar"], batch["label"]
        yhat = self(u, x)
        loss = self.model.loss_func(yhat.squeeze(), y)
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        u, x, y = batch["image"], batch["covar"], batch["label"]
        yhat = self(u, x)
        return yhat

    def predict_deep(self, batch, batch_idx=None):
        u, x = batch["image"], batch["covar"]
        h = self.model.backbone(u)
        eta_deep = self.model.deep_predictor(h) - x @ self.model.ortho_parameters
        return eta_deep

    def predict_struct(self, batch, batch_idx=None):
        x = batch["covar"]
        return self.intercept.repeat(len(batch["label"]), 1)

    def struct_coefs(self):
        return self.intercept.detach().numpy().squeeze()