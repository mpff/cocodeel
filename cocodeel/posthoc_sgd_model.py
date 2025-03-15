import copy
import torch
import lightning

class PostHocSGDModel(lightning.LightningModule):
    def __init__(self, model, train_dataloader, orthogonalize, num_covars,
                 optimizer, optimizer_params={}, scheduler=None, scheduler_params=None
                 ):
        super().__init__()
        self.model = copy.deepcopy(model)  # Pre-trained NeuralNetwork model
        self.model.num_covars = num_covars
        self.orthogonalize = orthogonalize
        self.loss_func = model.loss_func
        self.output_func = model.output_func
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params

        # Freeze Backbone and replace last layer.
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        self.model.deep_predictor = torch.nn.Linear(self.model.num_features, 1, bias=False)
        self.model.struct_predictor = torch.nn.Linear(self.model.num_covars, 1, bias=False)

        # Estimate orthogonalization of features.
        self.model.ortho_parameters = torch.zeros((self.model.num_covars, self.model.num_features), device=self.model.device)
        self.model.ortho_parameters.requires_grad = False
        if self.orthogonalize:
            self._orth_estimation(train_dataloader)

        self.save_hyperparameters()

    def _orth_estimation(self, data_loader):
        # See Rügamer (2023) ... Algorithm ...
        xtx = torch.zeros((self.model.num_covars, self.model.num_covars), device=self.model.device)
        xth = torch.zeros((self.model.num_covars, self.model.num_features), device=self.model.device)
        for batch in data_loader:
            u = batch["image"].to(self.model.device)
            x = batch["covar"].to(self.model.device)
            h = self.model.backbone(u)
            xtx += x.T @ x
            xth += x.T @ h
            self.model.ortho_parameters.data = torch.linalg.solve(xtx, xth)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optimizer_params)
        if self.scheduler is None:
            return optimizer
        else:
            scheduler = self.scheduler(optimizer, **self.scheduler_params)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler, "monitor": "val_loss"}

    def forward(self, u, x):
        h = self.model.backbone(u)
        if self.orthogonalize:
            h = h - x @ self.model.ortho_parameters
        eta = self.model.deep_predictor(h) + self.model.struct_predictor(x)
        return self.model.output_func(eta)

    def training_step(self, batch, batch_idx):
        u, x, y = batch["image"], batch["covar"], batch["label"]
        yhat = self(u, x)
        loss = self.loss_func(yhat.squeeze(), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        u, x, y = batch["image"], batch["covar"], batch["label"]
        yhat = self(u, x)
        loss = self.loss_func(yhat.squeeze(), y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        u, x, y = batch["image"], batch["covar"], batch["label"]
        yhat = self(u, x)
        loss = self.loss_func(yhat.squeeze(), y)
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        u, x, y = batch["image"], batch["covar"], batch["label"]
        yhat = self(u, x)
        return yhat

    def predict_deep(self, batch, batch_idx=None):
        u, x = batch["image"], batch["covar"]
        h = self.model.backbone(u)
        if self.orthogonalize:
            h = h - x @ self.model.ortho_parameters
        return self.model.deep_predictor(h)

    def predict_struct(self, batch, batch_idx=None):
        x = batch["covar"]
        return self.model.struct_predictor(x)

    def struct_coefs(self):
        return self.model.struct_predictor.weight.detach().numpy().squeeze()