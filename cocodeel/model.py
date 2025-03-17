import torch
import lightning

class NeuralNetwork(lightning.LightningModule):

    def __init__(self, backbone, output_func, loss_func, optimizer,
                 num_features= 32, num_covars=0, backbone_params={},
                 optimizer_params={}, scheduler=None, scheduler_params=None):
        """ Neural network without added in the last layer.
        Parameters:
            backbone (nn.Module): The backbone model for feature extraction.
            ....
        Methods:
            forward: Defines the forward computation at every call.
            ....
        """
        super().__init__()
        # Define network architecture
        self.backbone = backbone(**backbone_params)
        # Initialise last layer of the network.
        self.num_covars = num_covars
        self.num_features = num_features
        self.deep_predictor = torch.nn.Linear(self.num_features, 1, bias=True)
        self.output_func = output_func()
        self.loss_func = loss_func()
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optimizer_params)
        if self.scheduler is None:
            return optimizer
        else:
            scheduler = self.scheduler(optimizer, **self.scheduler_params)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler, "monitor": "val_loss"}

    def forward(self, u, x=None):
        h = self.backbone(u)
        eta = self.deep_predictor(h)
        return self.output_func(eta)

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
        return self.loss_func(yhat.squeeze(), y)

    def predict_step(self, batch, batch_idx):
        u, x, y = batch["image"], batch["covar"], batch["label"]
        yhat = self(u, x)
        return yhat

    def predict_deep(self, batch, batch_idx=None):
        u = batch["image"]
        h = self.backbone(u)
        return self.deep_predictor(h) - self.deep_predictor.bias

    def predict_struct(self, batch, batch_idx=None):
        return self.deep_predictor.bias.repeat(len(batch["label"]), 1)

    def struct_coefs(self):
        return self.deep_predictor.bias.detach().numpy().squeeze()


class CovarNeuralNetwork(lightning.LightningModule):

    def __init__(self, backbone, output_func, loss_func, optimizer,
                 num_features=32, num_covars=1, backbone_params={},
                 optimizer_params={}, scheduler=None, scheduler_params=None):
        """ Neural network with (or without) covariates added in the last layer.
        Parameters:
            backbone (nn.Module): The backbone model for feature extraction.
            ....
        Methods:
            forward: Defines the forward computation at every call.
            ....
        """
        super().__init__()
        # Define network architecture
        self.backbone = backbone(**backbone_params)
        # Initialise last layer of the network.
        self.num_covars = num_covars
        self.num_features = num_features
        # IMPORTANT: No explicit bias term, as we assume at least a covariate vector of ones!
        self.deep_predictor = torch.nn.Linear(self.num_features, 1, bias=False)
        self.struct_predictor = torch.nn.Linear(self.num_covars, 1, bias=False)
        self.output_func = output_func()
        self.loss_func = loss_func()
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optimizer_params)
        if self.scheduler is None:
            return optimizer
        else:
            scheduler = self.scheduler(optimizer, **self.scheduler_params)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler, "monitor": "val_loss"}

    def forward(self, u, x):
        h = self.backbone(u)
        eta = self.deep_predictor(h) + self.struct_predictor(x)
        return self.output_func(eta)

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
        return self.loss_func(yhat.squeeze(), y)

    def predict_step(self, batch, batch_idx):
        u, x, y = batch["image"], batch["covar"], batch["label"]
        yhat = self(u, x)
        return yhat

    def predict_deep(self, batch, batch_idx=None):
        u = batch["image"]
        h = self.backbone(u)
        return self.deep_predictor(h)

    def predict_struct(self, batch, batch_idx=None):
        x = batch["covar"]
        return self.struct_predictor(x)

    def struct_coefs(self):
        return self.struct_predictor.weight.detach().numpy().squeeze()