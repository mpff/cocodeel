import torch
import lightning

class CovarNeuralNetwork(lightning.LightningModule):

    def __init__(self, backbone, output_func, loss_func, optimizer,
                 num_features=32, num_covars=1,
                 backbone_params={}, optimizer_params={}):
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
        self.output_func = output_func()
        self.loss_func = loss_func()
        self.optimizer = optimizer
        self.num_covars = num_covars
        self.num_features = num_features
        self.optimizer_params = optimizer_params
        # Initialise last layer of the network.
        # IMPORTANT: No explicit bias term, as we assume at least a covariate vector of ones!
        self.deep_predictor = torch.nn.Linear(self.num_features, 1, bias=False)
        self.struct_predictor = torch.nn.Linear(self.num_covars, 1, bias=False)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optimizer_params)
        return optimizer

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

