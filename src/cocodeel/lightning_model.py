import lightning as pl
import torch.nn as nn


class CovarNeuralNetwork(pl.LightningModule):

    def __init__(self, backbone, output_func, loss_func,
                 num_covars=0, backbone_params={},
                 optimizer_params={}, scheduler_params={}):
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
        self.num_covars = num_covars
        self.num_features = self.backbone.num_features
        # Set optimizer parameters.
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params
        # Initialise last layer of the network.
        self.deep_predictor = nn.Linear(self.num_features, 1, bias=True)
        if self.num_covars > 0:
            self.struct_predictor = nn.Linear(self.num_covars, 1, bias=False)
        else:
            self.struct_predictor = lambda X: 0
        # Save hyperparameters to checkpoint.
        self.save_hyperparameters("num_covars", "num_features", "optimizer_params", "scheduler_params")

    def forward(self, U, X):
        H = self.backbone(U)
        eta = self.deep_predictor(H) + self.struct_predictor(X)
        return self.output_func(eta)

    def loss(self, U, X, y):
        H = self.backbone(U)
        eta = self.deep_predictor(H) + self.struct_predictor(X)
        return self.loss_func(eta.squeeze(), y)

    # PyTorch Lightning hooks
    def training_step(self, batch, batch_idx):
        loss = self.loss(batch["image"], batch["covar"], batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), **self.optimizer_params)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.scheduler_params)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
