import copy
import torch
import lightning


class CovarNeuralNetwork(lightning.LightningModule):

    def __init__(self, backbone, output_func, loss_func,
                 num_covars, backbone_params, optimizer_params, scheduler_params):
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
        # IMPORTANT: No bias term, because we assume that we always have at least a covariate vector of ones!
        self.deep_predictor = torch.nn.Linear(self.num_features, 1, bias=False)
        self.struct_predictor = torch.nn.Linear(self.num_covars, 1, bias=False)
        # Save hyperparameters to checkpoint.
        self.save_hyperparameters("num_covars", "num_features", "optimizer_params", "scheduler_params")

    def linear_predictor(self, u, x):
        h = self.model.backbone(u)
        return self.model.deep_predictor(h) + self.model.struct_predictor(x)

    def forward(self, u, x):
        eta = self.linear_predictor(u, x)
        return self.output_func(eta)

    def loss(self, u, x, y):
        eta = self.linear_predictor(u, x)
        return self.loss_func(eta.squeeze(), y)

    # PyTorch Lightning hooks
    def training_step(self, batch, batch_idx):
        loss = self.loss(batch["image"], batch["covar"], batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_params)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.scheduler_params)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}


class PostHocOrthogonalizedModel(lightning.LightningModule):

    def __init__(self, model, train_dataloader):
        """ Orthogonalizes a pre-trained model (that includes covars!) over the training data set. """
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

    def linear_predictor(self, u, x):
        h = self.model.backbone(u)
        h_orth = h - x @ self.ortho_parameters
        return self.model.deep_predictor(h_orth) + self.model.struct_predictor(x)

    def forward(self, u, x):
        eta = self.linear_predictor(u, x)
        return self.model.output_func(eta)

    def loss(self, u, x, y):
        eta = self.linear_predictor(u, x)
        loss = self.model.loss_func(eta.squeeze(), y)
        return loss