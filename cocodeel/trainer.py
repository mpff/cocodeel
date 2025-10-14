import copy

import torch
from torch import nn as nn


def covar_trainer(model, model_params, train_loader, val_loader,
                  loss_fn=nn.MSELoss(), epochs=1000, lr=1e-3, weight_decay=1e-4, patience=12):

    model = model(**model_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience-2, factor=0.5)

    best_val_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())
    counter = 0

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x, z, y = batch["X"], batch["Z"], batch["y"]
            optimizer.zero_grad()
            if model.num_covariates > 0:
                preds = model(x, z)
            else:
                preds = model(x)
            loss = loss_fn(preds, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping.
            optimizer.step()

        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                x, z, y = batch["X"], batch["Z"], batch["y"]
                if model.num_covariates > 0:
                    preds = model(x, z)
                else:
                    preds = model(x)
                loss = loss_fn(preds, y)
                val_loss += loss.item() * x.size(0)
                n_val += x.size(0)
        val_loss /= n_val

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    # Load best model
    model.load_state_dict(best_model_state)
    return model
