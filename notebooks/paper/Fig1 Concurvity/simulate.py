import torch
import lightning



### Global simulation variables.
p, q = 2, 256
x0 = 1/p * torch.ones(p)
u0 = 1/q * torch.ones(q) * 0
beta = 1 * torch.ones(p+1)
delta = torch.zeros(p+1, q)
delta[:, q//2:] = (p+1)/q * torch.ones(p+1, q//2)
gamma = 1 * torch.ones(q)
beta = - 0.5 * delta @ gamma

# Save direct and total effects.
beta_dir = beta
beta_tot = beta + delta @ gamma

def simulate(N=30000, binary=False, balanced=False):
    global p, q, x0, u0, beta, delta, gamma
    if balanced:
        beta = - delta @ gamma
    else:
        beta = - 0.5 * delta @ gamma
    X = x0 + torch.randn(N, p)
    X = torch.cat((torch.ones(N, 1), X), 1)
    U = u0 + X @ delta + torch.randn(N, q) * (p+1)**2/(q//2)
    eta = X @ beta + U @ gamma
    if binary == False:
        y = eta + torch.randn(N)
    else:
        proba = torch.sigmoid(eta)
        y = torch.bernoulli(proba)
    return U, X, y

### Models

def estimate_models(train_loader, val_loader, params):
    # Estimate Baseline Neural Network
    net = NeuralNetwork(**params)
    model_checkpoint = lightning.pytorch.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
    early_stop_callback = lightning.pytorch.callbacks.EarlyStopping(monitor="val_loss", patience=16, mode="min")
    trainer = lightning.Trainer(max_epochs=250, enable_progress_bar=False, callbacks=[model_checkpoint, early_stop_callback])
    trainer.fit(net, train_loader, val_loader)
    net = NeuralNetwork.load_from_checkpoint(model_checkpoint.best_model_path)

    # Estimate Baseline Neural Network
    net_covar = CovarNeuralNetwork(**params)
    model_checkpoint = lightning.pytorch.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
    early_stop_callback = lightning.pytorch.callbacks.EarlyStopping(monitor="val_loss", patience=16, mode="min")
    trainer = lightning.Trainer(max_epochs=250, enable_progress_bar=False, callbacks=[model_checkpoint, early_stop_callback])
    trainer.fit(net_covar, train_loader, val_loader)
    net_covar = CovarNeuralNetwork.load_from_checkpoint(model_checkpoint.best_model_path)

    # Estimate Post-Hoc models.
    #net_ls = PostHocIRLSModel(net, train_loader, num_covars=params['num_covars'], pen_factor=0.00, orthogonalize=False)
    net_ls = PostHocLSModel(net, train_loader, num_covars=params['num_covars'], pen_factor=0.00, orthogonalize=False)
    net_orth_ls = PostHocIRLSModel(net, train_loader, num_covars=params['num_covars'], pen_factor=0.00, orthogonalize=True)
    #net_ls = copy.deepcopy(net_orth_ls)
    #net_ls.orthogonalize = False
    #net_ls.model.struct_predictor.weight.data -= net_ls.model.deep_predictor.weight.data @ net_ls.model.ortho_parameters.T
    net_web = PHOWeber2024(net, num_covars=params['num_covars'], train_dataloader=train_loader)
    net_rug = PHORuegamer2023(net_covar, train_loader)

    return {
    #    "NN": net,
        "Covar NN": net_covar,
        "DE Model": net_ls,
        "TE Model": net_orth_ls,
        "PHO (Web)": net_web,
        "PHO (Rug)": net_rug,
    }

### Metrics

def get_direct_effects(X, U):
    global p, q, x0, u0, beta, delta, gamma
    eta_struct = X @ beta
    eta_deep = U @ gamma
    return {"struct": eta_struct, "deep": eta_deep}

def get_total_effects(X, U):
    global p, q, x0, u0, beta, delta, gamma
    eta_struct = X @ (beta + delta @ gamma) + u0 @ gamma
    eta_deep = (U - u0 - X @ delta) @ gamma
    return {"struct": eta_struct, "deep": eta_deep}


def struct_residuals(model, test_loader):
    model.eval()
    r_list = []
    for batch in test_loader:
        U, X, y = batch["image"], batch["covar"], batch["label"]
        with torch.no_grad():
            r = get_direct_effects(X, U)["struct"]
            r_list.append(model.predict_struct(batch).squeeze() - r)
    return torch.cat(r_list)

def struct_total_residuals(model, test_loader):
    model.eval()
    r_list = []
    for batch in test_loader:
        U, X, y = batch["image"], batch["covar"], batch["label"]
        with torch.no_grad():
            r_orth = get_total_effects(X, U)["struct"]
            r_list.append(model.predict_struct(batch).squeeze() - r_orth)
    return torch.cat(r_list)

def deep_residuals(model, test_loader):
    model.eval()
    r_list = []
    for batch in test_loader:
        U, X, y = batch["image"], batch["covar"], batch["label"]
        with torch.no_grad():
            r = get_direct_effects(X, U)["deep"]
            r_list.append(model.predict_deep(batch).squeeze() - r)
    return torch.cat(r_list)

def deep_orthogonal_residuals(model, test_loader):
    model.eval()
    r_list = []
    for batch in test_loader:
        U, X, y = batch["image"], batch["covar"], batch["label"]
        with torch.no_grad():
            r_orth = get_total_effects(X, U)["deep"]
            r_list.append(model.predict_deep(batch).squeeze() - r_orth)
    return torch.cat(r_list)

def residuals(model, test_loader):
    model.eval()
    r_list = []
    for batch in test_loader:
        U, X, y = batch["image"], batch["covar"], batch["label"]
        with torch.no_grad():
            if model.output_func.__class__.__name__ == "Identity":
                r = model.predict_step(batch, None).squeeze() - y
            else:  # Pearson residuals
                p = model.predict_step(batch, None).squeeze()
                r = (y - p) / torch.sqrt(p * (1 - p))
            r_list.append(r)
    return torch.cat(r_list)

def mse_decomposition(rhat):
    return {"mse": torch.mean(rhat**2).numpy(), "bias": torch.mean(rhat).numpy(), "variance": torch.var(rhat).numpy()}


def get_residuals(model, test_loader):
    r = residuals(model, test_loader)
    r_struct = struct_residuals(model, test_loader)
    r_struct_total = struct_total_residuals(model, test_loader)
    r_deep = deep_residuals(model, test_loader)
    r_deep_orth = deep_orthogonal_residuals(model, test_loader)
    return {
        "Residuals": r,
        "Structural Residuals": r_struct,
        "Deep Residuals": r_deep,
        "Total Structural Residuals": r_struct_total,
        "Orthogonalized Deep Residuals": r_deep_orth
    }

def get_bias(model):
    bias = get_structural_bias(model)
    b_dir = bias["direct"]
    b_tot = bias["total"]
    return {"direct": torch.linalg.vector_norm(b_dir).numpy(), "total": torch.linalg.vector_norm(b_tot).numpy()}


def get_structural_bias(model):
    global p, q, x0, u0, beta, delta, gamma
    beta_hat = torch.tensor(model.struct_coefs())
    return {"direct": beta - beta_hat, "total": (beta + delta @ gamma) - beta_hat}
