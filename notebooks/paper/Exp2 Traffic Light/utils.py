import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


import matplotlib.pyplot as plt

from cocodeel.model import BaseNetwork



# ----------------------------
# Simulated dataset
# ----------------------------

def simulate_traffic_light_data(n=800, h=20, w=60, circle_radius=8,
    corr_strength_v1=0.9, corr_strength_v2=0.25, seed=0):

    torch.manual_seed(seed)

    # 1. Covariate Z
    Z = torch.randn(n, 1)
    Z = (Z - Z.min()) / (Z.max() - Z.min()) # normalize to [0,1]

    # 2. Latent variables
    v1_raw = torch.rand(n, 1)
    v2_raw = torch.rand(n, 1)
    v3 = torch.rand(n, 1) # independent

    # Correlate v1 and v2 with Z
    v1 = (1 - corr_strength_v1) * v1_raw + corr_strength_v1 * Z
    v2 = (1 - corr_strength_v2) * v2_raw + corr_strength_v2 * Z

    # Normalize to [0,1]
    v1 = (v1 - v1.min()) / (v1.max() - v1.min())
    v2 = (v2 - v2.min()) / (v2.max() - v2.min())
    v3 = (v3 - v3.min()) / (v3.max() - v3.min())

    # 3. Build traffic light images with white noise background
    X = torch.rand((n, 1, h, w)) * 0.2 # white noise background, low intensity

    # Circle centers (left, center, right)
    centers = [(h//2, w//6), (h//2, w//2), (h//2, 5*w//6)]

    # Create circle mask function
    def circle_mask(h, w, center, radius):
        Y, Xg = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        dist = (Xg - center[1])**2 + (Y - center[0])**2
        return dist <= radius**2

    mask1 = circle_mask(h, w, centers[0], circle_radius)
    mask2 = circle_mask(h, w, centers[1], circle_radius)
    mask3 = circle_mask(h, w, centers[2], circle_radius)

    # Paint circles per sample in grayscale according to latent values
    for i in range(n):
        X[i, 0][mask1] = v1[i]
        X[i, 0][mask2] = v2[i]
        X[i, 0][mask3] = v3[i]

    # 4. Outcome y depends on v2, v3, and Z with balanced magnitudes
    y = 1.0*v2 + 1.0*v3 + 1.0*Z + 0.1*torch.randn(n, 1)

    # 5. Also save effects:
    fx = 1.0*v2 + 1.0*v3 - torch.mean(1.0*v2 + 1.0*v3)
    fz = 1.0*Z - torch.mean(1.0*Z)
    fr = 1.0*v3 - torch.mean(1.0*v3)

    return X, Z, y, v1, v2, v3, fx, fz, fr


def show_samples(X, Z, v1, v2, v3, y, n_show=5):
    fig, axes = plt.subplots(1, n_show, figsize=(3*n_show, 3))
    for i in range(n_show):
        ax = axes[i]
        ax.imshow(X[i,0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(
            f"X: v1={v1[i].item():.2f}, v2={v2[i].item():.2f}, v3={v3[i].item():.2f}\n"
            f"Z={Z[i].item():.2f}, y={y[i].item():.2f}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()



# ----------------------------
# Backbones
# ----------------------------

class TrafficBackbone(nn.Module):
    def __init__(self, out_features):
        super(TrafficBackbone, self).__init__()
        self.out_features = out_features
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # fixed output size 32 * 4 * 4
        self.fc = nn.Linear(32 * 4 * 4, self.out_features)

    def forward(self, X):
        # X: (batch, 1, h, w)
        x = F.relu(self.conv1(X))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # flatten to (batch, 32 * 4 * 4)
        x = self.fc(x)
        return x



# ----------------------------
# Training routine
# ----------------------------

def train_base_model(backbone, backbone_params, train_loader, val_loader,
                     epochs=1000, lr=1e-3, weight_decay=1e-4, patience=12):
    model = BaseNetwork(backbone=backbone, backbone_params=backbone_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience-2, factor=0.5)

    best_val_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())
    counter = 0

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x, y = batch["X"], batch["y"]
            optimizer.zero_grad()
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
                x, y = batch["X"], batch["y"]
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
