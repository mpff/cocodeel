from torch.utils.data import Dataset

class CovarDataset(Dataset):
    """Dataset with covariates."""

    def __init__(self, X, Z, y, transform=None):
        """
        Args:
            X (torch.tensor): Array of input images.
            Z (torch.tensor): Array of covariates.
            y (torch.tensor): Array of outputs.
            transform (torchvision.transforms): Optional transform applied to X.
        """
        self.X = X
        self.Z = Z
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = self.X[idx]
        Z = self.Z[idx]
        y = self.y[idx]
        if self.transform:
            X = self.transform(X)
        return {"X": X, "Z": Z, "y": y}
