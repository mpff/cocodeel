
from cocodeel.dataset import CovarDataset


### Dataloaders

def create_dataloaders(U, X, y, batch_size=256):
    N = len(y)
    # Create training, validation and test data.
    train_data = CovarDataset(U[:N // 3], X[:N // 3], y[:N // 3])
    val_data = CovarDataset(U[N // 3:2 * N // 3], X[N // 3:2 * N // 3], y[N // 3:2 * N // 3])
    test_data = CovarDataset(U[2 * N // 3:], X[2 * N // 3:], y[2 * N // 3:])
    # Create dataloaders.
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
