import torch


class CovarDataset(torch.utils.data.Dataset):
    def __init__(self, image, covar, label, transform=None):
        self.image = image
        self.covar = covar
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        image = self.image[idx]
        covar = self.covar[idx]
        label = self.label[idx]
        if self.transform:
            image = self.transform(image)
        return {"image": image, "covar": covar, "label": label}
