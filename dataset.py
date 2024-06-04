import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
from pathlib import Path


class ArtificialImagesDataset(Dataset):
    def __init__(self, image_dir:Path, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = list(image_dir.glob("*.png"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = read_image(str(self.images[idx]))
        if self.transform:
            image = self.transform(image)
        return image