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
    


if __name__ == "__main__":
    path = Path("data/train")
    images = path.glob("*.png")
    prompt = "Portrait of a person, photo, high quality, high resolution, vivid, sharp, clear, detailed, realistic"

    meta_data_file = "metadata.jsonl"

    meta_data = []
    for image in images:
        meta_data.append('{' + f'"file_name": "{image}", "text": "{prompt}"' + '}\n')

    with open(path/meta_data_file, 'w') as f:
        for x in meta_data:
            f.write(x)
