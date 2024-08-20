import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from transformers import CLIPTokenizer
from PIL import Image

class ArtificialImagesDataset(Dataset):
    def __init__(self, data_dir:Path, transform=None, model_name:str="stabilityai/stable-diffusion-2-1"):
        self.data_dir = data_dir
        self.transform = transform
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        self.meta_data = dict()
        meta_data_file = data_dir/"metadata.jsonl"
        with open(meta_data_file, 'r') as f:
            for line in f:
                data = eval(line)
                self.meta_data[data["file_name"]] = data["text"]
        self.images = list(self.meta_data.keys())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.data_dir/self.images[idx]
        image = Image.open(img_path)
        text = self.meta_data[self.images[idx]]
        if self.transform:
            image = self.transform(image)
            
        tokens = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
            ).input_ids
        
        return image, tokens
    
class RealImagesDataset(Dataset):
    def __init__(self, data_dir:Path, transform=None, num_real_images:int=16):
        self.data_dir = data_dir
        self.transform = transform
        self.images = list(self.data_dir.glob("*.jpg"))
        self.num_real_images = num_real_images
        if len(self.images) < num_real_images:
            self.num_real_images = len(self.images)

    def __len__(self):
        return self.num_real_images

    def __getitem__(self, idx):
        rand_idx = torch.randint(0, len(self.images), (1,)).item()
        img_path = self.images[rand_idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, 0
    


if __name__ == "__main__":
    path = Path("data/train")
    images = path.glob("*.png")
    prompt = "Portrait of a person, photo, high quality, high resolution, vivid, sharp, clear, detailed, realistic"

    meta_data_file = "metadata.jsonl"

    meta_data = []
    for image in images:
        meta_data.append('{' + f'"file_name": "{image.name}", "text": "{prompt}"' + '}\n')

    with open(path/meta_data_file, 'w') as f:
        for x in meta_data:
            f.write(x)
