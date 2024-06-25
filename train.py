import argparse
from pathlib import Path
from typing import Tuple
import torch
from tqdm import tqdm

import argparse
import os
from pathlib import Path
import uuid

import datasets
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)

from dataset import ArtificialImagesDataset
from typing import List, Tuple

MODEL_NAME = "stabilityai/stable-diffusion-2-1"


def train(
    dataloader: torch.utils.data.DataLoader,
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    text_encoder: CLIPTextModel,
    scheduler: DDPMScheduler,
    optimizer: torch.optim.Adam,
    args: argparse.Namespace,
    vae_device: torch.device,
    text_encoder_device: torch.device,
    unet_device: torch.device,
):
    # Gradient scaler
    print("Starting training loop")
    for epoch in range(args.epochs):
        pb = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for i, (image, tokens) in enumerate(pb):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                # print("Sending data to devices")
                image = image.to(vae_device)
                tokens = tokens.to(text_encoder_device)

                # print("Encoding image")
                # latents = vae.encode(image).latent_dist.sample()
                latents = vae(image, "encode")
                # latents = latents * vae.config.scaling_factor
                latents = latents * vae.module.model.config.scaling_factor

                # print("Generating noise")
                noise = torch.randn_like(latents).to(vae_device)

                bsz = image.shape[0]

                # print("Getting timesteps")
                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps, (bsz,)
                ).to(vae_device)

                timesteps = timesteps.long()

                # print("Encoding text")
                encoder_hidden_states = text_encoder(tokens)[0]

                # print("Getting target")
                target = scheduler.get_velocity(latents, noise, timesteps)

                # print("Adding noise")
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)

                # print("Sending data to unet device")
                noisy_latents = noisy_latents.to(unet_device)
                encoder_hidden_states = encoder_hidden_states.to(unet_device)
                timesteps = timesteps.to(unet_device)
                target = target.to(unet_device)

                # print("Getting model prediction")
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states, return_dict=False
                )[0]

                # print("Calculating loss")
                loss = F.mse_loss(model_pred.float(), target.float())

            # print("Backpropagating")
            loss.backward()
            optimizer.step()

            pb.set_postfix({"Loss": loss.item()})


def load_models(
    load_path: Path,
    generation: int,
) -> Tuple[
    AutoencoderKL, UNet2DConditionModel, CLIPTextModel, DDPMScheduler, CLIPTokenizer
]:
    
    vae = AutoencoderKL.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, subfolder="vae"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, subfolder="text_encoder"
    )
    if generation == 0:
        unet = UNet2DConditionModel.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, subfolder="unet"
        )
    else:
        print("Loading from checkpoint")
        lpth = './' + str(load_path)
        print(str(lpth))
        unet = UNet2DConditionModel.from_pretrained(
            str(lpth), torch_dtype=torch.bfloat16
        )
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")

    print("Models loaded")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.eval()
    text_encoder.eval()

    return vae, unet, text_encoder, noise_scheduler, tokenizer


def get_dataset(data_dir: str, args: argparse.Namespace) -> datasets.Dataset:
    print("Getting dataset")
    data_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            (
                transforms.CenterCrop(args.resolution)
                if args.center_crop
                else transforms.RandomCrop(args.resolution)
            ),
            (
                transforms.RandomHorizontalFlip()
                if args.random_flip
                else transforms.Lambda(lambda x: x)
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    dataset = ArtificialImagesDataset(
        data_dir=Path(data_dir),
        transform=data_transforms,
        model_name=MODEL_NAME,
    )
    return dataset

def main(args):
    RUN_PATH = Path(f"runs/{args.run_id}")
    DATA_PATH = RUN_PATH / f"data/{args.generation}"
    MODEL_PATH = RUN_PATH / "models"
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    SAVE_PATH = MODEL_PATH / f"unet_{args.generation + 1}"
    LOAD_PATH = MODEL_PATH / f"unet_{args.generation}"

    vae, unet, text_encoder, scheduler, tokenizer = load_models(LOAD_PATH, args.generation)

    class CustomVAE(torch.nn.Module):
        def __init__(self, model):
            super(CustomVAE, self).__init__()
            self.model = model

        def forward(self, x, function):
            if function == "encode":
                return self.model.encode(x).latent_dist.sample()
            elif function == "decode":
                return self.model.decode(x)
            
    vae = CustomVAE(vae)
            

    vae_device = torch.device("cuda:0")
    text_encoder_device = torch.device("cuda:0")
    unet_device = torch.device("cuda:0")

    vae = vae.to(vae_device)
    text_encoder = text_encoder.to(text_encoder_device)
    unet = unet.to(unet_device)
    unet.train()
    if args.compile:
        unet = unet.compile()
        vae = vae.compile()
        text_encoder = text_encoder.compile()

    if args.dataparallel:
        unet = torch.nn.DataParallel(unet)
        text_encoder = torch.nn.DataParallel(text_encoder)
        vae = torch.nn.DataParallel(vae)

    optimizer = torch.optim.Adam(
        unet.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        eps=args.adam_epsilon,
    )

    dataset = get_dataset(DATA_PATH, args)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    train(
        dataloader,
        vae,
        unet,
        text_encoder,
        scheduler,
        optimizer,
        args,
        vae_device,
        text_encoder_device,
        unet_device,
    )

    if args.dataparallel:
        unet = unet.module

    if args.dataparallel:
        unet = unet.module

    unet.save_pretrained(SAVE_PATH)      
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--dataparallel", action="store_true")
    parser.add_argument("--clip_grad_norm", type=float, default=-1)
    parser.add_argument("--no_split", action="store_true")
    parser.add_argument("--generation", type=int, required=True)

    args = parser.parse_args()
    main(args)
