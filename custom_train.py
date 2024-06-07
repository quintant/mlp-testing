import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import (
    check_min_version,
    deprecate,
    is_wandb_available,
    make_image_grid,
)
from diffusers.utils.import_utils import is_xformers_available

from dataset import ArtificialImagesDataset


from typing import List, Tuple
from contextlib import nullcontext

MODEL_NAME = "stabilityai/stable-diffusion-2-1"



torch.backends.cuda.matmul.allow_tf32 = True


def load_models(
    model_name: str,
    vae_device: str = None,
    unet_device: str = None,
    text_encoder_device: str = None,
    compile: bool = False,
) -> Tuple[
    AutoencoderKL, UNet2DConditionModel, CLIPTextModel, DDPMScheduler, CLIPTokenizer
]:
    vae = AutoencoderKL.from_pretrained(
        model_name, torch_dtype=torch.float16, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        model_name, torch_dtype=torch.float16, subfolder="unet"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_name, torch_dtype=torch.float16, subfolder="text_encoder"
    )
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")

    if vae_device is not None:
        vae.to(vae_device)
    if unet_device is not None:
        unet.to(unet_device)
    if text_encoder_device is not None:
        text_encoder.to(text_encoder_device)

    if compile:
        vae = torch.compile(vae)
        unet = torch.compile(unet)
        text_encoder = torch.compile(text_encoder)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.eval()
    text_encoder.eval()

    return vae, unet, text_encoder, noise_scheduler, tokenizer


def create_parallel_models(
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        text_encoder: CLIPTextModel,
        compile: bool = False,

) -> Tuple[
    AutoencoderKL, UNet2DConditionModel, CLIPTextModel
]:
    # Get number of devices
    num_devices = torch.cuda.device_count()
    print(f"Number of devices: {num_devices}")
    print(f"Will use approximately {num_devices//3} devices for each model")

    if num_devices < 3:
        print("Not enough devices to create parallel models")
        print("Falling back to single device")
        return vae, unet, text_encoder
    
    curr_device = 0
    for model in [vae, text_encoder]:
        if model is not None:
            model.to(f"cuda:{curr_device}")
            if compile:
                model = torch.compile(model)
            model = torch.nn.DataParallel(model, device_ids=list(range(curr_device, curr_device + num_devices // 3)))
            curr_device += num_devices // 3
    if unet is not None:
        unet.to(f"cuda:{curr_device}")
        if compile:
            unet = torch.compile(unet)
        unet = torch.nn.DataParallel(unet, device_ids=list(range(num_devices))[curr_device:])

    return vae, unet, text_encoder

def get_dataset(args: argparse.Namespace) -> datasets.Dataset:
    data_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    dataset = ArtificialImagesDataset(
        data_dir=args.data_dir,
        transform=data_transforms,
        model_name=MODEL_NAME,
    )
    return dataset


def main(args: argparse.Namespace):

    vae, unet, text_encoder, scheduler, _ = load_models(MODEL_NAME)
    
    if args.dataparallel:
        vae, unet, text_encoder = create_parallel_models(vae, unet, text_encoder, compile=args.compile)

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        eps=args.adam_epsilon,
    )

    dataset = get_dataset(args)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Gradient scaler
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.epochs):
        pb = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for i, (image, tokens) in enumerate(pb):
            with torch.cuda.amp.autocast():
                image = image.to(vae.device)
                tokens = tokens.to(text_encoder.device)

                latents = vae.encode(image).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents).to(vae.device)

                bsz = image.shape[0]

                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,)).to(latents.device)

                timesteps = timesteps.long()

                encoder_hidden_states = text_encoder(tokens)[0]

                target = scheduler.get_velocity(latents, noise, timesteps)

                noisy_latents = scheduler.add_noise(latents, noise, timesteps)

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                loss = F.mse_loss(model_pred.float(), target.float())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()



    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
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
    args = parser.parse_args()
    main(args)
