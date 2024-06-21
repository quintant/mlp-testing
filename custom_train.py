import argparse
from copy import deepcopy
import logging
import math
import os
import random
import shutil
from pathlib import Path
import uuid

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



# torch.backends.cuda.matmul.allow_tf32 = True


def load_models(
    model_name: str,
    vae_device: str = None,
    unet_device: str = None,
    text_encoder_device: str = None,
    compile: bool = False,
) -> Tuple[
    AutoencoderKL, UNet2DConditionModel, CLIPTextModel, DDPMScheduler, CLIPTokenizer
]:
    print("Loading models")
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

    print("Models loaded")

    if vae_device is not None:
        vae.to(vae_device)
    if unet_device is not None:
        unet.to(unet_device)
    if text_encoder_device is not None:
        text_encoder.to(text_encoder_device)

    if compile:
        print("Compiling models")
        vae = torch.compile(vae)
        unet = torch.compile(unet)
        text_encoder = torch.compile(text_encoder)
        print("Models compiled")

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
        no_split: bool = True,

) -> Tuple[
    AutoencoderKL, UNet2DConditionModel, CLIPTextModel, int, int, int
]:
    """
    Create parallel models for training
    returns: Tuple[AutoencoderKL, UNet2DConditionModel, CLIPTextModel, vae_device, text_encoder_device, unet_device]
    """
    if no_split:
        print("Creating parallel models without splitting")
        vae_device = 0
        text_encoder_device = 0
        unet_device = 0
        vae.to(f"cuda:{vae_device}")
        text_encoder.to(f"cuda:{text_encoder_device}")
        unet.to(f"cuda:{unet_device}")
        if compile:
            vae = torch.compile(vae)
            text_encoder = torch.compile(text_encoder)
            unet = torch.compile(unet)
        vae = torch.nn.DataParallel(vae)
        text_encoder = torch.nn.DataParallel(text_encoder)
        unet = torch.nn.DataParallel(unet)
        return vae, unet, text_encoder, vae_device, text_encoder_device, unet_device
    # Get number of devices
    num_devices = torch.cuda.device_count()
    print(f"Number of devices: {num_devices}")
    print(f"Will use approximately {num_devices//3} devices for each model")

    if num_devices < 3:
        print("Not enough devices to create parallel models")
        print("Falling back to single device")
        return vae, unet, text_encoder
    
    curr_device = 0
    main_devices = []
    for model in [vae, text_encoder]:
        print(f"Creating parallel model for {model.__class__.__name__}")
        print(f"Current device: {curr_device}")
        print(f"Will use devices: {list(range(curr_device, curr_device + 1))}")
        if model is not None:
            model.to(f"cuda:{curr_device}")
            main_devices.append(curr_device)
            if compile:
                model = torch.compile(model)
            model = torch.nn.DataParallel(model, device_ids=list(range(curr_device, curr_device + 1)))
            curr_device += 1
    if unet is not None:
        print(f"Creating parallel model for {unet.__class__.__name__}")
        print(f"Current device: {curr_device}")
        print(f"Will use devices: {list(range(num_devices)[curr_device:])}")
        unet.to(f"cuda:{curr_device}")
        main_devices.append(curr_device)
        if compile:
            unet = torch.compile(unet)
        unet = torch.nn.DataParallel(unet, device_ids=list(range(num_devices))[curr_device:])

    return vae, unet, text_encoder, *main_devices

def get_dataset(data_dir: str, args: argparse.Namespace) -> datasets.Dataset:
    print("Getting dataset")
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
        data_dir=Path(data_dir),
        transform=data_transforms,
        model_name=MODEL_NAME,
    )
    return dataset

def save_model(unet: UNet2DConditionModel, run_id: str, generation: int):
    print("Saving model")
    unet.save_pretrained(f"runs/{run_id}/models/unet_{generation}")

def generate_training_data(
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        scheduler: DDPMScheduler,
        tokenizer: CLIPTokenizer,
        run_id: str,
        num_images: int,
        generation: int,
        device: str,
        resolution: int,
        prompt: str,
        no_images_per_generation: int = 16,
    ):
    unet.eval()
    with torch.no_grad():
        print("Generating training data")
        pipe = StableDiffusionPipeline(
            vae=vae,
            unet=unet,
            text_encoder=text_encoder,
            scheduler=scheduler,
            tokenizer=tokenizer,
            requires_safety_checker=False,
            safety_checker=None,
            feature_extractor=None,
        )
        pipe = pipe.to(device)

        os.makedirs(f"runs/{run_id}/data/{generation}", exist_ok=True)
        metadata = []
        for i in range(num_images // no_images_per_generation):
            images = pipe(
                prompt=prompt,
                return_dict=False,
                num_images_per_prompt=no_images_per_generation,
                resolution=resolution,
            )[0]
            for img in images:
                file_id = uuid.uuid4()
                img.save(f"runs/{run_id}/data/{generation}/{file_id}.png")
                metadata.append(
                    f"{{'file_name': '{file_id}.png', 'text': '{prompt}'}}"
                )

    with open(f"runs/{run_id}/data/{generation}/metadata.jsonl", "w") as f:
        for x in metadata:
            f.write(x + "\n")

    unet.train()


def train(dataloader, vae, unet, text_encoder, scheduler, optimizer, args, vae_device, text_encoder_device, unet_device):
    # Gradient scaler
    print("Starting training loop")
    for epoch in range(args.epochs):
        pb = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for i, (image, tokens) in enumerate(pb):
            with torch.cuda.amp.autocast():
                print("Sending data to devices")
                image = image.to(vae_device)
                tokens = tokens.to(text_encoder_device)

                print("Encoding image")
                latents = vae.encode(image).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                print("Generating noise")
                noise = torch.randn_like(latents).to(vae_device)

                bsz = image.shape[0]

                print("Getting timesteps")
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,)).to(vae_device)

                timesteps = timesteps.long()

                print("Encoding text")
                encoder_hidden_states = text_encoder(tokens)[0]

                print("Getting target")
                target = scheduler.get_velocity(latents, noise, timesteps)

                print("Adding noise")
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)

                print("Sending data to unet device")
                noisy_latents = noisy_latents.to(unet_device)
                encoder_hidden_states = encoder_hidden_states.to(unet_device)
                timesteps = timesteps.to(unet_device)
                target = target.to(unet_device)

                print("Getting model prediction")
                model_pred = unet(
                    noisy_latents, 
                    timesteps, 
                    encoder_hidden_states,
                    return_dict=False
                )[0]

                print("Calculating loss")
                loss = F.mse_loss(model_pred.float(), target.float())

            print("Backpropagating")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Updating progress bar")


def main(args: argparse.Namespace):
    print("Starting training")
    generation = 0
    run_id = args.run_id
    print("Creating run directory")
    os.makedirs(f"runs/{run_id}", exist_ok=True)
    total_generations = args.num_generations
    num_images = args.num_images


    vae_o, unet_o, text_encoder_o, scheduler, tokenizer = load_models(MODEL_NAME)
    
    if args.dataparallel:
        print("Creating parallel models")
        vae, unet, text_encoder, vae_device, text_encoder_device, unet_device = create_parallel_models(deepcopy(vae_o), deepcopy(unet_o), deepcopy(text_encoder_o), compile=args.compile, no_split=args.no_split)

        vae_device = f"cuda:{vae_device}"
        text_encoder_device = f"cuda:{text_encoder_device}"
        unet_device = f"cuda:{unet_device}"
    else:
        vae_device = "cuda" if torch.cuda.is_available() else "cpu"
        text_encoder_device = "cuda" if torch.cuda.is_available() else "cpu"
        unet_device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Creating optimizer")
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        eps=args.adam_epsilon,
    )

    for generation in range(total_generations):
        generate_training_data(
            unet_o,
            vae_o,
            text_encoder_o,
            scheduler,
            tokenizer,
            run_id,
            num_images=num_images,
            generation=generation,
            device=unet_device,
            resolution=args.resolution,
            prompt="image of hands, photo, high quality, high resolution, vivid, sharp, clear, detailed, realistic",
        )

        dataset = get_dataset(f"runs/{run_id}/data/{generation}", args)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        train(dataloader, vae, unet, text_encoder, scheduler, optimizer, args, vae_device, text_encoder_device, unet_device)

        save_model(unet, run_id, generation)





    
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
    parser.add_argument('--num_generations', type=int, default=10)
    parser.add_argument('--num_images', type=int, default=10_000)

    args = parser.parse_args()
    main(args)

