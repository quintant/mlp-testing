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




from typing import List, Tuple

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


def main(args: argparse.Namespace):

    vae, unet, text_encoder, scheduler, toenizer = load_models(MODEL_NAME)
    
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

    


