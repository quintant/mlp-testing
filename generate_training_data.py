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
from train import load_models
from accelerate import PartialState

MODEL_NAME = "stabilityai/stable-diffusion-2-1"


# torch.backends.cuda.matmul.allow_tf32 = True


def generate_training_data(
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    scheduler: DDPMScheduler,
    tokenizer: CLIPTokenizer,
    run_id: str,
    num_images: int,
    generation: int,
    distributed_state: PartialState,
    resolution: int,
    prompt_text: str,
    no_images_per_generation: int = 16,
    save_path: Path = Path("data"),

):
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
        pipe: StableDiffusionPipeline = pipe.to(distributed_state.device)

        metadata = []
        for i in range(num_images // no_images_per_generation // distributed_state.num_processes ):
            with distributed_state.split_between_processes([prompt_text], apply_padding=True) as prompt:
                images = pipe(
                    prompt=prompt,
                    return_dict=False,
                    num_images_per_prompt=no_images_per_generation,
                    resolution=resolution,
                )[0]
                print(f"Generated {len(images)} images")
                print(f"Saving images to {save_path}")
                print(f"Saving metadata to {save_path / 'metadata.jsonl'}")
                for img in images:
                    file_id = uuid.uuid4()
                    img.save(save_path / f"{file_id}.png")
                    metadata.append(f"{{'file_name': '{file_id}.png', 'text': '{prompt}'}}")

    with open(save_path / "metadata.jsonl", "w") as f:
        for x in metadata:
            f.write(x + "\n")


def main(args: argparse.Namespace):
    RUN_PATH = Path(f"runs/{args.run_id}")
    DATA_PATH = RUN_PATH / f"data/{args.generation}"
    MODEL_PATH = RUN_PATH / "models"
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    LOAD_PATH = MODEL_PATH / f"unet_{args.generation}"
    DATA_PATH.mkdir(parents=True, exist_ok=True)

    distributed_state = PartialState()

    vae, unet, text_encoder, scheduler, tokenizer = load_models(
        LOAD_PATH, args.generation
    )

    vae_device = distributed_state
    text_encoder_device = distributed_state
    unet_device = distributed_state

    vae = vae.to(vae_device.device)
    text_encoder = text_encoder.to(text_encoder_device.device)
    unet = unet.to(unet_device.device)

    vae.eval()
    text_encoder.eval()
    unet.eval()

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    generate_training_data(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        scheduler=scheduler,
        tokenizer=tokenizer,
        run_id=args.run_id,
        num_images=args.num_images,
        generation=args.generation,
        distributed_state=distributed_state,
        resolution=args.resolution,
        prompt_text="image of hands, photo, high quality, high resolution, vivid, sharp, clear, detailed, realistic",
        no_images_per_generation=args.images_per_generation,
        save_path=DATA_PATH,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--num_images", type=int, default=10_000)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--images_per_generation", type=int, default=16)

    args = parser.parse_args()
    main(args)
