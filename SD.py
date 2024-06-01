import PIL.Image
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import PIL
import os
import uuid

os.makedirs("output", exist_ok=True)
MODEL_ID = "stabilityai/stable-diffusion-2-1"
NUM_IMG = 1_000
NUM_IMG_PER_PROMPT = 2

PROMPT = "Portrait of a person, photo, high quality, high resolution, vivid, sharp, clear, detailed, realistic"
NEG_PROMPT = "low quality, blurry, noisy, bad quality, worse quality, poor quality, low resolution, low res, pixelated, camera, black and white, old, vintage, painting, drawing"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")


# Generate images
for i in range(NUM_IMG//NUM_IMG_PER_PROMPT):

    images:tuple[PIL.Image.Image] = pipe(
        prompt=PROMPT, 
        neg_prompt=NEG_PROMPT, 
        return_dict=False, 
        num_images_per_prompt=NUM_IMG_PER_PROMPT
    )[0]

    print(f"Generated {len(images)} images")
    
    for img in images:
        file_id = uuid.uuid4()
        img.save(f"output/{file_id}.png")
        