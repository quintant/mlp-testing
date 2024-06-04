#!/bin/bash
#SBATCH --nodelist=crannog02
#SBATCH --gres=gpu:4
#SBATCH --output=SD.txt
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8
#SBATCH -p PGR-Standard
#SBATCH --mail-type=END
#SBATCH --mail-user=s2595230@ed.ac.uk

source /home/s2595230/mlp-testing/.venv/bin/activate

export MODEL_NAME="stabilityai/stable-diffusion-2-1"
export dataset_name="./data/"

accelerate launch --multi_gpu --mixed_precision="fp16"  train_text_to_image.py --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$dataset_name \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=8 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --enable_xformers_memory_efficient_attention \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="fine-tuned-test-model"