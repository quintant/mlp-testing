#!/bin/bash
#SBATCH --gres=gpu:a6000:2
#SBATCH --output=SD.txt
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8
#SBATCH -p Teach-Standard
#SBATCH --mail-type=END
#SBATCH --mail-user=s2595230@ed.ac.uk

source /home/s2595230/mlp-testing/.venv/bin/activate
rm /home/s2595230/.cache/huggingface/accelerate/default_config.yaml
accelerate config default

while true; do
  nvidia-smi >> gpu_usage.log
  top -b -n 1 | head -n 20 >> cpu_usage.log
  sleep 10
done &

python3 run.py --run_id "hands_only" --batch_size 32 --num_workers 8 --lr 1e-5 --resolution 768 --center_crop --random_flip --dataparallel --no_split --num_generations 100 --num_images 64 --images_per_generation 16

kill %1s
