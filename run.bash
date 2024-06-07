#!/bin/bash
#SBATCH --gres=gpu:8
#SBATCH --output=SD.txt
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=16
#SBATCH -p PGR-Standard
#SBATCH --mail-type=END
#SBATCH --mail-user=s2595230@ed.ac.uk

source /home/s2595230/mlp-testing/.venv/bin/activate

while true; do
  nvidia-smi >> gpu_usage.log
  top -b -n 1 | head -n 20 >> cpu_usage.log
  sleep 10
done &

python3 custom_train.py --data_dir "./data/train" --batch_size 12 --num_workers 16 --lr 1e-5 --resolution 512 --center_crop --random_flip --dataparallel

kill %1s