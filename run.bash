#!/bin/bash
#SBATCH --nodelist=landonia06
#SBATCH --gres=gpu:a6000:2
#SBATCH --output=SD.txt
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=12
#SBATCH -p PGR-Standard
#SBATCH --mail-type=END
#SBATCH --mail-user=s2595230@ed.ac.uk

source /home/s2595230/mlp-testing/.venv/bin/activate

while true; do
  nvidia-smi >> gpu_usage.log
  top -b -n 1 | head -n 20 >> cpu_usage.log
  sleep 10
done &

python3 custom_train.py --run_id "hands_only" --batch_size 32 --num_workers 12 --lr 1e-5 --resolution 768 --center_crop --random_flip --dataparallel --no_split --num_generations 100

kill %1s