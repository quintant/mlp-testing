#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --output=SD.txt
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8
#SBATCH -p PGR-Standard
#SBATCH --mail-type=END
#SBATCH --mail-user=s2595230@ed.ac.uk

source /home/s2595230/mlp-testing/.venv/bin/activate

python3 custom_train.py --data_dir "./data/train" --batch_size 16 --num_workers 4 --lr 1e-5 --resolution 768 --center_crop --random_flip --compile --dataparallel