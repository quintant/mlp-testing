from subprocess import Popen, PIPE, STDOUT
from pathlib import Path
import argparse


def main(args):
    generation = 0

    for generation in range(args.num_generations):
        proc = Popen(
            [
                "accelerate",
                "launch",
                "--multi_gpu",
                '--mixed_precision=fp16',
                "generate_training_data.py",
                "--run_id", f"{args.run_id}",
                "--resolution", f"{args.resolution}",
                "--num_images", f"{args.num_images}",
                "--generation", f"{generation}",
                "--images_per_generation", f"{args.images_per_generation}"
            ],
            stdout=PIPE,
            stderr=STDOUT,
            text=True,
        )

        for c in iter(lambda: proc.stdout.read(1), ""):
            print(c, end="", flush=True)

        remaining_output = proc.communicate()
        print(remaining_output[0].strip())

        meta_data = []
        # Get all partial metadata files
        path = Path(f"runs/{args.run_id}/data/{generation}")
        meta_files = list(path.glob("metadata_*.jsonl"))
        for meta_file in meta_files:
            with open(meta_file, "r") as f:
                meta_data.extend(f.readlines())

        with open(path / "metadata.jsonl", "w") as f:
            for x in meta_data:
                f.write(x)

        # Remove all partial metadata files
        for meta_file in meta_files:
            meta_file.unlink()

        proc = Popen(
            [
                "python3",
                "train.py",
                "--run_id", f"{args.run_id}",
                "--batch_size", f"{args.batch_size}",
                "--num_workers", f"{args.num_workers}",
                "--lr", f"{args.lr}",
                "--beta1", f"{args.beta1}",
                "--beta2", f"{args.beta2}",
                "--weight_decay", f"{args.weight_decay}",
                "--adam_epsilon", f"{args.adam_epsilon}",
                "--epochs", f"{args.epochs}",
                "--resolution", f"{args.resolution}",
                "--center_crop", 
                "--random_flip", 
                "--no_split", 
                "--generation", f"{generation}",
                "--compile" if args.compile else "",
                "--dataparallel" if args.dataparallel else "",
            ],
            stdout=PIPE,
            stderr=STDOUT,
            text=True,
        )

        for c in iter(lambda: proc.stdout.read(1), ""):
            print(c, end="", flush=True)

        remaining_output = proc.communicate()
        print(remaining_output[0].strip())



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default="default")
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
    parser.add_argument("--num_generations", type=int, required=True)
    parser.add_argument("--num_images", type=int, default=10_000)
    parser.add_argument("--images_per_generation", type=int, default=16)

    args = parser.parse_args()

    main(args)
