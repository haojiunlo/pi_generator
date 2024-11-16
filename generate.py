import argparse
from pathlib import Path

import numpy as np

from models.vae import PiPointVAE
from visualize.utils import plot_points


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default="saved_model",
        type=str,
        help="which model to load for generating",
    )
    parser.add_argument(
        "--device",
        default="mps",
        type=str,
        help="device",
        choices=["cpu", "cuda", "mps"],
    )
    parser.add_argument("--num_samples", default=5000, type=int, help="num of samples")
    parser.add_argument(
        "--saved_folder", default="saved_outputs", type=str, help="file name"
    )
    args = parser.parse_args()
    return args


def main(args):
    model = PiPointVAE.load(args.model_dir, device=args.device)

    # Generate new points and visualize
    generated_points = model.generate(args.num_samples)
    gen_image = plot_points(generated_points, point_size=1)
    output_dir = Path("gen_outputs") / args.saved_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    gen_image.save(output_dir / "gen_image.png")
    print(f"Generated image is saved at {output_dir}")

    np.save(output_dir / "gen_points.npy", generated_points.cpu().numpy())
    print(f"Generated points is saved at {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
