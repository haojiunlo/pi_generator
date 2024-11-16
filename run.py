import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_utils.dataset import PiPointDataset
from models.vae import PiPointVAE
from visualize.utils import visualize_results_pil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--d_model",
        type=int,
        help="the number of expected features in the transformer input",
        default=256,
    )
    parser.add_argument(
        "--nhead",
        type=int,
        help="the number of heads in the multiheadattention models",
        default=8,
    )
    parser.add_argument(
        "--num_layers",
        default=3,
        type=int,
        help="the number of sub-encoder/decoder-layers in the transformers",
    )
    parser.add_argument(
        "--latent_dim",
        default=32,
        type=int,
        help="latent dim of VAE",
    )
    parser.add_argument(
        "--num_epochs",
        default=100,
        type=int,
        help="num of epochs",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type=float,
        help="learning rate",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="batch size",
    )
    parser.add_argument(
        "--device",
        default="mps",
        type=str,
        help="device",
        choices=["cpu", "cuda", "mps"],
    )
    parser.add_argument(
        "--saved_dir",
        default="saved_model",
        type=str,
        help="where to save the trained model",
    )
    parser.add_argument(
        "--dataset_dir",
        default="data",
        type=str,
        help="training data location",
    )
    parser.add_argument(
        "--beta",
        default=0.01,
        type=float,
        help="beta for beta-VAE",
    )
    args = parser.parse_args()
    return args


def train_model(
    model, train_loader, beta=0.01, num_epochs=100, learning_rate=1e-4, device="mps"
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Main epoch progress bar
    epoch_pbar = tqdm(range(num_epochs), desc="Training", position=0)

    # Initialize metrics for tracking
    metrics = {"total_loss": 0.0, "recon_loss": 0.0, "kl_loss": 0.0}

    for epoch in epoch_pbar:
        # Reset metrics
        for k in metrics:
            metrics[k] = 0.0

        # Batch progress bar
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, position=1)

        for batch in batch_pbar:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass
            recon_batch, mu, logvar = model(batch)

            # Loss computation
            recon_loss = (
                F.cross_entropy(recon_batch[0], batch[:, 0].long())
                + F.cross_entropy(recon_batch[1], batch[:, 1].long())
                + F.cross_entropy(recon_batch[2], batch[:, 2].long())
                + F.cross_entropy(recon_batch[3], batch[:, 3].long())
                + F.cross_entropy(recon_batch[4], batch[:, 4].long())
            )

            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            loss = recon_loss + beta * kl_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update metrics
            metrics["total_loss"] += loss.item()
            metrics["recon_loss"] += recon_loss.item()
            metrics["kl_loss"] += kl_loss.item()

            # Update batch progress bar
            batch_pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "recon": f"{recon_loss.item():.4f}",
                    "kl": f"{kl_loss.item():.4f}",
                }
            )

        # Average metrics over batches
        for k in metrics:
            metrics[k] /= len(train_loader)

        # Update epoch progress bar
        epoch_pbar.set_postfix(
            {
                "avg_loss": f"{metrics['total_loss']:.4f}",
                "avg_recon": f"{metrics['recon_loss']:.4f}",
                "avg_kl": f"{metrics['kl_loss']:.4f}",
            }
        )


def main(args):
    # Load data
    data_dir = Path(args.dataset_dir)
    xs = np.load(data_dir / "pi_xs.npy")
    ys = np.load(data_dir / "pi_ys.npy")
    image = np.array(Image.open(data_dir / "sparse_pi_colored.jpg"))

    # Create dataset and dataloader
    dataset = PiPointDataset(xs, ys, image)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize and train model
    model = PiPointVAE(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        latent_dim=args.latent_dim,
    )
    train_model(
        model,
        dataloader,
        beta=args.beta,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=args.device,
    )

    # save model
    output_folder = Path("outputs")
    output_folder.mkdir(parents=True, exist_ok=True)
    saved_dir = output_folder / args.saved_dir
    model.save(
        saved_dir,
        beta=args.beta,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
    )
    print(f"model save at {saved_dir}")

    # Generate new points
    generated_points = model.generate(5000)
    generated_points = generated_points.cpu().numpy()
    np.save(saved_dir / "generated_points.npy", generated_points)
    print(f"generated_points save at {saved_dir}")

    # Create visualization
    img = visualize_results_pil(
        dataset.points.numpy().astype(int),
        generated_points,
        image_size=(300, 300),
        point_size=1,
    )
    img.save(saved_dir / "visualization.png")
    print(f"comparison image save at {saved_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
