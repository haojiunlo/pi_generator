from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from PIL import ImageDraw
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm


# Dataset class to handle the pi points
class PiPointDataset(Dataset):
    def __init__(self, xs, ys, image_array):
        self.points = []
        # Get RGB values for each point
        rgb_values = image_array[xs, ys]
        # Combine coordinates and colors
        self.points = np.column_stack([xs, ys, rgb_values])
        self.points = torch.FloatTensor(self.points)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return self.points[idx]


class PiPointVAE(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=3, latent_dim=32):
        super().__init__()
        self.d_model = d_model

        # Embeddings for discrete values
        self.x_embed = nn.Embedding(300, d_model)  # x: 0-299
        self.y_embed = nn.Embedding(300, d_model)  # y: 0-299
        self.rgb_embed = nn.Embedding(256, d_model)  # rgb: 0-255

        # Separate positional encodings for spatial and color channels
        self.spatial_pos = nn.Parameter(torch.randn(1, 2, d_model))
        self.color_pos = nn.Parameter(torch.randn(1, 3, d_model))

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # VAE components
        self.mu = nn.Linear(d_model * 5, latent_dim)
        self.logvar = nn.Linear(d_model * 5, latent_dim)
        self.latent_proj = nn.Linear(latent_dim, d_model * 5)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output projections
        self.x_out = nn.Linear(d_model, 300)
        self.y_out = nn.Linear(d_model, 300)
        self.rgb_out = nn.Linear(d_model, 256)

    def encode(self, points):
        # Split points into coordinates and colors
        x = points[:, 0].long().clamp(0, 299)
        y = points[:, 1].long().clamp(0, 299)
        r = points[:, 2].long().clamp(0, 255)
        g = points[:, 3].long().clamp(0, 255)
        b = points[:, 4].long().clamp(0, 255)

        # Embed each dimension
        sequence = torch.stack(
            [
                self.x_embed(x),
                self.y_embed(y),
                self.rgb_embed(r),
                self.rgb_embed(g),
                self.rgb_embed(b),
            ],
            dim=1,
        )

        # Add channel-specific positional encodings
        spatial_seq = sequence[:, :2] + self.spatial_pos
        color_seq = sequence[:, 2:] + self.color_pos
        sequence = torch.cat([spatial_seq, color_seq], dim=1)

        # Encode
        memory = self.encoder(sequence)
        hidden = memory.reshape(memory.shape[0], -1)

        return self.mu(hidden), self.logvar(hidden)

    def decode(self, z):
        # Project latent to sequence
        hidden = self.latent_proj(z)
        memory = hidden.view(-1, 5, self.d_model)

        # Generate target sequence with positional encodings
        spatial_seq = (
            torch.zeros(z.shape[0], 2, self.d_model, device=z.device) + self.spatial_pos
        )
        color_seq = (
            torch.zeros(z.shape[0], 3, self.d_model, device=z.device) + self.color_pos
        )
        tgt = torch.cat([spatial_seq, color_seq], dim=1)

        # Decode
        output = self.decoder(tgt, memory)

        # Get distributions for each dimension
        return (
            self.x_out(output[:, 0]),
            self.y_out(output[:, 1]),
            self.rgb_out(output[:, 2]),
            self.rgb_out(output[:, 3]),
            self.rgb_out(output[:, 4]),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, points):
        mu, logvar = self.encode(points)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def generate(self, num_samples, device="mps"):
        with torch.no_grad():
            z = torch.randn(num_samples, self.mu.out_features, device=device)
            logits = self.decode(z)

            # Sample from categorical distributions
            x = torch.multinomial(F.softmax(logits[0], dim=-1), 1).squeeze()
            y = torch.multinomial(F.softmax(logits[1], dim=-1), 1).squeeze()
            r = torch.multinomial(F.softmax(logits[2], dim=-1), 1).squeeze()
            g = torch.multinomial(F.softmax(logits[3], dim=-1), 1).squeeze()
            b = torch.multinomial(F.softmax(logits[4], dim=-1), 1).squeeze()

            return torch.stack([x, y, r, g, b], dim=-1)


def train_model(model, train_loader, num_epochs=100, learning_rate=3e-4, device="mps"):
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

            loss = recon_loss + 0.01 * kl_loss  # Beta-VAE with beta=0.01

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


def plot_points(points, image_size=(300, 300), point_size=3):
    """
    Plot points on a black image using PIL.

    Parameters:
    points: list of tuples
        List of (x, y) coordinates to plot
    image_size: tuple
        Size of the image as (width, height)
    point_size: int
        Radius of each point
    point_color: tuple
        Color of points in RGB format

    Returns:
    PIL.Image: The resulting image
    """
    # Create a black image
    image = Image.new("RGB", image_size, color="black")
    draw = ImageDraw.Draw(image)

    # Plot each point
    for y, x, r, g, b in points:
        # Ensure coordinates are within image bounds
        if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
            # Calculate bounding box for the circle
            left_up = (x - point_size, y - point_size)
            right_down = (x + point_size, y + point_size)
            # Draw the point as a filled circle
            draw.ellipse([left_up, right_down], fill=(r, g, b))

    return image


def visualize_results_pil(
    original_points: np.ndarray,
    generated_points: np.ndarray,
    image_size: Tuple[int, int] = (300, 300),
    point_size: int = 1,
) -> Image.Image:
    """
    Visualize original and generated points side by side.

    Args:
        original_points: numpy array of shape (N, 5) for original points
        generated_points: numpy array of shape (N, 5) for generated points
        image_size: tuple of (width, height) for each image
        point_size: size of each point in pixels

    Returns:
        PIL Image containing both visualizations side by side
    """
    # Create images for original and generated points
    original_img = plot_points(original_points, image_size, point_size)
    generated_img = plot_points(generated_points, image_size, point_size)

    # Create a new image to hold both visualizations
    combined_img = Image.new("RGB", (image_size[0] * 2 + 10, image_size[1]), "black")

    # Paste the images side by side
    combined_img.paste(original_img, (0, 0))
    combined_img.paste(generated_img, (image_size[0] + 10, 0))

    return combined_img


if __name__ == "__main__":
    # Load data
    xs = np.load("data/pi_xs.npy")
    ys = np.load("data/pi_ys.npy")
    image = np.array(Image.open("data/sparse_pi_colored.jpg"))

    # Create dataset and dataloader
    dataset = PiPointDataset(xs, ys, image)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize and train model
    model = PiPointVAE()
    train_model(model, dataloader)

    torch.save(model.state_dict(), "saved_weights.pt")

    # Generate new points
    generated_points = model.generate(5000)

    # Create visualization
    img = visualize_results_pil(
        dataset.points.numpy().astype(int),
        generated_points.cpu().numpy(),
        image_size=(300, 300),
        point_size=1,
    )

    img.save("visualization.png")
