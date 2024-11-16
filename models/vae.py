from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class PiPointVAE(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=3, latent_dim=32):
        super().__init__()
        self.config = {
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "latent_dim": latent_dim,
        }

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
        memory = hidden.view(-1, 5, self.config["d_model"])

        # Generate target sequence with positional encodings
        spatial_seq = (
            torch.zeros(z.shape[0], 2, self.config["d_model"], device=z.device)
            + self.spatial_pos
        )
        color_seq = (
            torch.zeros(z.shape[0], 3, self.config["d_model"], device=z.device)
            + self.color_pos
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

    def generate(self, num_samples):
        with torch.no_grad():
            z = torch.randn(
                num_samples, self.mu.out_features, device=next(self.parameters()).device
            )
            logits = self.decode(z)

            # Sample from categorical distributions
            x = torch.multinomial(F.softmax(logits[0], dim=-1), 1).squeeze()
            y = torch.multinomial(F.softmax(logits[1], dim=-1), 1).squeeze()
            r = torch.multinomial(F.softmax(logits[2], dim=-1), 1).squeeze()
            g = torch.multinomial(F.softmax(logits[3], dim=-1), 1).squeeze()
            b = torch.multinomial(F.softmax(logits[4], dim=-1), 1).squeeze()

            return torch.stack([x, y, r, g, b], dim=-1)

    def save(self, save_dir, **kwargs):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config,
                "training_cfg": {**kwargs},
                "model_state_dict": self.state_dict(),
            },
            save_dir / "saved_model.pt",
        )

    @classmethod
    def load(cls, load_dir, device="mps"):
        load_dir = Path(load_dir)
        # Load saved state
        state_dict = torch.load(
            load_dir / "saved_model.pt", map_location=device, weights_only=True
        )

        # Create new model instance with saved config
        model = cls(**state_dict["config"])
        model.load_state_dict(state_dict["model_state_dict"])
        model = model.to(device)
        return model
