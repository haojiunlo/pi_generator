# Colored Pi Point Generator
## Folder Structure
```
.
root/
├─ dataset_utils/
│  ├─ dataset.py    # PyTorch dataset implementations
├─ models/
│  ├─ vae.py        # Transformer-VAE architecture
├─ utils/
│  ├─ utils.py      # utils function for visualization and dist comparison
├─ run.py           # Training script and create distribution comparison plot
├─ generate.py      # Sample generation
```
## Usage
* Prepare dataset `pi_xs.npy`, `pi_ys.npy`, `sparse_pi_colored.jpg`
### Training Script
* Specify parameters and run training script and generate image for visualization
    ```bash
    python script.py
        --d_model (default: 256) [Number of expected features in the transformer input]
        --nhead (default: 8) [Number of heads in the multiheadattention models]
        --num_layers (default: 3) [Number of sub-encoder/decoder-layers in the transformers]
        --latent_dim (default: 32) [Latent dim of VAE]
        --num_epochs (default: 100) [Number of training epochs]
        --learning_rate (default: 1e-4) [Learning rate]
        --batch_size (default: 32) [Batch size]
        --device (default: mps) [Device: cpu, cuda, or mps]
        --saved_dir (default: saved_model) [Directory to save the trained model]
        --dataset_dir (default: data) [Training data location]
        --beta (default: 0.01) [Beta parameter for beta-VAE]
    ```
* Trained model `saved_model.pt` and visualization `feature_distribution_comparison.png`, `spatial_distribution_comparison.png` and `visualization.png` will save at `outputs/[--saved_dir]`
### Generator Script
* Use trained model to generate points and visualization
    ```bash
    python generate.py
            --model_dir (default: saved_model) [Which model to load for generating]
            --device (default: mps) [Device: cpu, cuda, or mps]
            --num_samples (default: 5000) [Number of samples to generate]
            --saved_folder (default: saved_outputs) [Output directory for generated samples]
    ```
* Generated outputs `gen_image.png` and `gen_points.npy` will save at `gen_outputs/[--saved_folder]`
