from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw


def plot_points(points, image_size=(300, 300), point_size=3):
    image = Image.new("RGB", image_size, color="black")
    draw = ImageDraw.Draw(image)

    for y, x, r, g, b in points:
        if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
            left_up = (x - point_size, y - point_size)
            right_down = (x + point_size, y + point_size)
            draw.ellipse([left_up, right_down], fill=(r, g, b))

    return image


def visualize_results_pil(
    original_points,
    generated_points,
    image_size=(300, 300),
    point_size=1,
):
    # Create images for original and generated points
    original_img = plot_points(original_points, image_size, point_size)
    generated_img = plot_points(generated_points, image_size, point_size)

    # Create a new image to hold both visualizations
    combined_img = Image.new("RGB", (image_size[0] * 2 + 10, image_size[1]), "black")

    # Paste the images side by side
    combined_img.paste(original_img, (0, 0))
    combined_img.paste(generated_img, (image_size[0] + 10, 0))

    return combined_img


def compare_distributions(
    real_points,
    generated_points,
    saved_dir="output",
    feature_names=["x", "y", "r", "g", "b"],
):
    saved_dir = Path(saved_dir)
    saved_dir.mkdir(parents=True, exist_ok=True)
    n_features = real_points.shape[1]

    fig, axes = plt.subplots(1, n_features, figsize=(20, 8))
    for i in range(n_features):
        axes[i].hist(
            real_points[:, i],
            bins=50,
            alpha=0.5,
            density=True,
            color="blue",
            label="Real",
        )
        axes[i].hist(
            generated_points[:, i],
            bins=50,
            alpha=0.5,
            density=True,
            color="red",
            label="Generated",
        )
        axes[i].set_title(f"{feature_names[i]} Distribution")
        axes[i].legend()
    fig.savefig(saved_dir / "feature_distribution_comparison.png")

    plt.tight_layout()
    plt.figure(figsize=(6, 6))
    plt.scatter(
        real_points[:, 1], 300 - real_points[:, 0], c="blue", alpha=0.5, label="Real"
    )
    plt.scatter(
        generated_points[:, 1],
        300 - generated_points[:, 0],
        c="red",
        alpha=0.5,
        label="Generated",
    )
    plt.title("Spatial Distribution Comparison")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.legend()
    plt.savefig(saved_dir / "spatial_distribution_comparison.png")
