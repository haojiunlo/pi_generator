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
