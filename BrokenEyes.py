import os
from PIL import Image, ImageDraw, ImageFilter
import cv2
import numpy as np
import random

# Define transformation functions (reuse your existing functions)
def cataract(cataract_image_path):
    cataract_image = cv2.imread(cataract_image_path)
    cataract_image = cv2.cvtColor(cataract_image, cv2.COLOR_BGR2RGB)
    hsv_image = cv2.cvtColor(cataract_image, cv2.COLOR_RGB2HSV)
    hsv_image[..., 1] = hsv_image[..., 1] * 0.9
    desaturated_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    haze = np.full_like(desaturated_image, fill_value=220, dtype=np.uint8)
    hazy_image = cv2.addWeighted(desaturated_image, 0.7, haze, 0.3, 0)
    cataract_image = cv2.GaussianBlur(hazy_image, (15, 15), 0)
    return Image.fromarray(cataract_image)

def amd(image):
    image = image.convert("RGBA")
    width, height = image.size
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    center_x, center_y = width // 2, height // 2
    blur_radius = min(width, height) // 3
    draw.ellipse(
        (center_x - blur_radius, center_y - blur_radius, center_x + blur_radius, center_y + blur_radius),
        fill=(0, 0, 0, 180)
    )
    for i in range(blur_radius, blur_radius + 50, 5):
        alpha = int(180 * (1 - (i - blur_radius) / 50))
        draw.ellipse(
            (center_x - i, center_y - i, center_x + i, center_y + i),
            outline=(0, 0, 0, alpha),
            width=10
        )
    return Image.alpha_composite(image, overlay)

def glaucoma(glu_image):
    width, height = glu_image.size
    black_overlay = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(black_overlay)
    vignette_radius = min(width, height) // 4
    center_x, center_y = width // 2, height // 2
    draw.ellipse(
        (center_x - vignette_radius, center_y - vignette_radius, center_x + vignette_radius, center_y + vignette_radius),
        fill=255
    )
    vignette_mask = black_overlay.filter(ImageFilter.GaussianBlur(50))
    return Image.composite(glu_image.convert("RGB"), Image.new("RGB", glu_image.size, "black"), vignette_mask)

def refractive(ref_image_path):
    ref_image = cv2.imread(ref_image_path)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
    blur_kernel_size = random.choice(range(5, 30, 2))
    blur_sigma = random.uniform(0.1, 5.0)
    blurred_image = cv2.GaussianBlur(ref_image, (blur_kernel_size, blur_kernel_size), blur_sigma)
    return Image.fromarray(blurred_image)

def retinopathy(ret_image, max_width=150, max_height=100):
    draw = ImageDraw.Draw(ret_image)
    width, height = ret_image.size
    pouch_count = random.randint(2, 5)
    for _ in range(pouch_count):
        x = random.randint(width // 4, 3 * width // 4) + random.randint(-30, 30)
        y = random.randint(height // 4, 3 * height // 4) + random.randint(-50, 50)
        w = random.randint(max_width // 2, max_width)
        h = random.randint(max_height // 2, max_height)
        shape = [(x - w // 5, y - h // 2), (x + w // 5, y + h // 2)]
        draw.ellipse(shape, fill="black")
    return ret_image

# Main script to apply transformations
input_dir = "data"
output_dirs = {
    "amd": "disorder/amd-data",
    "cataract": "disorder/cataract-data",
    "glaucoma": "disorder/glaucoma-data",
    "refractive": "disorder/refractive-data",
    "retinopathy": "disorder/retinopathy-data",
}

# Create output directories
for key, path in output_dirs.items():
    os.makedirs(os.path.join(path, "human"), exist_ok=True)
    os.makedirs(os.path.join(path, "non-human"), exist_ok=True)

# Process images
for category in ["humans", "non-humans"]:
    input_category_path = os.path.join(input_dir, category)
    for image_name in os.listdir(input_category_path):
        image_path = os.path.join(input_category_path, image_name)
        if category == "humans":
            out_category = "human"
        else:
            out_category = "non-human"

        # Open the image
        original_image = Image.open(image_path)

        # Apply transformations
        cataract_image = cataract(image_path)
        cataract_image.save(os.path.join(output_dirs["cataract"], out_category, image_name))

        amd_image = amd(original_image)
        amd_image = amd_image.convert("RGB")  # Convert RGBA to RGB
        amd_image.save(os.path.join(output_dirs["amd"], out_category, image_name))

        glaucoma_image = glaucoma(original_image)
        glaucoma_image = glaucoma_image.convert("RGB")  # Convert in case it's RGBA
        glaucoma_image.save(os.path.join(output_dirs["glaucoma"], out_category, image_name))

        refractive_image = refractive(image_path)
        refractive_image.save(os.path.join(output_dirs["refractive"], out_category, image_name))

        retinopathy_image = retinopathy(original_image)
        retinopathy_image = retinopathy_image.convert("RGB")  # Convert RGBA to RGB
        retinopathy_image.save(os.path.join(output_dirs["retinopathy"], out_category, image_name))


print("Processing complete!")
