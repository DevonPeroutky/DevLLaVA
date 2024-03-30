import base64
import requests

from io import BytesIO
from PIL import Image


def determine_media_type(image_url) -> str:
    image_media_url = image_url.split("?")[0]
    image_media_type = "image/png"

    # derive media_type from image_url
    if image_media_url.endswith(".png"):
        image_media_type = "image/png"
    elif image_media_url.endswith(".jpeg") or image_media_url.endswith(".jpg"):
        image_media_type = "image/jpeg"
    elif image_media_url.endswith(".webp"):
        image_media_type = "image/webp"
    else:
        raise ValueError("Unsupported image format", image_url)
    return image_media_type


def download_and_encode_image(image_url) -> str:
    # Download the image data
    response = requests.get(image_url)

    # Check if the request was successful
    if response.status_code == 200:

        # Download the image into a PIL and then resize it such that no dimension is greater than 8000 pixels
        image = Image.open(BytesIO(response.content))
        encode_pil_to_base64(image)
    else:
        print(f"Error: Failed to download image from {image_url}")
        return None


def encode_pil_to_base64(image: Image, max_dimension: int = 4096, image_format: str = "JPEG") -> str:

    # Download the image into a PIL and then resize it such that no dimension is greater than 8000 pixels
    if image.width > max_dimension or image.height > max_dimension:
        if image.width > image.height:
            new_width = max_dimension
            new_height = int(image.height * (max_dimension / image.width))
        else:
            new_height = max_dimension
            new_width = int(image.width * (max_dimension / image.height))
        image = image.resize((new_width, new_height))

    # Encode the PIL image data as a base64 string
    buffered = BytesIO()
    image.save(buffered, format=image_format)
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return base64_image
