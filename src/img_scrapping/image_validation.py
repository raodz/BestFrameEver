import yaml

from src.img_scrapping.img_download_utils import bytes_to_cv_image
from src.utils.constants import (
    MAX_ASPECT_RATIO,
    MIN_IMAGE_BYTE_SIZE,
    MIN_IMAGE_DIMENSION,
    N_BGR_CHANNELS,
    N_BGRA_CHANNELS,
    N_SINGLE_CHANNEL,
)
from src.utils.paths import IMAGE_VALIDATION_CONFIG_ABS_PATH

with open(IMAGE_VALIDATION_CONFIG_ABS_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

SKIP_PATTERNS = cfg["skip_patterns"]
ALLOWED_EXTENSIONS = cfg["allowed_extensions"]


def is_valid_image_url(url):
    if not url or not url.startswith("http"):
        return False

    url_lower = url.lower()
    if any(pattern in url_lower for pattern in SKIP_PATTERNS):
        return False

    return any(url_lower.endswith(ext) for ext in ALLOWED_EXTENSIONS)


def is_suitable_for_face_detection(image_bytes):
    img = bytes_to_cv_image(image_bytes)

    if img is None:
        return False, "Could not decode image"

    height, width = img.shape[:2]
    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        return False, f"Too small: {width}x{height}"

    if img.shape[2] not in [N_SINGLE_CHANNEL, N_BGR_CHANNELS, N_BGRA_CHANNELS]:
        return False, f"Invalid number of channels: {img.shape[2]}"

    aspect_ratio = max(width, height) / min(width, height)
    if aspect_ratio > MAX_ASPECT_RATIO:
        return False, f"Extreme aspect ratio: {aspect_ratio:.1f}:1"

    if len(image_bytes) < MIN_IMAGE_BYTE_SIZE:
        return False, "File too small (probably icon)"

    return True, "OK"
