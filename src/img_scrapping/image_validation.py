import cv2
import numpy as np


def is_valid_image_url(url):
    if not url or not url.startswith("http"):
        return False

    skip_patterns = [
        "logo",
        "icon",
        "avatar",
        "thumb",
        "button",
        "banner",
        "pixel",
        "1x1",
        "transparent",
        "spacer",
        "blank",
    ]

    url_lower = url.lower()
    for pattern in skip_patterns:
        if pattern in url_lower:
            return False

    return any(ext in url_lower for ext in [".jpg", ".jpeg", ".png", ".webp"])


def is_suitable_for_face_detection(image_data):
    try:
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)

        if img is None:
            return False, "Could not decode image"

        height, width = img.shape[:2]
        if width < 100 or height < 100:
            return False, f"Too small: {width}x{height}"

        if img.shape[2] not in [1, 3, 4]:
            return False, f"Invalid number of channels: {img.shape[2]}"

        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 3:
            return False, f"Extreme aspect ratio: {aspect_ratio:.1f}:1"

        if len(image_data) < 5000:
            return False, "File too small (probably icon)"

        return True, "OK"

    except Exception as e:
        return False, f"Image analysis error: {e}"
