from pathlib import Path

import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk and return a NumPy array.

    Raises:
        FileNotFoundError: When the image path does not exist.
        ValueError: When OpenCV cannot read the image file.
    """
    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_file))
    if image is None:
        try:
            file_bytes = np.fromfile(str(image_file), dtype=np.uint8)
            if file_bytes.size > 0:
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        except Exception:
            image = None

    if image is None:
        raise ValueError(f"Could not read image file: {image_path}")

    return image


def preprocess_for_ocr(image_path: str) -> np.ndarray:
    """Load and preprocess an image for OCR.

    The preprocessing steps include grayscale conversion, optional resizing,
    denoising, and thresholding to improve text visibility.
    """
    image = load_image(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape[:2]
    if width < 1000:
        scale = 1000 / float(width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    processed = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )

    return processed


def save_debug_image(image, output_path: str) -> str:
    """Save a processed image to disk for debugging."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_file.suffix.lower() or ".png"
    if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}:
        suffix = ".png"

    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise IOError(f"Failed to encode debug image for: {output_path}")

    with output_file.open("wb") as output_file_handle:
        output_file_handle.write(encoded.tobytes())

    return str(output_file)
