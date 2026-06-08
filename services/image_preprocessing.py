from pathlib import Path


def load_image(image_path: str) -> Path:
    """Load an image from disk and return a reference to it.

    This function is a placeholder for future preprocessing logic.
    """
    return Path(image_path)


def preprocess_question_image(image_path: str) -> dict:
    """Prepare an image for OCR or model inference.

    Returns a normalized representation of the image.
    """
    image_file = load_image(image_path)
    return {"path": str(image_file), "status": "ready"}
