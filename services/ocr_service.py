from typing import Dict


def extract_text_from_image(image_path: str) -> str:
    """Extract text from a question image using OCR.

    Future implementation will use pytesseract or an OCR service.
    """
    return ""


def clean_ocr_text(raw_text: str) -> str:
    """Normalize OCR output for downstream language model processing."""
    return raw_text.strip()


def ocr_question(image_path: str) -> Dict[str, str]:
    """Process the question image and return extracted text.

    This is a placeholder for OCR pipeline logic.
    """
    extracted_text = extract_text_from_image(image_path)
    return {"image_path": image_path, "text": clean_ocr_text(extracted_text)}
