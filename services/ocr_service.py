from pathlib import Path
from typing import Dict

import pytesseract

from services.image_preprocessing import preprocess_for_ocr, save_debug_image


def extract_text_from_image(image_path: str, save_debug: bool = False) -> dict:
    """Extract text from an image and return a structured OCR result."""
    debug_image_path = None
    try:
        processed_image = preprocess_for_ocr(image_path)
    except FileNotFoundError as exc:
        return {"text": "", "status": "failed", "error": str(exc), "debug_image_path": None}
    except ValueError as exc:
        return {"text": "", "status": "failed", "error": str(exc), "debug_image_path": None}
    except Exception as exc:
        return {"text": "", "status": "failed", "error": f"OCR preprocessing failed: {exc}", "debug_image_path": None}

    if save_debug:
        image_name = Path(image_path).stem + "_debug.png"
        debug_path = Path("outputs") / "debug" / image_name
        try:
            debug_image_path = save_debug_image(processed_image, str(debug_path))
        except Exception as exc:
            return {"text": "", "status": "failed", "error": f"Failed to save debug image: {exc}", "debug_image_path": None}

    try:
        raw_text = pytesseract.image_to_string(processed_image, lang="eng", config="--oem 3 --psm 6")
    except pytesseract.pytesseract.TesseractNotFoundError as exc:
        return {"text": "", "status": "failed", "error": f"Tesseract not found: {exc}", "debug_image_path": debug_image_path}
    except Exception as exc:
        return {"text": "", "status": "failed", "error": f"OCR engine error: {exc}", "debug_image_path": debug_image_path}

    cleaned_text = clean_ocr_text(raw_text)
    if not cleaned_text:
        return {"text": "", "status": "failed", "error": "OCR extracted no text from image.", "debug_image_path": debug_image_path}

    return {"text": cleaned_text, "status": "success", "error": None, "debug_image_path": debug_image_path}


def clean_ocr_text(raw_text: str) -> str:
    """Normalize OCR output for downstream language model processing."""
    lines = [line.strip() for line in raw_text.strip().splitlines()]
    normalized_lines = []
    previous_empty = False

    for line in lines:
        if not line:
            if previous_empty:
                continue
            previous_empty = True
            normalized_lines.append("")
            continue

        normalized_lines.append(line)
        previous_empty = False

    return "\n".join(normalized_lines).strip()


def ocr_question(image_path: str) -> Dict[str, str]:
    """Process the question image and return extracted text."""
    result = extract_text_from_image(image_path)
    return {
        "image_path": image_path,
        "text": clean_ocr_text(result["text"]),
        "status": result["status"],
        "error": result["error"],
        "debug_image_path": result["debug_image_path"],
    }
