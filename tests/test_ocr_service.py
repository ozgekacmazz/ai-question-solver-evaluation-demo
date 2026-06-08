import shutil
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from services.ocr_service import clean_ocr_text, extract_text_from_image, ocr_question


def test_clean_ocr_text_removes_whitespace() -> None:
    assert clean_ocr_text("  Example question  ") == "Example question"


def test_extract_text_from_image_invalid_path_returns_failed_status() -> None:
    result = extract_text_from_image("does_not_exist.png")

    assert result["status"] == "failed"
    assert result["text"] == ""
    assert result["debug_image_path"] is None
    assert "not found" in result["error"].lower()


def test_extract_text_from_image_returns_expected_keys(tmp_path: Path) -> None:
    if shutil.which("tesseract") is None:
        pytest.skip("Tesseract OCR binary is not installed.")

    sample_path = tmp_path / "sample_question.png"
    image = Image.new("RGB", (800, 240), color="white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (16, 16),
        "What is 2 + 2?\nA) 3\nB) 4\nC) 5",
        fill="black",
    )
    image.save(sample_path)

    result = extract_text_from_image(str(sample_path), save_debug=True)

    assert set(result.keys()) == {"text", "status", "error", "debug_image_path"}
    assert result["status"] in {"success", "failed"}
    assert result["debug_image_path"] is not None
    assert Path(result["debug_image_path"]).exists()


def test_clean_ocr_text_collapses_empty_lines() -> None:
    raw_text = "  What is 2 + 2?\n\n\nA) 3\n\nB) 4\nC) 5  "
    cleaned = clean_ocr_text(raw_text)

    assert cleaned == "What is 2 + 2?\n\nA) 3\n\nB) 4\nC) 5"


def test_ocr_question_returns_expected_keys() -> None:
    result = ocr_question("dummy_path.png")
    assert "image_path" in result
    assert "text" in result
    assert "status" in result
    assert "error" in result
    assert "debug_image_path" in result
