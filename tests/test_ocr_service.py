from services.ocr_service import clean_ocr_text, ocr_question


def test_clean_ocr_text_removes_whitespace() -> None:
    assert clean_ocr_text("  Example question  ") == "Example question"


def test_ocr_question_returns_expected_keys() -> None:
    result = ocr_question("dummy_path.png")
    assert "image_path" in result
    assert "text" in result
