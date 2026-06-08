from services.solver_pipeline import run_ocr_llm_pipeline, run_vision_llm_pipeline


def test_run_ocr_llm_pipeline_returns_expected_keys() -> None:
    result = run_ocr_llm_pipeline("dummy_image.png")
    assert result["pipeline"] == "ocr_llm"
    assert "solution" in result


def test_run_vision_llm_pipeline_returns_expected_keys() -> None:
    result = run_vision_llm_pipeline("dummy_image.png")
    assert result["pipeline"] == "vision_llm"
    assert "solution" in result
