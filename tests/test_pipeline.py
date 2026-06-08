from services.solver_pipeline import run_ocr_llm_pipeline, run_vision_llm_pipeline, solve_question_image


def test_run_ocr_llm_pipeline_returns_expected_keys() -> None:
    result = run_ocr_llm_pipeline("dummy_image.png")
    assert result["pipeline"] == "ocr_llm"
    assert "solution" in result


def test_run_vision_llm_pipeline_returns_expected_keys() -> None:
    result = run_vision_llm_pipeline("dummy_image.png")
    assert result["pipeline"] == "vision_llm"
    assert "solution" in result


def test_solve_question_image_ocr_mode(monkeypatch) -> None:
    def mock_run_ocr_llm_pipeline(image_path: str) -> dict:
        return {
            "pipeline": "ocr_llm",
            "image_path": image_path,
            "ocr_result": {"text": "What is 2 + 2?", "status": "success"},
            "llm_result": {"status": "success"},
            "answer": "B",
            "solution": "B",
            "explanation": "2 + 2 equals 4.",
            "confidence": 0.95,
            "status": "success",
            "error": None,
        }

    monkeypatch.setattr("services.solver_pipeline.run_ocr_llm_pipeline", mock_run_ocr_llm_pipeline)

    result = solve_question_image("question.png", mode="ocr")

    assert result["pipeline"] == "ocr_llm"
    assert result["answer"] == "B"


def test_solve_question_image_vision_mode(monkeypatch) -> None:
    def mock_run_vision_llm_pipeline(image_path: str) -> dict:
        return {
            "pipeline": "vision_llm",
            "image_path": image_path,
            "ocr_result": {},
            "llm_result": {"status": "success"},
            "answer": "B",
            "solution": "B",
            "explanation": "Vision selected B.",
            "confidence": 0.90,
            "status": "success",
            "error": None,
        }

    monkeypatch.setattr("services.solver_pipeline.run_vision_llm_pipeline", mock_run_vision_llm_pipeline)

    result = solve_question_image("question.png", mode="vision")

    assert result["pipeline"] == "vision_llm"
    assert result["answer"] == "B"


def test_solve_question_image_both_mode_returns_recommendation(monkeypatch) -> None:
    def mock_run_ocr_llm_pipeline(image_path: str) -> dict:
        return {
            "pipeline": "ocr_llm",
            "image_path": image_path,
            "ocr_result": {"text": "What is 2 + 2?", "status": "success"},
            "llm_result": {"status": "success"},
            "answer": "B",
            "solution": "B",
            "explanation": "OCR selected B.",
            "confidence": 0.95,
            "status": "success",
            "error": None,
        }

    def mock_run_vision_llm_pipeline(image_path: str) -> dict:
        return {
            "pipeline": "vision_llm",
            "image_path": image_path,
            "ocr_result": {},
            "llm_result": {"status": "success"},
            "answer": "B",
            "solution": "B",
            "explanation": "Vision selected B.",
            "confidence": 0.90,
            "status": "success",
            "error": None,
        }

    monkeypatch.setattr("services.solver_pipeline.run_ocr_llm_pipeline", mock_run_ocr_llm_pipeline)
    monkeypatch.setattr("services.solver_pipeline.run_vision_llm_pipeline", mock_run_vision_llm_pipeline)

    result = solve_question_image("question.png", mode="both")

    assert result["pipeline"] == "both"
    assert result["recommended_pipeline"] == "both_agree"
    assert "comparison_summary" in result
    assert result["answer"] == "B"


def test_both_mode_does_not_recommend_agreement_when_both_unknown(monkeypatch) -> None:
    def mock_unknown_pipeline(pipeline: str):
        def _mock(image_path: str) -> dict:
            return {
                "pipeline": pipeline,
                "image_path": image_path,
                "ocr_result": {},
                "llm_result": {"status": "success"},
                "answer": "unknown",
                "solution": "unknown",
                "explanation": "Unknown.",
                "confidence": 0.0,
                "status": "success",
                "error": None,
            }

        return _mock

    monkeypatch.setattr("services.solver_pipeline.run_ocr_llm_pipeline", mock_unknown_pipeline("ocr_llm"))
    monkeypatch.setattr("services.solver_pipeline.run_vision_llm_pipeline", mock_unknown_pipeline("vision_llm"))

    result = solve_question_image("question.png", mode="both")

    assert result["recommended_pipeline"] == "none"
    assert result["recommended_pipeline"] != "both_agree"
    assert result["answer"] == "unknown"
    assert result["confidence"] == 0.0
    assert result["status"] == "success_with_no_reliable_answer"
    assert result["comparison_summary"] == "Neither pipeline produced a reliable answer."


def test_both_mode_recommends_vision_when_ocr_unknown(monkeypatch) -> None:
    def mock_run_ocr_llm_pipeline(image_path: str) -> dict:
        return {
            "pipeline": "ocr_llm",
            "image_path": image_path,
            "ocr_result": {"text": "", "status": "failed"},
            "llm_result": {"status": "success"},
            "answer": "unknown",
            "solution": "unknown",
            "explanation": "Unknown.",
            "confidence": 0.0,
            "status": "success",
            "error": None,
        }

    def mock_run_vision_llm_pipeline(image_path: str) -> dict:
        return {
            "pipeline": "vision_llm",
            "image_path": image_path,
            "ocr_result": {},
            "llm_result": {"status": "success"},
            "answer": "B",
            "solution": "B",
            "explanation": "Vision selected B.",
            "confidence": 0.92,
            "status": "success",
            "error": None,
        }

    monkeypatch.setattr("services.solver_pipeline.run_ocr_llm_pipeline", mock_run_ocr_llm_pipeline)
    monkeypatch.setattr("services.solver_pipeline.run_vision_llm_pipeline", mock_run_vision_llm_pipeline)

    result = solve_question_image("question.png", mode="both")

    assert result["recommended_pipeline"] == "vision_llm"
    assert result["answer"] == "B"
    assert result["confidence"] == 0.92


def test_solve_question_image_unsupported_mode_returns_failed_status() -> None:
    result = solve_question_image("question.png", mode="invalid")

    assert result["status"] == "failed"
    assert "supported modes" in result["error"].lower()
