from app.config import settings

from services.llm_service import solve_image_question, solve_text_question


def test_empty_question_returns_failed_status() -> None:
    result = solve_text_question("")

    assert result["status"] == "failed"
    assert result["error"] == "Question text is empty."
    assert result["answer"] == ""
    assert result["latency_ms"] >= 0


def test_mock_mode_two_plus_two_question_returns_B(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_mock_mode", True)
    result = solve_text_question("What is 2 + 2?")

    assert result["status"] == "success"
    assert result["answer"] == "B"
    assert result["confidence"] == 0.95
    assert result["error"] is None


def test_unknown_mock_question_returns_safe_result(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_mock_mode", True)
    result = solve_text_question("What is the capital of France?")

    assert result["status"] == "success"
    assert result["answer"] == "unknown"
    assert result["error"] is None


def test_solve_image_question_missing_image_fails_safely() -> None:
    result = solve_image_question("dummy_path.png")

    assert result["status"] == "failed"
    assert "not found" in result["error"].lower()
    assert result["latency_ms"] >= 0


def test_mock_vision_two_plus_two_sample_returns_B(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_mock_mode", True)
    result = solve_image_question("data/sample_questions/q01_text.png")

    assert result["status"] == "success"
    assert result["answer"] == "B"
    assert result["confidence"] == 0.90
    assert result["error"] is None


def test_llm_service_result_keys(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_mock_mode", True)
    result = solve_text_question("What is 2 + 2?")

    assert set(result.keys()) == {
        "answer",
        "explanation",
        "confidence",
        "raw_response",
        "status",
        "error",
        "latency_ms",
    }


def test_real_mode_missing_model_name_fails_gracefully(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_mock_mode", False)
    monkeypatch.setattr(settings, "llm_model_name", None)
    result = solve_text_question("What is 2 + 2?")

    assert result["status"] == "failed"
    assert "model name" in result["error"].lower()
    assert result["answer"] == ""
