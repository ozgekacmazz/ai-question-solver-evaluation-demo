from app.config import settings

from scripts.create_benchmark_questions import create_all_benchmark_questions
from scripts.create_sample_questions import create_all_sample_questions
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


def test_mock_vision_returns_B_for_all_known_samples(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_mock_mode", True)
    create_all_sample_questions()

    expected_confidences = {
        "q01_text.png": 0.90,
        "q02_math.png": 0.90,
        "q03_equation.png": 0.91,
        "q04_table.png": 0.92,
        "q05_chart.png": 0.92,
        "q06_geometry.png": 0.93,
        "q07_mixed.png": 0.90,
        "q08_noisy.png": 0.88,
    }

    for file_name, confidence in expected_confidences.items():
        result = solve_image_question(f"data/sample_questions/{file_name}")
        assert result["status"] == "success"
        assert result["answer"] == "B"
        assert result["confidence"] == confidence
        assert result["error"] is None


def test_mock_vision_does_not_hardcode_benchmark_answers(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_mock_mode", True)
    create_all_benchmark_questions()

    result = solve_image_question("data/benchmark_questions/q09_parabola_vertex.png")

    assert result["status"] == "success"
    assert result["answer"] == "unknown"
    assert result["confidence"] == 0.0
    assert "real model evaluation" in result["explanation"]


def test_mock_text_rules_return_B_for_known_patterns(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_mock_mode", True)

    examples = [
        "What is 12 / 3 + 2?",
        "Solve for x: 2x + 3 = 11",
        "Product Price Notebook 20 Question: Which product costs 20?",
        "Rectangle width 4 height 3 Question: What is the area?",
        "4 stars and 2 more stars. How many stars are there in total?",
    ]

    for text in examples:
        result = solve_text_question(text)
        assert result["status"] == "success"
        assert result["answer"] == "B"
        assert result["confidence"] > 0


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
