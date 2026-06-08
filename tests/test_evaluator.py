from pathlib import Path

import pandas as pd

from services.evaluator import (
    evaluate_answer,
    load_ground_truth,
    normalize_answer,
    run_batch_evaluation,
)


def test_normalize_answer_single_letter() -> None:
    assert normalize_answer("B") == "B"


def test_normalize_answer_with_prefix() -> None:
    assert normalize_answer("Answer: B") == "B"


def test_normalize_answer_sentence() -> None:
    assert normalize_answer("The correct answer is C.") == "C"


def test_evaluate_answer_correct() -> None:
    result = evaluate_answer("B", "B")
    assert result["is_correct"] is True
    assert result["needs_manual_review"] is False


def test_evaluate_answer_incorrect() -> None:
    result = evaluate_answer("A", "B")
    assert result["is_correct"] is False
    assert result["needs_manual_review"] is True


def test_load_ground_truth_returns_list() -> None:
    records = load_ground_truth("data/ground_truth.json")
    assert isinstance(records, list)
    assert records


def test_run_batch_evaluation_creates_csv(tmp_path: Path, monkeypatch) -> None:
    output_csv = tmp_path / "results.csv"

    import services.evaluator as evaluator_module

    def mock_solve_question_image(image_path: str, mode: str = "ocr"):
        return {
            "ocr_result": {"text": "What is 2 + 2?\nA) 3\nB) 4\nC) 5"},
            "answer": "B",
            "explanation": "2 + 2 equals 4, so the correct option is B.",
            "confidence": 0.95,
            "raw_response": "Answer: B\nExplanation: 2 + 2 equals 4.",
            "status": "success",
            "error": None,
        }

    monkeypatch.setattr(evaluator_module, "solve_question_image", mock_solve_question_image)

    results = evaluator_module.run_batch_evaluation("data/ground_truth.json", str(output_csv))
    assert output_csv.exists()
    df = pd.read_csv(output_csv)
    assert "question_id" in df.columns
    assert len(results) == len(df)
