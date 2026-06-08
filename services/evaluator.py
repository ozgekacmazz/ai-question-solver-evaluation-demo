import json
from pathlib import Path
from typing import Any

import pandas as pd

from services.solver_pipeline import solve_question_image


def load_ground_truth(path: str = "data/ground_truth.json") -> list[dict[str, Any]]:
    """Load ground truth data from a JSON file."""
    ground_truth_file = Path(path)
    if not ground_truth_file.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    try:
        data = json.loads(ground_truth_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Ground truth file is not valid JSON: {exc}")

    if not isinstance(data, list):
        raise ValueError("Ground truth file must contain a JSON list of records.")

    return data


def normalize_answer(answer: str) -> str:
    """Normalize predicted answers for comparison."""
    if not isinstance(answer, str):
        return ""

    normalized = answer.strip().upper()
    if not normalized:
        return ""

    normalized = normalized.replace(",", " ").replace(".", " ")
    tokens = normalized.split()
    for token in tokens:
        candidate = token.strip()
        if candidate in {"A", "B", "C", "D", "E"}:
            return candidate
        if candidate.endswith(")") and candidate[:-1] in {"A", "B", "C", "D", "E"}:
            return candidate[:-1]

    return ""


def evaluate_answer(predicted_answer: str, correct_answer: str) -> dict[str, Any]:
    """Compare the predicted answer with the correct answer."""
    predicted_normalized = normalize_answer(predicted_answer)
    correct_normalized = normalize_answer(correct_answer)
    is_correct = bool(predicted_normalized and predicted_normalized == correct_normalized)
    needs_manual_review = predicted_normalized == "" or not is_correct

    return {
        "predicted_normalized": predicted_normalized,
        "correct_normalized": correct_normalized,
        "is_correct": is_correct,
        "needs_manual_review": needs_manual_review,
    }


def evaluate_question(record: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a single question record against the OCR + LLM pipeline."""
    question_id = record.get("id", "")
    image_path = record.get("image_path", "")
    question_type = record.get("question_type", "")
    correct_answer = record.get("correct_answer", "")

    try:
        result = solve_question_image(image_path, mode="ocr")
    except Exception as exc:
        return {
            "question_id": question_id,
            "image_path": image_path,
            "question_type": question_type,
            "correct_answer": correct_answer,
            "predicted_answer": "",
            "is_correct": False,
            "needs_manual_review": True,
            "status": "failed",
            "error": str(exc),
            "confidence": 0.0,
            "ocr_text": "",
        }

    predicted_answer = result.get("answer", "")
    comparison = evaluate_answer(predicted_answer, correct_answer)

    return {
        "question_id": question_id,
        "image_path": image_path,
        "question_type": question_type,
        "correct_answer": correct_answer,
        "predicted_answer": predicted_answer,
        "is_correct": comparison["is_correct"],
        "needs_manual_review": comparison["needs_manual_review"],
        "status": result.get("status", "failed"),
        "error": result.get("error"),
        "confidence": float(result.get("confidence", 0.0) or 0.0),
        "ocr_text": result.get("ocr_result", {}).get("text", ""),
    }


def run_batch_evaluation(
    ground_truth_path: str = "data/ground_truth.json",
    output_csv_path: str = "outputs/results.csv",
) -> list[dict[str, Any]]:
    """Run batch evaluation over a ground truth dataset and save results."""
    records = load_ground_truth(ground_truth_path)
    results = [evaluate_question(record) for record in records]

    output_file = Path(output_csv_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

    return results
