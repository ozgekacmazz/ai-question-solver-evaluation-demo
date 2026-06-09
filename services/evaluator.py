import json
from pathlib import Path
from typing import Any

import pandas as pd

from services.llm_service import normalize_answer as normalize_llm_answer
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
    return normalize_llm_answer(answer)


def evaluate_answer(predicted_answer: str, correct_answer: str) -> dict[str, Any]:
    """Compare the predicted answer with the correct answer."""
    predicted_normalized = normalize_answer(predicted_answer)
    correct_normalized = normalize_answer(correct_answer)
    is_correct = bool(predicted_normalized and predicted_normalized == correct_normalized)
    needs_manual_review = predicted_normalized == "unknown" or not is_correct

    return {
        "predicted_normalized": predicted_normalized,
        "correct_normalized": correct_normalized,
        "is_correct": is_correct,
        "needs_manual_review": needs_manual_review,
    }


def _extract_pipeline_value(result: dict[str, Any], pipeline_key: str, value_key: str, default: Any = "") -> Any:
    pipeline_result = result.get(pipeline_key, {})
    if not isinstance(pipeline_result, dict):
        return default
    return pipeline_result.get(value_key, default)


def evaluate_question(record: dict[str, Any], mode: str = "both") -> dict[str, Any]:
    """Evaluate a single question record against the configured solve pipeline."""
    question_id = record.get("id", "")
    image_path = record.get("image_path", "")
    question_type = record.get("question_type", "")
    correct_answer = record.get("correct_answer", "")

    try:
        result = solve_question_image(image_path, mode=mode)
    except Exception as exc:
        return {
            "id": question_id,
            "question_id": question_id,
            "image_path": image_path,
            "question_type": question_type,
            "correct_answer": correct_answer,
            "predicted_answer": "",
            "is_correct": False,
            "needs_manual_review": True,
            "recommended_pipeline": "",
            "ocr_answer": "",
            "vision_answer": "",
            "ocr_confidence": 0.0,
            "vision_confidence": 0.0,
            "final_confidence": 0.0,
            "status": "failed",
            "error": str(exc),
            "confidence": 0.0,
            "ocr_text": "",
            "ocr_raw_response": "",
            "vision_raw_response": "",
            "ocr_original_answer": "",
            "vision_original_answer": "",
            "ocr_answer_repaired": False,
            "vision_answer_repaired": False,
            "ocr_repair_reason": "",
            "vision_repair_reason": "",
            "adaptive_selected_mode": "",
            "adaptive_initial_mode": "",
            "adaptive_fallback_mode": "",
            "router_decision": "",
        }

    predicted_answer = result.get("answer", "")
    comparison = evaluate_answer(predicted_answer, correct_answer)
    final_confidence = float(result.get("confidence", 0.0) or 0.0)
    ocr_confidence = float(_extract_pipeline_value(result, "ocr_pipeline_result", "confidence", 0.0) or 0.0)
    vision_confidence = float(_extract_pipeline_value(result, "vision_pipeline_result", "confidence", 0.0) or 0.0)
    ocr_llm_result = _extract_pipeline_value(result, "ocr_pipeline_result", "llm_result", {})
    vision_llm_result = _extract_pipeline_value(result, "vision_pipeline_result", "llm_result", {})
    ocr_raw_response = ocr_llm_result.get("raw_response", "") if isinstance(ocr_llm_result, dict) else ""
    vision_raw_response = vision_llm_result.get("raw_response", "") if isinstance(vision_llm_result, dict) else ""
    ocr_original_answer = _extract_pipeline_value(result, "ocr_pipeline_result", "original_answer", "")
    vision_original_answer = _extract_pipeline_value(result, "vision_pipeline_result", "original_answer", "")
    ocr_answer_repaired = bool(_extract_pipeline_value(result, "ocr_pipeline_result", "answer_repaired", False))
    vision_answer_repaired = bool(_extract_pipeline_value(result, "vision_pipeline_result", "answer_repaired", False))
    ocr_repair_reason = _extract_pipeline_value(result, "ocr_pipeline_result", "repair_reason", "")
    vision_repair_reason = _extract_pipeline_value(result, "vision_pipeline_result", "repair_reason", "")

    return {
        "id": question_id,
        "question_id": question_id,
        "image_path": image_path,
        "question_type": question_type,
        "correct_answer": correct_answer,
        "predicted_answer": predicted_answer,
        "is_correct": comparison["is_correct"],
        "needs_manual_review": comparison["needs_manual_review"],
        "recommended_pipeline": result.get("recommended_pipeline", result.get("pipeline", "")),
        "ocr_answer": _extract_pipeline_value(result, "ocr_pipeline_result", "answer", ""),
        "vision_answer": _extract_pipeline_value(result, "vision_pipeline_result", "answer", ""),
        "ocr_confidence": ocr_confidence,
        "vision_confidence": vision_confidence,
        "final_confidence": final_confidence,
        "ocr_raw_response": ocr_raw_response,
        "vision_raw_response": vision_raw_response,
        "ocr_original_answer": ocr_original_answer,
        "vision_original_answer": vision_original_answer,
        "ocr_answer_repaired": ocr_answer_repaired,
        "vision_answer_repaired": vision_answer_repaired,
        "ocr_repair_reason": ocr_repair_reason,
        "vision_repair_reason": vision_repair_reason,
        "adaptive_selected_mode": result.get("adaptive_selected_mode", ""),
        "adaptive_initial_mode": result.get("adaptive_initial_mode", ""),
        "adaptive_fallback_mode": result.get("adaptive_fallback_mode", ""),
        "router_decision": result.get("router_decision", ""),
        "status": result.get("status", "failed"),
        "error": result.get("error"),
        "confidence": final_confidence,
        "ocr_text": result.get("ocr_result", {}).get("text", ""),
    }


def run_batch_evaluation(
    ground_truth_path: str = "data/ground_truth.json",
    output_csv_path: str = "outputs/results.csv",
    mode: str = "both",
) -> list[dict[str, Any]]:
    """Run batch evaluation over a ground truth dataset and save results."""
    records = load_ground_truth(ground_truth_path)
    results = [evaluate_question(record, mode=mode) for record in records]

    output_file = Path(output_csv_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

    return results
