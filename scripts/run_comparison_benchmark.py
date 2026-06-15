import argparse
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from scripts.create_comparison_questions import METADATA_PATH, create_all_comparison_questions
from services.evaluator import evaluate_question, load_ground_truth


DEFAULT_OUTPUT_CSV = Path("reports") / "comparison_tool_results.csv"


def _detect_ocr_issue(result: dict[str, Any]) -> str:
    ocr_text = str(result.get("ocr_text", "") or "").strip()
    error = str(result.get("error", "") or "").lower()

    if not ocr_text:
        return "empty_ocr_text"
    if "ocr" in error and result.get("status") == "failed":
        return "ocr_failed"
    if len(ocr_text) < 30:
        return "possibly_incomplete_ocr"
    return ""


def _detect_error_type(result: dict[str, Any]) -> str:
    if result.get("is_correct"):
        return ""
    ocr_issue = _detect_ocr_issue(result)
    if ocr_issue:
        return "ocr_issue"
    if result.get("status") == "failed":
        return "pipeline_error"
    predicted = str(result.get("predicted_answer", "") or "").strip()
    if not predicted or predicted.lower() == "unknown":
        return "no_answer"
    return "solution_or_model_error"


def _build_output_row(record: dict[str, Any], result: dict[str, Any], mode: str) -> dict[str, Any]:
    ocr_issue = _detect_ocr_issue(result)
    return {
        "question_id": record.get("question_id", record.get("id", "")),
        "subject": record.get("subject", ""),
        "topic": record.get("topic", ""),
        "difficulty": record.get("difficulty", ""),
        "correct_answer": record.get("correct_answer", ""),
        "final_answer": record.get("final_answer", ""),
        "tool_answer": result.get("predicted_answer", ""),
        "tool_correct": result.get("is_correct", False),
        "tool_confidence": result.get("final_confidence", result.get("confidence", 0.0)),
        "recommended_pipeline": result.get("recommended_pipeline", ""),
        "evaluation_mode": mode,
        "ocr_text": result.get("ocr_text", ""),
        "ocr_issue": ocr_issue,
        "error_type": _detect_error_type(result),
        "status": result.get("status", ""),
        "error": result.get("error", ""),
        "needs_manual_review": result.get("needs_manual_review", True),
        "image_path": record.get("image_path", ""),
        "expected_solution": record.get("expected_solution", record.get("expected_reasoning", "")),
    }


def run_comparison_benchmark(
    metadata_path: Path = METADATA_PATH,
    output_csv_path: Path = DEFAULT_OUTPUT_CSV,
    mode: str = "both",
    mock: bool = False,
    regenerate: bool = True,
) -> list[dict[str, Any]]:
    if mock:
        settings.llm_mock_mode = True

    if regenerate or not metadata_path.exists():
        create_all_comparison_questions(metadata_path.parent)

    records = load_ground_truth(str(metadata_path))
    rows: list[dict[str, Any]] = []

    for record in records:
        result = evaluate_question(record, mode=mode)
        rows.append(_build_output_row(record, result, mode))

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv_path, index=False)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the model-comparison benchmark with the local tool.")
    parser.add_argument(
        "--mode",
        choices=["ocr", "vision", "both", "adaptive", "ocr_langflow"],
        default="both",
        help="Pipeline mode for the local tool. Default: both.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_CSV),
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT_CSV}",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force local/mock LLM mode so the script can run without active model API credentials.",
    )
    parser.add_argument(
        "--no-regenerate",
        action="store_true",
        help="Use existing PNG and metadata files without regenerating them first.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rows = run_comparison_benchmark(
        output_csv_path=Path(args.output),
        mode=args.mode,
        mock=args.mock,
        regenerate=not args.no_regenerate,
    )
    total = len(rows)
    correct = sum(1 for row in rows if row.get("tool_correct"))
    accuracy = (correct / total * 100) if total else 0.0

    print(f"Comparison benchmark complete: {total} questions")
    print(f"Mode: {args.mode}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Results saved to {args.output}")
