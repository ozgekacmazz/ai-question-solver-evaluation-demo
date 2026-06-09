from typing import Any, Dict

from services.llm_service import (
    normalize_answer,
    repair_llm_result_with_options,
    solve_image_question_direct,
    solve_text_question,
)
from services.ocr_service import ocr_question


def _build_pipeline_result(
    pipeline: str,
    image_path: str,
    ocr_result: Dict[str, Any],
    llm_result: Dict[str, Any],
    repair_source_text: str = "",
) -> Dict[str, Any]:
    repaired_llm_result = repair_llm_result_with_options(llm_result, repair_source_text)
    answer = repaired_llm_result.get("answer", "")

    return {
        "pipeline": pipeline,
        "image_path": image_path,
        "ocr_result": ocr_result,
        "llm_result": repaired_llm_result,
        "answer": answer,
        "solution": repaired_llm_result.get("solution", answer),
        "explanation": repaired_llm_result.get("explanation", ""),
        "confidence": repaired_llm_result.get("confidence", 0.0),
        "status": repaired_llm_result.get("status", "failed"),
        "error": repaired_llm_result.get("error"),
        "provider_mode": repaired_llm_result.get("provider_mode", ""),
        "original_answer": repaired_llm_result.get("original_answer", answer),
        "answer_repaired": repaired_llm_result.get("answer_repaired", False),
        "repair_reason": repaired_llm_result.get("repair_reason", ""),
    }


def _repair_pipeline_result(pipeline_result: Dict[str, Any], repair_source_text: str) -> Dict[str, Any]:
    llm_result = pipeline_result.get("llm_result", {})
    if not isinstance(llm_result, dict) or not llm_result:
        llm_result = {
            "answer": pipeline_result.get("answer", ""),
            "solution": pipeline_result.get("solution", pipeline_result.get("answer", "")),
            "explanation": pipeline_result.get("explanation", ""),
            "confidence": pipeline_result.get("confidence", 0.0),
            "raw_response": pipeline_result.get("raw_response", ""),
            "status": pipeline_result.get("status", "failed"),
            "error": pipeline_result.get("error"),
            "provider_mode": pipeline_result.get("provider_mode", ""),
        }
    else:
        llm_result = dict(llm_result)
        llm_result.setdefault("answer", pipeline_result.get("answer", ""))
        llm_result.setdefault("solution", pipeline_result.get("solution", pipeline_result.get("answer", "")))
        llm_result.setdefault("explanation", pipeline_result.get("explanation", ""))
        llm_result.setdefault("confidence", pipeline_result.get("confidence", 0.0))
        llm_result.setdefault("raw_response", pipeline_result.get("raw_response", ""))
        llm_result.setdefault("status", pipeline_result.get("status", "failed"))
        llm_result.setdefault("error", pipeline_result.get("error"))
        llm_result.setdefault("provider_mode", pipeline_result.get("provider_mode", ""))

    repaired_llm_result = repair_llm_result_with_options(llm_result, repair_source_text)
    answer = repaired_llm_result.get("answer", "")
    updated_result = dict(pipeline_result)
    updated_result["llm_result"] = repaired_llm_result
    updated_result["answer"] = answer
    updated_result["solution"] = repaired_llm_result.get("solution", answer)
    updated_result["explanation"] = repaired_llm_result.get("explanation", "")
    updated_result["confidence"] = repaired_llm_result.get("confidence", 0.0)
    updated_result["status"] = repaired_llm_result.get("status", "failed")
    updated_result["error"] = repaired_llm_result.get("error")
    updated_result["provider_mode"] = repaired_llm_result.get("provider_mode", "")
    updated_result["original_answer"] = repaired_llm_result.get("original_answer", answer)
    updated_result["answer_repaired"] = repaired_llm_result.get("answer_repaired", False)
    updated_result["repair_reason"] = repaired_llm_result.get("repair_reason", "")
    return updated_result


def run_ocr_llm_pipeline(image_path: str) -> Dict[str, Any]:
    """Run the OCR + LLM pipeline on a question image."""
    ocr_result = ocr_question(image_path)
    text = ocr_result.get("text", "")

    if ocr_result.get("status") != "success" or not text:
        llm_result = {
            "answer": "",
            "solution": "",
            "explanation": "",
            "confidence": 0.0,
            "raw_response": "",
            "status": "failed",
            "error": ocr_result.get("error", "OCR step failed."),
            "latency_ms": 0,
            "provider_mode": "",
        }
        return _build_pipeline_result("ocr_llm", image_path, ocr_result, llm_result, text)

    llm_result = solve_text_question(text)
    return _build_pipeline_result("ocr_llm", image_path, ocr_result, llm_result, text)


def solve_question_image(image_path: str, mode: str = "ocr") -> Dict[str, Any]:
    """Solve an uploaded question image using the requested pipeline mode."""
    if mode == "ocr":
        return run_ocr_llm_pipeline(image_path)

    if mode == "vision":
        return run_vision_llm_pipeline(image_path)

    if mode == "both":
        return run_both_pipelines(image_path)

    return {
        "pipeline": mode,
        "image_path": image_path,
        "ocr_result": {},
        "llm_result": {},
        "answer": "",
        "solution": "",
        "explanation": "",
        "confidence": 0.0,
        "status": "failed",
        "error": f"Unsupported mode: {mode}. Supported modes are: ocr, vision, both.",
        "provider_mode": "",
    }


def run_vision_llm_pipeline(image_path: str) -> Dict[str, Any]:
    """Run the direct Vision LLM pipeline on a question image."""
    result = solve_image_question_direct(image_path)
    return _build_pipeline_result("vision_llm", image_path, {}, result, "")


def _pipeline_succeeded(result: Dict[str, Any]) -> bool:
    return result.get("status") == "success"


def _get_llm_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    llm_result = result.get("llm_result")
    if isinstance(llm_result, dict):
        return llm_result
    return result


def _get_answer(result: Dict[str, Any]) -> str:
    payload = _get_llm_payload(result)
    answer = result.get("answer") if isinstance(result, dict) and result.get("answer") not in (None, "") else None
    if answer is None:
        answer = payload.get("answer", "") if isinstance(payload, dict) else ""
    return normalize_answer(str(answer))


def _get_confidence(result: Dict[str, Any]) -> float:
    payload = _get_llm_payload(result)
    confidence = result.get("confidence") if isinstance(result, dict) and result.get("confidence") is not None else None
    if confidence is None:
        confidence = payload.get("confidence", 0.0) if isinstance(payload, dict) else 0.0
    try:
        return float(confidence or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _get_status(result: Dict[str, Any]) -> str:
    payload = _get_llm_payload(result)
    status = result.get("status") if isinstance(result, dict) and result.get("status") is not None else None
    if status is None:
        status = payload.get("status", "failed") if isinstance(payload, dict) else "failed"
    return str(status or "failed")


def _get_error(result: Dict[str, Any]) -> str | None:
    payload = _get_llm_payload(result)
    error = result.get("error") if isinstance(result, dict) and result.get("error") is not None else None
    if error is None and isinstance(payload, dict):
        error = payload.get("error")
    return str(error) if error is not None else None


def _is_reliable_result(result: Dict[str, Any]) -> bool:
    answer = _get_answer(result)
    confidence = _get_confidence(result)
    return _get_status(result) == "success" and answer in {"A", "B", "C", "D", "E"} and confidence > 0


def _detect_visual_question_type(ocr_text: str, image_path: str) -> bool:
    combined = f"{ocr_text or ''} {image_path or ''}".lower()
    visual_hints = {
        "chart",
        "graph",
        "geometry",
        "table",
        "mixed",
        "visual",
        "rectangle",
        "triangle",
        "angle",
        "x-intercepts",
        "x-intercept",
        "x intercepts",
        "x intercept",
        "bar chart",
        "product price",
        "fruit values",
        "parabola",
        "coordinate",
    }
    return any(hint in combined for hint in visual_hints)


def _is_suspicious_ocr_text(ocr_text: str, image_path: str = "") -> bool:
    text = ocr_text or ""
    stripped = text.strip()
    lower_text = stripped.lower()

    if not stripped:
        return True

    suspicious_fragments = ["| |||", "|||", " jl", "jl ", "ds", "x42"]
    if any(fragment in lower_text for fragment in suspicious_fragments):
        return True

    if _detect_visual_question_type(stripped, image_path) and len(stripped) < 80:
        return True

    if "table" in lower_text and not ("notebook" in lower_text and "20" in lower_text):
        return True

    return False


def _comparison_summary(
    ocr_pipeline_result: Dict[str, Any],
    vision_pipeline_result: Dict[str, Any],
    recommended_pipeline: str,
) -> str:
    ocr_reliable = _is_reliable_result(ocr_pipeline_result)
    vision_reliable = _is_reliable_result(vision_pipeline_result)
    ocr_answer = _get_answer(ocr_pipeline_result)
    vision_answer = _get_answer(vision_pipeline_result)

    if not ocr_reliable and not vision_reliable:
        return "Neither pipeline produced a reliable answer."
    if ocr_reliable and not vision_reliable:
        return "OCR + LLM produced the only reliable answer."
    if vision_reliable and not ocr_reliable:
        return "Vision LLM produced the only reliable answer."
    if ocr_answer == vision_answer:
        return f"Both pipelines succeeded and agreed on answer {ocr_answer}."
    return (
        "Both pipelines produced reliable but different answers; "
        f"{recommended_pipeline} was recommended based on higher confidence."
    )


def _is_high_trust_repair(result: Dict[str, Any]) -> bool:
    return bool(result.get("answer_repaired")) and _get_confidence(result) >= 0.8


def _was_answer_repaired(result: Dict[str, Any]) -> bool:
    return bool(result.get("answer_repaired"))


def _get_selected_field(result: Dict[str, Any], field: str, default: Any = "") -> Any:
    if not isinstance(result, dict):
        return default
    value = result.get(field)
    if value not in (None, ""):
        return value
    payload = _get_llm_payload(result)
    if isinstance(payload, dict):
        return payload.get(field, default)
    return default


def run_both_pipelines(image_path: str) -> Dict[str, Any]:
    """Run OCR + LLM and direct Vision LLM, then recommend the best result."""
    ocr_pipeline_result = run_ocr_llm_pipeline(image_path)
    ocr_text = ocr_pipeline_result.get("ocr_result", {}).get("text", "")
    vision_pipeline_result = _repair_pipeline_result(run_vision_llm_pipeline(image_path), ocr_text)

    ocr_reliable = _is_reliable_result(ocr_pipeline_result)
    vision_reliable = _is_reliable_result(vision_pipeline_result)
    recommended_pipeline = "none"
    selected_result: Dict[str, Any] | None = None
    ocr_answer = _get_answer(ocr_pipeline_result)
    vision_answer = _get_answer(vision_pipeline_result)
    ocr_confidence = _get_confidence(ocr_pipeline_result)
    vision_confidence = _get_confidence(vision_pipeline_result)
    visual_question = _detect_visual_question_type(ocr_text, image_path)

    if ocr_reliable and vision_reliable:
        if ocr_answer == vision_answer:
            recommended_pipeline = "both_agree"
            selected_result = ocr_pipeline_result
        else:
            ocr_repaired = _was_answer_repaired(ocr_pipeline_result)
            vision_repaired = _was_answer_repaired(vision_pipeline_result)
            if visual_question and vision_confidence >= 0.8:
                recommended_pipeline = "vision_llm"
                selected_result = vision_pipeline_result
            elif ocr_repaired and not vision_repaired:
                recommended_pipeline = "ocr_llm"
                selected_result = ocr_pipeline_result
            elif vision_repaired and not ocr_repaired:
                recommended_pipeline = "vision_llm"
                selected_result = vision_pipeline_result
            elif vision_confidence > ocr_confidence:
                recommended_pipeline = "vision_llm"
                selected_result = vision_pipeline_result
            else:
                recommended_pipeline = "ocr_llm"
                selected_result = ocr_pipeline_result
    elif ocr_reliable:
        recommended_pipeline = "ocr_llm"
        selected_result = ocr_pipeline_result
    elif vision_reliable:
        recommended_pipeline = "vision_llm"
        selected_result = vision_pipeline_result

    comparison_summary = _comparison_summary(
        ocr_pipeline_result,
        vision_pipeline_result,
        recommended_pipeline,
    )

    if selected_result is None:
        answer = "unknown"
        solution = "unknown"
        explanation = ""
        confidence = 0.0
        status = "success_with_no_reliable_answer"
        error = "No reliable answer produced by either pipeline."
        provider_mode = ""
    else:
        answer = ocr_answer if recommended_pipeline == "both_agree" else _get_answer(selected_result)
        solution = _get_selected_field(selected_result, "solution", answer)
        explanation = _get_selected_field(selected_result, "explanation", "")
        confidence = _get_confidence(selected_result)
        status = _get_status(selected_result)
        error = _get_error(selected_result)
        provider_mode = _get_selected_field(selected_result, "provider_mode", "")

    ocr_provider_mode = ocr_pipeline_result.get("provider_mode", "")
    vision_provider_mode = vision_pipeline_result.get("provider_mode", "")

    return {
        "pipeline": "both",
        "image_path": image_path,
        "ocr_result": ocr_pipeline_result.get("ocr_result", {}),
        "llm_result": selected_result.get("llm_result", {}) if selected_result else {},
        "ocr_pipeline_result": ocr_pipeline_result,
        "vision_pipeline_result": vision_pipeline_result,
        "recommended_pipeline": recommended_pipeline,
        "comparison_summary": comparison_summary,
        "answer": answer,
        "solution": solution,
        "explanation": explanation,
        "confidence": confidence,
        "status": status,
        "error": error,
        "provider_mode": provider_mode,
        "ocr_provider_mode": ocr_provider_mode,
        "vision_provider_mode": vision_provider_mode,
        "ocr_original_answer": ocr_pipeline_result.get("original_answer", ocr_pipeline_result.get("answer", "")),
        "vision_original_answer": vision_pipeline_result.get("original_answer", vision_pipeline_result.get("answer", "")),
        "ocr_answer_repaired": ocr_pipeline_result.get("answer_repaired", False),
        "vision_answer_repaired": vision_pipeline_result.get("answer_repaired", False),
        "ocr_repair_reason": ocr_pipeline_result.get("repair_reason", ""),
        "vision_repair_reason": vision_pipeline_result.get("repair_reason", ""),
    }
