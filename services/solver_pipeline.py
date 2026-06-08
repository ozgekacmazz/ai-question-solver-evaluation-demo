from typing import Any, Dict
from services.ocr_service import ocr_question
from services.llm_service import solve_text_question, solve_image_question_direct


def run_ocr_llm_pipeline(image_path: str) -> Dict[str, Any]:
    """Run the OCR + LLM pipeline on a question image."""
    ocr_result = ocr_question(image_path)
    text = ocr_result.get("text", "")
    llm_result: Dict[str, Any]

    if ocr_result.get("status") != "success" or not text:
        llm_result = {
            "answer": "",
            "explanation": "",
            "confidence": 0.0,
            "raw_response": "",
            "status": "failed",
            "error": ocr_result.get("error", "OCR step failed."),
            "latency_ms": 0,
        }
        return {
            "pipeline": "ocr_llm",
            "image_path": image_path,
            "ocr_result": ocr_result,
            "llm_result": llm_result,
            "answer": llm_result.get("answer", ""),
            "solution": llm_result.get("answer", ""),
            "explanation": llm_result.get("explanation", ""),
            "confidence": llm_result.get("confidence", 0.0),
            "status": llm_result.get("status", "failed"),
            "error": llm_result.get("error"),
        }

    llm_result = solve_text_question(text)
    answer = llm_result.get("answer", "")
    return {
        "pipeline": "ocr_llm",
        "image_path": image_path,
        "ocr_result": ocr_result,
        "llm_result": llm_result,
        "answer": answer,
        "solution": answer,
        "explanation": llm_result.get("explanation", ""),
        "confidence": llm_result.get("confidence", 0.0),
        "status": llm_result.get("status", "failed"),
        "error": llm_result.get("error"),
    }


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
    }


def run_vision_llm_pipeline(image_path: str) -> Dict[str, Any]:
    """Run the direct Vision LLM pipeline on a question image."""
    result = solve_image_question_direct(image_path)
    answer = result.get("answer", "")
    return {
        "pipeline": "vision_llm",
        "image_path": image_path,
        "ocr_result": {},
        "llm_result": result,
        "answer": answer,
        "solution": answer,
        "explanation": result.get("explanation", ""),
        "confidence": result.get("confidence", 0.0),
        "status": result.get("status", "failed"),
        "error": result.get("error"),
    }


def _pipeline_succeeded(result: Dict[str, Any]) -> bool:
    return result.get("status") == "success"


def _is_reliable_result(result: Dict[str, Any]) -> bool:
    answer = str(result.get("answer", "") or "").strip().lower()
    confidence = float(result.get("confidence", 0.0) or 0.0)
    return result.get("status") == "success" and bool(answer) and answer != "unknown" and confidence > 0


def _comparison_summary(
    ocr_pipeline_result: Dict[str, Any],
    vision_pipeline_result: Dict[str, Any],
    recommended_pipeline: str,
) -> str:
    ocr_reliable = _is_reliable_result(ocr_pipeline_result)
    vision_reliable = _is_reliable_result(vision_pipeline_result)
    ocr_answer = ocr_pipeline_result.get("answer", "")
    vision_answer = vision_pipeline_result.get("answer", "")

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


def run_both_pipelines(image_path: str) -> Dict[str, Any]:
    """Run OCR + LLM and direct Vision LLM, then recommend the best result."""
    ocr_pipeline_result = run_ocr_llm_pipeline(image_path)
    vision_pipeline_result = run_vision_llm_pipeline(image_path)

    ocr_reliable = _is_reliable_result(ocr_pipeline_result)
    vision_reliable = _is_reliable_result(vision_pipeline_result)
    recommended_pipeline = "none"
    selected_result: Dict[str, Any] | None = None

    if ocr_reliable and vision_reliable:
        if ocr_pipeline_result.get("answer") == vision_pipeline_result.get("answer"):
            recommended_pipeline = "both_agree"
            selected_result = ocr_pipeline_result
        else:
            ocr_confidence = float(ocr_pipeline_result.get("confidence", 0.0) or 0.0)
            vision_confidence = float(vision_pipeline_result.get("confidence", 0.0) or 0.0)
            if vision_confidence > ocr_confidence:
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
        explanation = ""
        confidence = 0.0
        status = "success_with_no_reliable_answer"
        error = "No reliable answer produced by either pipeline."
    else:
        answer = selected_result.get("answer", "")
        explanation = selected_result.get("explanation", "")
        confidence = selected_result.get("confidence", 0.0)
        status = selected_result.get("status", "failed")
        error = selected_result.get("error")

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
        "solution": answer,
        "explanation": explanation,
        "confidence": confidence,
        "status": status,
        "error": error,
    }
