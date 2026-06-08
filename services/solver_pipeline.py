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
