from typing import Dict
from services.ocr_service import ocr_question
from services.llm_service import solve_text_question, solve_image_question_direct


def run_ocr_llm_pipeline(image_path: str) -> Dict[str, str]:
    """Run the OCR + LLM pipeline on a question image."""
    ocr_result = ocr_question(image_path)
    solution_result = solve_text_question(ocr_result["text"])
    return {
        "pipeline": "ocr_llm",
        "image_path": image_path,
        "solution": solution_result["solution"],
    }


def run_vision_llm_pipeline(image_path: str) -> Dict[str, str]:
    """Run the direct Vision LLM pipeline on a question image."""
    result = solve_image_question_direct(image_path)
    return {
        "pipeline": "vision_llm",
        "image_path": image_path,
        "solution": result["solution"],
    }
