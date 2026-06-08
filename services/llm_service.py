import time
from typing import Any, Dict

from app.config import settings

try:
    from litellm import completion
except ImportError:  # pragma: no cover
    completion = None


def _is_simple_two_plus_two(question_text: str) -> bool:
    normalized = question_text.lower().strip()
    return "2 + 2" in normalized and "what" in normalized


def _build_prompt(question_text: str) -> str:
    return f"Solve the following multiple-choice question:\n{question_text}\n\nProvide the answer and a short explanation."


def _extract_raw_text(response: Any) -> str:
    if response is None:
        return ""
    if hasattr(response, "text"):
        return str(response.text)
    return str(response)


def parse_llm_response(raw_response: str) -> Dict[str, Any]:
    """Parse a raw LLM response into a simple structured form."""
    text = str(raw_response or "").strip()
    answer = "unknown"
    explanation_lines = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        lower_line = stripped.lower()
        if lower_line.startswith("answer:"):
            answer_candidate = stripped.split(":", 1)[1].strip()
            if answer_candidate:
                answer = answer_candidate
            continue

        if lower_line.startswith("explanation:"):
            explanation_candidate = stripped.split(":", 1)[1].strip()
            if explanation_candidate:
                explanation_lines.append(explanation_candidate)
            continue

        explanation_lines.append(stripped)

    explanation = " ".join(explanation_lines).strip() or text
    return {
        "answer": answer,
        "explanation": explanation,
        "confidence": 0.0,
        "raw_response": text,
    }


def _mock_solve(question_text: str) -> Dict[str, Any]:
    if _is_simple_two_plus_two(question_text):
        raw_response = "Answer: B\nExplanation: 2 + 2 equals 4, so the correct option is B."
        return {
            "answer": "B",
            "explanation": "2 + 2 equals 4, so the correct option is B.",
            "confidence": 0.95,
            "raw_response": raw_response,
            "status": "success",
            "error": None,
        }

    return {
        "answer": "unknown",
        "explanation": "Mock mode cannot solve this question reliably.",
        "confidence": 0.0,
        "raw_response": question_text,
        "status": "success",
        "error": None,
    }


def _real_llm_solve(question_text: str) -> Dict[str, Any]:
    if not settings.llm_model_name:
        return {
            "answer": "",
            "explanation": "LLM model name is not configured.",
            "confidence": 0.0,
            "raw_response": "",
            "status": "failed",
            "error": "LLM_MODEL_NAME is required for real LLM mode.",
        }

    if completion is None or not hasattr(completion, "create"):
        return {
            "answer": "",
            "explanation": "LiteLLM integration is not available.",
            "confidence": 0.0,
            "raw_response": "",
            "status": "failed",
            "error": "litellm package is not installed or completion API is unavailable.",
        }

    prompt = _build_prompt(question_text)
    try:
        response = completion.create(
            model=settings.llm_model_name,
            prompt=prompt,
            api_key=settings.llm_api_key,
            base_url=settings.llm_api_base,
        )
        raw_text = _extract_raw_text(response)
        parsed = parse_llm_response(raw_text)
        parsed["status"] = "success"
        parsed["error"] = None
        return parsed
    except Exception as exc:  # pragma: no cover
        return {
            "answer": "",
            "explanation": "Failed to request the real LLM.",
            "confidence": 0.0,
            "raw_response": "",
            "status": "failed",
            "error": f"Real LLM request failed: {exc}",
        }


def solve_text_question(question_text: str) -> Dict[str, Any]:
    """Solve a text question using mock mode or a real LLM backend."""
    start = time.perf_counter()
    if not question_text or not question_text.strip():
        duration = int((time.perf_counter() - start) * 1000)
        return {
            "answer": "",
            "explanation": "Question text is required.",
            "confidence": 0.0,
            "raw_response": "",
            "status": "failed",
            "error": "Question text is empty.",
            "latency_ms": duration,
        }

    if settings.llm_mock_mode:
        result = _mock_solve(question_text)
    else:
        result = _real_llm_solve(question_text)

    result["latency_ms"] = int((time.perf_counter() - start) * 1000)
    return result


def solve_image_question(image_path: str) -> Dict[str, Any]:
    """Placeholder for the vision question solving pipeline."""
    start = time.perf_counter()
    result = {
        "answer": "",
        "explanation": "Vision LLM pipeline is not implemented yet.",
        "confidence": 0.0,
        "raw_response": "",
        "status": "failed",
        "error": "Vision LLM pipeline is not implemented yet.",
        "latency_ms": int((time.perf_counter() - start) * 1000),
    }
    return result


def solve_image_question_direct(image_path: str) -> Dict[str, Any]:
    return solve_image_question(image_path)
