import base64
import mimetypes
from pathlib import Path
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


def _normalize_mock_text(question_text: str) -> str:
    return " ".join(question_text.lower().replace("\n", " ").split())


def _build_messages(question_text: str) -> list[dict[str, str]]:
    """Build messages for the LLM completion API."""
    return [
        {
            "role": "user",
            "content": f"Solve the following multiple-choice question:\n{question_text}\n\nProvide your response in this format:\nAnswer: [A/B/C/D/E]\nExplanation: [your explanation]\nConfidence: [0.0-1.0]",
        }
    ]


def _encode_image_to_base64(image_path: str) -> str:
    """Read an image from disk and encode it for multimodal LLM requests."""
    return base64.b64encode(Path(image_path).read_bytes()).decode("ascii")


def _guess_mime_type(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    return mime_type or "image/png"


def _build_vision_messages(image_path: str) -> list[dict]:
    encoded_image = _encode_image_to_base64(image_path)
    mime_type = _guess_mime_type(image_path)
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Solve this multiple-choice question image. Return exactly this format: Answer: [A/B/C/D/E], Explanation: ..., Confidence: [0.0-1.0]",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{encoded_image}",
                    },
                },
            ],
        }
    ]


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
    explanation = ""
    confidence_score = 0.0
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

        if lower_line.startswith("confidence:"):
            conf_candidate = stripped.split(":", 1)[1].strip()
            try:
                confidence_score = float(conf_candidate)
            except (ValueError, TypeError):
                pass
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
        "confidence": confidence_score,
        "raw_response": text,
    }


def _mock_solve(question_text: str) -> Dict[str, Any]:
    normalized = _normalize_mock_text(question_text)
    compact = normalized.replace(" ", "")
    rules = [
        (
            _is_simple_two_plus_two(question_text),
            "2 + 2 equals 4, so the correct option is B.",
            0.95,
        ),
        (
            "12/3+2" in compact,
            "12 / 3 + 2 equals 6, so the correct option is B.",
            0.92,
        ),
        (
            "2x+3=11" in compact,
            "Solving 2x + 3 = 11 gives x = 4, so the correct option is B.",
            0.92,
        ),
        (
            "notebook" in normalized and "20" in normalized and "cost" in normalized,
            "The text indicates that Notebook costs 20, so the correct option is B.",
            0.90,
        ),
        (
            "width" in normalized and "4" in normalized and "height" in normalized and "3" in normalized,
            "A rectangle with width 4 and height 3 has area 12, so the correct option is B.",
            0.90,
        ),
        (
            "4 stars" in normalized and "2" in normalized and ("total" in normalized or "more" in normalized),
            "4 stars plus 2 more stars gives 6 total, so the correct option is B.",
            0.90,
        ),
    ]

    for matched, explanation, confidence in rules:
        if matched:
            raw_response = f"Answer: B\nExplanation: {explanation}\nConfidence: {confidence:.2f}"
            return {
                "answer": "B",
                "explanation": explanation,
                "confidence": confidence,
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


def _mock_vision_success(explanation: str, confidence: float) -> Dict[str, Any]:
    raw_response = f"Answer: B\nExplanation: {explanation}\nConfidence: {confidence:.2f}"
    return {
        "answer": "B",
        "explanation": explanation,
        "confidence": confidence,
        "raw_response": raw_response,
        "status": "success",
        "error": None,
    }


def _mock_solve_image(image_path: str) -> Dict[str, Any]:
    image_name = Path(image_path).name.lower()
    known_samples = {
        "q01_text.png": (
            "Mock vision mode identified the simple 2 + 2 question and selected option B.",
            0.90,
        ),
        "q02_math.png": (
            "Mock vision mode solved 12 / 3 + 2 = 6 and selected option B.",
            0.90,
        ),
        "q03_equation.png": (
            "Mock vision mode solved 2x + 3 = 11, so x = 4, and selected option B.",
            0.91,
        ),
        "q04_table.png": (
            "Mock vision mode read the table and identified that Notebook costs 20, so the correct option is B.",
            0.92,
        ),
        "q05_chart.png": (
            "Mock vision mode interpreted the chart and identified Bananas as the highest value, so the correct option is B.",
            0.92,
        ),
        "q06_geometry.png": (
            "Mock vision mode calculated the rectangle area as 4 x 3 = 12, so the correct option is B.",
            0.93,
        ),
        "q07_mixed.png": (
            "Mock vision mode counted 4 stars plus 2 stars, giving 6 in total, so the correct option is B.",
            0.90,
        ),
        "q08_noisy.png": (
            "Mock vision mode handled the noisy image and identified the 2 + 2 question, so the correct option is B.",
            0.88,
        ),
    }

    if image_name in known_samples:
        explanation, confidence = known_samples[image_name]
        return _mock_vision_success(explanation, confidence)

    for sample_name, (explanation, confidence) in known_samples.items():
        if sample_name.removesuffix(".png") in image_name:
            return _mock_vision_success(explanation, confidence)

    return {
        "answer": "unknown",
        "explanation": "Mock vision mode cannot solve this image reliably.",
        "confidence": 0.0,
        "raw_response": "",
        "status": "success",
        "error": None,
    }


def _real_llm_solve(question_text: str) -> Dict[str, Any]:
    if not settings.llm_model_name:
        return {
            "answer": "",
            "explanation": "Model name is not configured.",
            "confidence": 0.0,
            "raw_response": "",
            "status": "failed",
            "error": "Model name is required for real LLM mode. Set LLM_MODEL_NAME.",
        }

    if completion is None:
        return {
            "answer": "",
            "explanation": "LiteLLM is not installed.",
            "confidence": 0.0,
            "raw_response": "",
            "status": "failed",
            "error": "litellm package is not installed. Install it with: pip install litellm",
        }

    messages = _build_messages(question_text)
    completion_kwargs = {
        "model": settings.llm_model_name,
        "messages": messages,
    }

    if settings.llm_api_key:
        completion_kwargs["api_key"] = settings.llm_api_key

    if settings.llm_api_base:
        completion_kwargs["base_url"] = settings.llm_api_base

    if settings.litellm_proxy_url:
        completion_kwargs["api_base"] = settings.litellm_proxy_url

    try:
        response = completion(**completion_kwargs)
        raw_text = _extract_raw_text(response)
        if hasattr(response, "choices") and response.choices:
            raw_text = response.choices[0].message.content
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


def _real_vision_solve(image_path: str) -> Dict[str, Any]:
    if not settings.llm_model_name:
        return {
            "answer": "",
            "explanation": "Model name is not configured.",
            "confidence": 0.0,
            "raw_response": "",
            "status": "failed",
            "error": "Model name is required for real vision LLM mode. Set LLM_MODEL_NAME.",
        }

    if completion is None:
        return {
            "answer": "",
            "explanation": "LiteLLM is not installed.",
            "confidence": 0.0,
            "raw_response": "",
            "status": "failed",
            "error": "litellm package is not installed. Install it with: pip install litellm",
        }

    completion_kwargs = {
        "model": settings.llm_model_name,
        "messages": _build_vision_messages(image_path),
    }

    if settings.llm_api_key:
        completion_kwargs["api_key"] = settings.llm_api_key

    if settings.llm_api_base:
        completion_kwargs["base_url"] = settings.llm_api_base

    if settings.litellm_proxy_url:
        completion_kwargs["api_base"] = settings.litellm_proxy_url

    try:
        response = completion(**completion_kwargs)
        raw_text = _extract_raw_text(response)
        if hasattr(response, "choices") and response.choices:
            raw_text = response.choices[0].message.content
        parsed = parse_llm_response(raw_text)
        parsed["status"] = "success"
        parsed["error"] = None
        return parsed
    except Exception as exc:  # pragma: no cover
        return {
            "answer": "",
            "explanation": "Failed to request the real vision LLM.",
            "confidence": 0.0,
            "raw_response": "",
            "status": "failed",
            "error": f"Real vision LLM request failed: {exc}",
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
    """Solve an image question using mock mode or a real vision LLM backend."""
    start = time.perf_counter()
    if not image_path:
        duration = int((time.perf_counter() - start) * 1000)
        return {
            "answer": "",
            "explanation": "Image path is required.",
            "confidence": 0.0,
            "raw_response": "",
            "status": "failed",
            "error": "Image path is empty.",
            "latency_ms": duration,
        }

    if not Path(image_path).exists():
        duration = int((time.perf_counter() - start) * 1000)
        return {
            "answer": "",
            "explanation": "Image file does not exist.",
            "confidence": 0.0,
            "raw_response": "",
            "status": "failed",
            "error": f"Image file not found: {image_path}",
            "latency_ms": duration,
        }

    if settings.llm_mock_mode:
        result = _mock_solve_image(image_path)
    else:
        result = _real_vision_solve(image_path)

    result["latency_ms"] = int((time.perf_counter() - start) * 1000)
    return result


def solve_image_question_direct(image_path: str) -> Dict[str, Any]:
    return solve_image_question(image_path)
