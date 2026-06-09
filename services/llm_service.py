import base64
import json
import mimetypes
from pathlib import Path
import re
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
            "content": (
                "Solve the multiple-choice question.\n"
                "Return ONLY valid JSON. Do not include Markdown. Do not include extra text.\n"
                "The JSON schema must be:\n"
                '{\n  "answer": "A|B|C|D|E",\n  "explanation": "short explanation",\n  "confidence": 0.0\n}\n'
                "The answer must be a single uppercase letter only. Confidence must be a number between 0 and 1.\n"
                "If the question is unclear, return:\n"
                '{\n  "answer": "unknown",\n  "explanation": "The question could not be solved reliably.",\n  "confidence": 0.0\n}\n\n'
                f"Question:\n{question_text}"
            ),
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
                    "text": (
                        "Solve the multiple-choice question in this image.\n"
                        "Return ONLY valid JSON. Do not include Markdown. Do not include extra text.\n"
                        "The JSON schema must be:\n"
                        '{\n  "answer": "A|B|C|D|E",\n  "explanation": "short explanation",\n  "confidence": 0.0\n}\n'
                        "The answer must be a single uppercase letter only. Confidence must be a number between 0 and 1.\n"
                        "If the question is unclear, return:\n"
                        '{\n  "answer": "unknown",\n  "explanation": "The question could not be solved reliably.",\n  "confidence": 0.0\n}'
                    ),
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


def _current_provider_mode() -> str:
    return "mock" if settings.llm_mock_mode else "real"


def _sanitize_error_message(message: str) -> str:
    sanitized = str(message)
    if settings.llm_api_key:
        sanitized = sanitized.replace(settings.llm_api_key, "[REDACTED_API_KEY]")
    return sanitized


def _real_failure_result(error: str) -> Dict[str, Any]:
    return {
        "answer": "unknown",
        "solution": "",
        "explanation": "",
        "confidence": 0.0,
        "raw_response": "",
        "status": "failed",
        "error": _sanitize_error_message(error),
        "provider_mode": "real",
    }


def normalize_answer(value: str) -> str:
    """Normalize model answer text to A-E or unknown."""
    if not isinstance(value, str):
        return "unknown"

    text = value.strip()
    if not text:
        return "unknown"

    lowered = text.lower().strip(" .:;!?)(")
    if lowered == "unknown":
        return "unknown"

    direct = text.strip().upper().strip(" .:;!?)(")
    if direct in {"A", "B", "C", "D", "E"}:
        return direct

    patterns = [
        r"\b(?:option|choice|answer)\s*[:#-]?\s*([A-E])\b",
        r"\bcorrect\s+answer\s+(?:is|:)?\s*([A-E])\b",
        r"\bfinal\s+answer\s+(?:is|:)?\s*([A-E])\b",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    return "unknown"


def _normalize_for_comparison(text: str) -> str:
    if not isinstance(text, str):
        return ""

    normalized = text.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip(" \t\r\n\"'`.,:;!?")


def normalize_option_value(value: str) -> str:
    """Normalize OCR/model math text so option values can be compared safely."""
    if not isinstance(value, str):
        return ""

    normalized = value.strip().lower()
    normalized = normalized.replace("\u00b2", "^2")
    normalized = normalized.replace("²", "^2")
    normalized = normalized.replace("×", "*").replace("⋅", "*").replace("·", "*")
    normalized = re.sub(r"\bx\s*\*\s*2\s*4\b", "x^2", normalized)
    normalized = re.sub(r"\bx\s*\*\s*2\b", "x^2", normalized)
    normalized = re.sub(r"\bx\s*4\s*2\b", "x^2", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s*([+\-*/=^()])\s*", r" \1 ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.strip(" \t\r\n\"'`.,:;!?")
    normalized = re.sub(r"\bx\s*\^\s*2\b", "x^2", normalized)
    return normalized


def _canonicalize_option_value(text: str) -> str:
    normalized = normalize_option_value(text)
    return re.sub(r"\s+", "", normalized)


def _is_numeric_like(text: str) -> bool:
    return bool(re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text or ""))


def _clean_extracted_option_value(value: str) -> str:
    if not isinstance(value, str):
        return ""

    cleaned = re.sub(r"\s+", " ", value).strip(" \t\r\n;|")
    visual_tail_match = re.match(
        r"^([+-]?\d+(?:\.\d+)?)\s+(?:height|width|length|area|base|diagram|figure|label|labels|rectangle|triangle)\b",
        cleaned,
        flags=re.IGNORECASE,
    )
    if visual_tail_match:
        return visual_tail_match.group(1)
    return cleaned


def extract_options_from_text(text: str) -> Dict[str, str]:
    """Extract multiple-choice option labels and values from OCR text."""
    if not isinstance(text, str) or not text.strip():
        return {}

    option_pattern = re.compile(
        r"(?<![A-Za-z0-9+^*/])([A-E])(?:[ \t]*\1)?[ \t]*[\)\.:\-]\s*",
        flags=re.IGNORECASE,
    )
    matches = list(option_pattern.finditer(text))
    if not matches:
        return {}

    options: Dict[str, str] = {}
    for index, match in enumerate(matches):
        label = match.group(1).upper()
        value_start = match.end()
        value_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = _clean_extracted_option_value(text[value_start:value_end])
        if value:
            options[label] = value

    return options


def infer_answer_from_explanation_and_options(explanation: str, options: Dict[str, str]) -> str:
    """Infer the answer letter from explanatory text and parsed option values."""
    if not isinstance(explanation, str) or not explanation.strip():
        return "unknown"

    normalized_options = {
        letter.upper(): _canonicalize_option_value(value)
        for letter, value in options.items()
        if isinstance(value, str) and _canonicalize_option_value(value)
    }
    if not normalized_options:
        return "unknown"

    explicit_patterns = [
        r"\b(?:correct answer|correct option|final answer|answer should|answer must be|the answer should|the answer is|answer is|corresponds to option|corresponds to|select option|choose option)\s*(?:is|:|-)?\s*\(?\s*([A-E])\s*\)?\b",
        r"\b(?:option|choice)\s*([A-E])\b",
    ]
    for pattern in explicit_patterns:
        matches = re.findall(pattern, explanation, flags=re.IGNORECASE)
        if matches:
            return str(matches[-1]).upper()

    candidate_patterns = [
        r"(?:=|equals|equal to|is|becomes|gives|yields|results in|evaluates to|simplifies to|produces)\s*([^\n.;,]+)",
        r"\b(?:final answer|answer|result|value)\s*(?:is|=|:|-)\s*([^\n.;,]+)",
    ]

    candidate_counts: Dict[str, int] = {}
    final_numeric_letters: list[str] = []
    normalized_explanation = _canonicalize_option_value(explanation)

    for pattern in candidate_patterns:
        for match in re.findall(pattern, explanation, flags=re.IGNORECASE):
            candidate = _canonicalize_option_value(str(match))
            if not candidate:
                continue
            matched_letters = [letter for letter, option_value in normalized_options.items() if option_value == candidate]
            if len(matched_letters) == 1:
                letter = matched_letters[0]
                candidate_counts[letter] = candidate_counts.get(letter, 0) + 1

            final_number_match = re.search(r"([+-]?\d+(?:\.\d+)?)\s*$", str(match).strip())
            if final_number_match:
                final_number = _canonicalize_option_value(final_number_match.group(1))
                numeric_matches = [
                    letter for letter, option_value in normalized_options.items() if option_value == final_number
                ]
                if len(numeric_matches) == 1:
                    letter = numeric_matches[0]
                    final_numeric_letters.append(letter)
                    candidate_counts[letter] = candidate_counts.get(letter, 0) + 1

    value_matches: list[tuple[str, str]] = []
    for letter, option_value in normalized_options.items():
        if option_value and not _is_numeric_like(option_value):
            pattern = rf"(?:^|[^a-z0-9]){re.escape(option_value)}(?:$|[^a-z0-9])"
            if re.search(pattern, normalized_explanation):
                value_matches.append((letter, option_value))

    if value_matches:
        longest_match_length = max(len(option_value) for _, option_value in value_matches)
        longest_matches = [
            letter for letter, option_value in value_matches if len(option_value) == longest_match_length
        ]
        if len(longest_matches) == 1:
            letter = longest_matches[0]
            candidate_counts[letter] = candidate_counts.get(letter, 0) + 1

    if not candidate_counts:
        return "unknown"

    best_score = max(candidate_counts.values())
    best_letters = [letter for letter, score in candidate_counts.items() if score == best_score]
    if len(best_letters) != 1:
        if final_numeric_letters and final_numeric_letters[-1] in best_letters:
            return final_numeric_letters[-1]
        return "unknown"
    return best_letters[0]


def repair_llm_result_with_options(result: Dict[str, Any], ocr_text: str) -> Dict[str, Any]:
    """Repair an LLM result when the explanation clearly matches one OCR option."""
    repaired = dict(result or {})
    current_answer = normalize_answer(str(repaired.get("answer", "")))
    explanation = str(repaired.get("explanation", "") or "").strip()
    raw_response = str(repaired.get("raw_response", "") or "").strip()
    source_text = "\n".join(part for part in [explanation, raw_response] if part)
    options_source = ocr_text if isinstance(ocr_text, str) and ocr_text.strip() else raw_response
    options = extract_options_from_text(options_source)

    inferred_answer = infer_answer_from_explanation_and_options(source_text, options)
    repaired["original_answer"] = current_answer
    repaired["answer_repaired"] = False
    repaired.setdefault("repair_reason", "")

    if inferred_answer in {"A", "B", "C", "D", "E"} and inferred_answer != current_answer:
        repaired["answer"] = inferred_answer
        repaired["solution"] = inferred_answer
        repaired["answer_repaired"] = True
        repaired["repair_reason"] = f"Explanation matched option {inferred_answer}"
        confidence = _clamp_confidence(repaired.get("confidence", 0.0))
        repaired["confidence"] = confidence if confidence >= 0.8 else max(confidence, 0.8)

    return repaired


def _clamp_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def _parse_json_response(text: str) -> Dict[str, Any] | None:
    candidates = [text]
    json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if json_match:
        candidates.append(json_match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        answer = normalize_answer(str(parsed.get("answer", "")))
        explanation = str(parsed.get("explanation", "") or "").strip()
        return {
            "answer": answer,
            "explanation": explanation,
            "confidence": _clamp_confidence(parsed.get("confidence", 0.0)),
            "raw_response": text,
        }
    return None


def _parse_regex_response(text: str) -> Dict[str, Any]:
    answer = "unknown"
    final_patterns = [
        r"\bfinal\s+answer\s*(?:is|:|-)?\s*([A-E])\b",
        r"\bcorrect\s+answer\s*(?:is|:|-)?\s*([A-E])\b",
    ]
    secondary_patterns = [
        r"\banswer\s*(?:is|:|-)?\s*([A-E])\b",
        r"\boption\s+([A-E])\b",
        r"\bchoice\s+([A-E])\b",
    ]

    for pattern in final_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            answer = matches[-1].upper()

    if answer == "unknown":
        for pattern in secondary_patterns:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            if matches:
                answer = matches[-1].upper()

    confidence = 0.0
    confidence_match = re.search(r"\bconfidence\s*(?:is|:|-)?\s*([01](?:\.\d+)?)\b", text, flags=re.IGNORECASE)
    if confidence_match:
        confidence = _clamp_confidence(confidence_match.group(1))

    explanation = text.strip()
    explanation_match = re.search(r"\bexplanation\s*(?:is|:|-)?\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if explanation_match:
        explanation = explanation_match.group(1).strip()

    return {
        "answer": answer,
        "explanation": explanation,
        "confidence": confidence,
        "raw_response": text,
    }


def parse_llm_response(raw_response: str) -> Dict[str, Any]:
    """Parse a raw LLM response into a structured form."""
    text = str(raw_response or "").strip()
    if not text:
        return {"answer": "unknown", "explanation": "", "confidence": 0.0, "raw_response": ""}

    json_result = _parse_json_response(text)
    if json_result is not None:
        return json_result

    return _parse_regex_response(text)


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
                "solution": "B",
                "explanation": explanation,
                "confidence": confidence,
                "raw_response": raw_response,
                "status": "success",
                "error": None,
                "provider_mode": "mock",
            }

    return {
        "answer": "unknown",
        "solution": "unknown",
        "explanation": "Mock mode cannot solve this question reliably.",
        "confidence": 0.0,
        "raw_response": question_text,
        "status": "success",
        "error": None,
        "provider_mode": "mock",
    }


def _mock_vision_success(explanation: str, confidence: float) -> Dict[str, Any]:
    raw_response = f"Answer: B\nExplanation: {explanation}\nConfidence: {confidence:.2f}"
    return {
        "answer": "B",
        "solution": "B",
        "explanation": explanation,
        "confidence": confidence,
        "raw_response": raw_response,
        "status": "success",
        "error": None,
        "provider_mode": "mock",
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

    benchmark_tokens = [
        "parabola",
        "derivative",
        "limit",
        "integral",
        "angle",
        "graph",
        "reasoning",
        "mixed_math_visual",
    ]
    if image_name.startswith("q") and any(token in image_name for token in benchmark_tokens):
        return {
            "answer": "unknown",
            "solution": "unknown",
            "explanation": "Benchmark questions require real model evaluation. Mock mode does not solve this advanced question.",
            "confidence": 0.0,
            "raw_response": "",
            "status": "success",
            "error": None,
            "provider_mode": "mock",
        }

    return {
        "answer": "unknown",
        "solution": "unknown",
        "explanation": "Mock vision mode cannot solve this image reliably.",
        "confidence": 0.0,
        "raw_response": "",
        "status": "success",
        "error": None,
        "provider_mode": "mock",
    }


def _real_llm_solve(question_text: str) -> Dict[str, Any]:
    if not settings.llm_model_name:
        return _real_failure_result("Model name is required for real LLM mode. Set LLM_MODEL_NAME.")

    if completion is None:
        return _real_failure_result("litellm package is not installed. Install it with: pip install litellm")

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
        parsed["solution"] = parsed.get("answer", "")
        parsed["status"] = "success"
        parsed["error"] = None
        parsed["provider_mode"] = "real"
        return parsed
    except Exception as exc:  # pragma: no cover
        return _real_failure_result(f"Real LLM request failed: {exc}")


def _real_vision_solve(image_path: str) -> Dict[str, Any]:
    if not settings.llm_model_name:
        return _real_failure_result("Model name is required for real vision LLM mode. Set LLM_MODEL_NAME.")

    if completion is None:
        return _real_failure_result("litellm package is not installed. Install it with: pip install litellm")

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
        parsed["solution"] = parsed.get("answer", "")
        parsed["status"] = "success"
        parsed["error"] = None
        parsed["provider_mode"] = "real"
        return parsed
    except Exception as exc:  # pragma: no cover
        return _real_failure_result(f"Real vision LLM request failed: {exc}")


def solve_text_question(question_text: str) -> Dict[str, Any]:
    """Solve a text question using mock mode or a real LLM backend."""
    start = time.perf_counter()
    if not question_text or not question_text.strip():
        duration = int((time.perf_counter() - start) * 1000)
        return {
            "answer": "",
            "solution": "",
            "explanation": "Question text is required.",
            "confidence": 0.0,
            "raw_response": "",
            "status": "failed",
            "error": "Question text is empty.",
            "latency_ms": duration,
            "provider_mode": _current_provider_mode(),
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
            "solution": "",
            "explanation": "Image path is required.",
            "confidence": 0.0,
            "raw_response": "",
            "status": "failed",
            "error": "Image path is empty.",
            "latency_ms": duration,
            "provider_mode": _current_provider_mode(),
        }

    if not Path(image_path).exists():
        duration = int((time.perf_counter() - start) * 1000)
        return {
            "answer": "",
            "solution": "",
            "explanation": "Image file does not exist.",
            "confidence": 0.0,
            "raw_response": "",
            "status": "failed",
            "error": f"Image file not found: {image_path}",
            "latency_ms": duration,
            "provider_mode": _current_provider_mode(),
        }

    if settings.llm_mock_mode:
        result = _mock_solve_image(image_path)
    else:
        result = _real_vision_solve(image_path)

    result["latency_ms"] = int((time.perf_counter() - start) * 1000)
    return result


def solve_image_question_direct(image_path: str) -> Dict[str, Any]:
    return solve_image_question(image_path)
