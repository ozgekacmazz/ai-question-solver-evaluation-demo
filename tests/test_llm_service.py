from app.config import settings

from scripts.create_benchmark_questions import create_all_benchmark_questions
from scripts.create_sample_questions import create_all_sample_questions
from services.llm_service import (
    extract_options_from_text,
    infer_answer_from_explanation_and_options,
    normalize_answer,
    normalize_option_value,
    parse_llm_response,
    repair_llm_result_with_options,
    solve_image_question,
    solve_text_question,
)


def test_empty_question_returns_failed_status() -> None:
    result = solve_text_question("")

    assert result["status"] == "failed"
    assert result["error"] == "Question text is empty."
    assert result["answer"] == ""
    assert result["latency_ms"] >= 0


def test_mock_mode_two_plus_two_question_returns_B(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_mock_mode", True)
    result = solve_text_question("What is 2 + 2?")

    assert result["status"] == "success"
    assert result["answer"] == "B"
    assert result["confidence"] == 0.95
    assert result["error"] is None
    assert result["provider_mode"] == "mock"


def test_unknown_mock_question_returns_safe_result(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_mock_mode", True)
    result = solve_text_question("What is the capital of France?")

    assert result["status"] == "success"
    assert result["answer"] == "unknown"
    assert result["error"] is None
    assert result["provider_mode"] == "mock"


def test_solve_image_question_missing_image_fails_safely() -> None:
    result = solve_image_question("dummy_path.png")

    assert result["status"] == "failed"
    assert "not found" in result["error"].lower()
    assert result["latency_ms"] >= 0


def test_mock_vision_two_plus_two_sample_returns_B(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_mock_mode", True)
    result = solve_image_question("data/sample_questions/q01_text.png")

    assert result["status"] == "success"
    assert result["answer"] == "B"
    assert result["confidence"] == 0.90
    assert result["error"] is None
    assert result["provider_mode"] == "mock"


def test_mock_vision_returns_B_for_all_known_samples(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_mock_mode", True)
    create_all_sample_questions()

    expected_confidences = {
        "q01_text.png": 0.90,
        "q02_math.png": 0.90,
        "q03_equation.png": 0.91,
        "q04_table.png": 0.92,
        "q05_chart.png": 0.92,
        "q06_geometry.png": 0.93,
        "q07_mixed.png": 0.90,
        "q08_noisy.png": 0.88,
    }

    for file_name, confidence in expected_confidences.items():
        result = solve_image_question(f"data/sample_questions/{file_name}")
        assert result["status"] == "success"
        assert result["answer"] == "B"
        assert result["confidence"] == confidence
        assert result["error"] is None
        assert result["provider_mode"] == "mock"


def test_mock_vision_does_not_hardcode_benchmark_answers(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_mock_mode", True)
    create_all_benchmark_questions()

    result = solve_image_question("data/benchmark_questions/q09_parabola_vertex.png")

    assert result["status"] == "success"
    assert result["answer"] == "unknown"
    assert result["confidence"] == 0.0
    assert "real model evaluation" in result["explanation"]
    assert result["provider_mode"] == "mock"


def test_mock_text_rules_return_B_for_known_patterns(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_mock_mode", True)

    examples = [
        "What is 12 / 3 + 2?",
        "Solve for x: 2x + 3 = 11",
        "Product Price Notebook 20 Question: Which product costs 20?",
        "Rectangle width 4 height 3 Question: What is the area?",
        "4 stars and 2 more stars. How many stars are there in total?",
    ]

    for text in examples:
        result = solve_text_question(text)
        assert result["status"] == "success"
        assert result["answer"] == "B"
        assert result["confidence"] > 0


def test_mock_text_solves_chained_arithmetic_and_maps_to_option(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_mock_mode", True)

    result = solve_text_question("What is 12/3 + 2?\nA)4\nB) 6\nC)8")

    assert result["status"] == "success"
    assert result["answer"] == "B"
    assert "equals 6" in result["explanation"]


def test_mock_text_solves_basic_arithmetic_operations_from_options(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_mock_mode", True)

    examples = [
        ("What is 7 + 5?\nA) 10\nB) 11\nC) 12", "C"),
        ("What is 18 - 6?\nA) 8\nB) 10\nC) 12", "C"),
        ("What is 4 * 3?\nA) 7\nB) 12\nC) 16", "B"),
        ("What is 20 / 5?\nA) 4\nB) 5\nC) 10", "A"),
    ]

    for text, expected_answer in examples:
        result = solve_text_question(text)
        assert result["status"] == "success"
        assert result["answer"] == expected_answer


def test_mock_text_rules_solve_expanded_turkish_and_social_patterns(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_mock_mode", True)

    examples = [
        (
            "Paragrafta anlatilan kisi her sabah kitap okur. Bu kisinin en belirgin aliskanligi hangisidir?\n"
            "A) Spor yapmak\nB) Kitap okumak\nC) Resim cizmek\nD) Sarki soylemek\nE) Yolculuk yapmak",
            "B",
        ),
        (
            "'Zamanini iyi kullanan ogrenci islerini son gune birakmaz.' cumlesinde vurgulanan dusunce hangisidir?\n"
            "A) Planli olmak\nB) Hizli kosmak\nC) Sessiz kalmak\nD) Cok uyumak\nE) Oyun oynamak",
            "A",
        ),
        (
            "A region receives little rain and has very sparse plants. Which climate description best fits this region?\n"
            "A) Desert\nB) Rain forest\nC) Tundra\nD) Oceanic\nE) Mountain",
            "A",
        ),
    ]

    for text, expected_answer in examples:
        result = solve_text_question(text)
        assert result["status"] == "success"
        assert result["answer"] == expected_answer
        assert result["confidence"] > 0


def test_mock_text_rules_solve_expanded_math_patterns(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_mock_mode", True)

    examples = [
        (
            "The ratio of red beads to blue beads is 2 to 3. If there are 6 red beads, how many blue beads are there?\n"
            "A) 6\nB) 7\nC) 8\nD) 9\nE) 12",
            "D",
        ),
        (
            "Solve for x: 3x - 5 = 16\nA) 5\nB) 6\nC) 7\nD) 8\nE) 9",
            "C",
        ),
        (
            "If f(x) = x^2, what is f'(3)?\nA) 3\nB) 4\nC) 5\nD) 6\nE) 9",
            "D",
        ),
        (
            "What is the integral of 4 dx?\nA) 4x + C\nB) x^4 + C\nC) 4 + C\nD) 2x + C\nE) x + C",
            "A",
        ),
    ]

    for text, expected_answer in examples:
        result = solve_text_question(text)
        assert result["status"] == "success"
        assert result["answer"] == expected_answer
        assert result["confidence"] > 0


def test_mock_text_rules_tolerate_common_ocr_math_artifacts(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_mock_mode", True)

    examples = [
        (
            "If f(x) = 2x + 3, what is (4)?\nA)8\nB)9\nC)10\nD)11\nE)12",
            "D",
        ),
        (
            "The parabola y = (x - 1)42 + 2 has vertex at which point?\n"
            "A) (1, 2)\nB) (2, 1)\nC) (-1, 2)\nD) (1, -2)\nE) (0, 2)",
            "A",
        ),
        (
            "A cart with mass 3 kg accelerates at 2 m/s*2. What is the net force?\n"
            "A)4N\nB)SN\nC)EN\nD)8N\nE)10N",
            "C",
        ),
    ]

    for text, expected_answer in examples:
        result = solve_text_question(text)
        assert result["status"] == "success"
        assert result["answer"] == expected_answer
        assert result["confidence"] > 0


def test_mock_text_rules_solve_expanded_chart_table_and_science_patterns(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_mock_mode", True)

    examples = [
        (
            "Team Points Red 5 Blue 8 Green 6 Which team has the highest number of points?\n"
            "A) Red\nB) Blue\nC) Green\nD) Yellow\nE) All equal",
            "B",
        ),
        (
            "Score Table Name Score A 4 B 9 C 6 Which row has the score 9?\n"
            "A) Row A\nB) Row C\nC) No row\nD) Row B\nE) All rows",
            "D",
        ),
        (
            "A cart with mass 3 kg accelerates at 2 m/s^2. What is the net force?\n"
            "A) 4 N\nB) 5 N\nC) 6 N\nD) 8 N\nE) 10 N",
            "C",
        ),
    ]

    for text, expected_answer in examples:
        result = solve_text_question(text)
        assert result["status"] == "success"
        assert result["answer"] == expected_answer
        assert result["confidence"] > 0


def test_unknown_mock_question_does_not_echo_options_in_raw_response(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_mock_mode", True)

    result = solve_text_question("Unclear prompt.\nA) First\nB) Second\nC) Third\nD) Fourth\nE) Fifth")

    assert result["answer"] == "unknown"
    assert result["raw_response"] == ""


def test_parse_llm_response_json() -> None:
    result = parse_llm_response('{"answer": "B", "explanation": "6 is option B.", "confidence": 0.84}')

    assert result["answer"] == "B"
    assert result["explanation"] == "6 is option B."
    assert result["confidence"] == 0.84


def test_parse_correct_answer_phrase() -> None:
    result = parse_llm_response("The correct answer is B. It evaluates to 6.")

    assert result["answer"] == "B"


def test_parse_prefers_last_correct_answer_phrase() -> None:
    result = parse_llm_response("Initially A looked plausible, but the correct answer is B.")

    assert result["answer"] == "B"


def test_parse_does_not_select_from_option_list() -> None:
    result = parse_llm_response("A) 4 B) 6 C) 8")

    assert result["answer"] == "unknown"


def test_extract_options_from_text_parses_numeric_options() -> None:
    text = "A)4\nB) 6\nC)8"

    assert extract_options_from_text(text) == {"A": "4", "B": "6", "C": "8"}


def test_extract_options_from_text_parses_text_options() -> None:
    text = "A) Pen\nB) Notebook\nC) Bag"

    assert extract_options_from_text(text) == {"A": "Pen", "B": "Notebook", "C": "Bag"}


def test_extract_options_from_text_parses_duplicated_ocr_option_label() -> None:
    text = "A) 6\nB) 8\nCc) 10\nD) 12"

    assert extract_options_from_text(text) == {"A": "6", "B": "8", "C": "10", "D": "12"}


def test_extract_options_from_text_does_not_consume_plus_c_expression() -> None:
    text = "A) 2x + C\nB) x*24+C\nC) x*24"

    assert extract_options_from_text(text) == {"A": "2x + C", "B": "x*24+C", "C": "x*24"}


def test_extract_options_from_text_trims_visual_tail_from_numeric_option() -> None:
    text = "A)10\nB)12 height 52 width = 4\nC)14"

    assert extract_options_from_text(text)["B"] == "12"


def test_infer_answer_from_explanation_detects_correct_answer_phrase() -> None:
    options = {"A": "4", "B": "6", "C": "8"}

    assert infer_answer_from_explanation_and_options("The correct answer is B.", options) == "B"


def test_infer_answer_from_explanation_detects_parenthesized_correct_answer() -> None:
    options = {"A": "4", "B": "6", "C": "8"}

    assert infer_answer_from_explanation_and_options("The correct answer (B).", options) == "B"


def test_infer_answer_from_explanation_prefers_plus_c_integral_option() -> None:
    options = {"A": "2x + C", "B": "x42 + C", "C": "x42"}

    assert infer_answer_from_explanation_and_options("The integral is x^2 + C.", options) == "B"


def test_infer_answer_from_explanation_matches_q12_x_star_ocr_artifact() -> None:
    options = extract_options_from_text("A) 2x + C\nB) x*24+C\nC) x*24")

    assert infer_answer_from_explanation_and_options("The integral is x^2 + C.", options) == "B"


def test_infer_answer_from_explanation_maps_final_numeric_assignment_to_option() -> None:
    options = {"A": "6", "B": "8", "C": "10", "D": "12"}

    assert infer_answer_from_explanation_and_options("x = 40 / 5 = 8", options) == "B"


def test_infer_answer_from_explanation_matches_q16_duplicated_label_options() -> None:
    options = extract_options_from_text("A) 6\nB) 8\nCc) 10\nD) 12")

    assert infer_answer_from_explanation_and_options("x = 8", options) == "B"


def test_infer_answer_from_explanation_maps_function_value_to_option() -> None:
    options = {"A": "5", "B": "7", "C": "9", "D": "12"}

    assert infer_answer_from_explanation_and_options("f'(2) = 7", options) == "B"


def test_infer_answer_from_explanation_maps_limit_value_to_option() -> None:
    options = {"A": "2", "B": "4", "C": "0", "D": "undefined"}

    assert infer_answer_from_explanation_and_options("The limit becomes 4.", options) == "B"


def test_infer_answer_from_explanation_maps_text_value_to_option() -> None:
    options = {"A": "Pen", "B": "Notebook", "C": "Bag"}

    assert infer_answer_from_explanation_and_options("The answer is Notebook.", options) == "B"


def test_repair_llm_result_with_options_changes_answer_when_explanation_matches_letter() -> None:
    result = {
        "answer": "C",
        "solution": "C",
        "explanation": "The correct answer is B.",
        "confidence": 0.77,
        "raw_response": "Answer: C\nExplanation: The correct answer is B.",
        "status": "success",
        "error": None,
    }

    repaired = repair_llm_result_with_options(result, "A) 4\nB) 6\nC) 8")

    assert repaired["answer"] == "B"
    assert repaired["solution"] == "B"
    assert repaired["original_answer"] == "C"
    assert repaired["answer_repaired"] is True
    assert repaired["repair_reason"] == "Explanation matched option B"
    assert repaired["confidence"] == 0.8


def test_repair_llm_result_with_options_repairs_q02_style_parenthesized_answer() -> None:
    result = {
        "answer": "C",
        "solution": "C",
        "explanation": (
            "12 divided by 3 is 4, plus 2 equals 6, but since option B is 6, "
            "option C is 8 which is incorrect. The correct answer (B)."
        ),
        "confidence": 0.77,
        "raw_response": "",
        "status": "success",
        "error": None,
    }

    repaired = repair_llm_result_with_options(result, "A) 4\nB) 6\nC) 8")

    assert repaired["answer"] == "B"
    assert repaired["original_answer"] == "C"
    assert repaired["answer_repaired"] is True


def test_repair_llm_result_with_options_repairs_q02_final_computed_value() -> None:
    result = {
        "answer": "C",
        "solution": "C",
        "explanation": "12 divided by 3 is 4, plus 2 equals 6.",
        "confidence": 0.77,
        "raw_response": "",
        "status": "success",
        "error": None,
    }

    repaired = repair_llm_result_with_options(result, "A)4\nB)6\nC)8")

    assert repaired["answer"] == "B"
    assert repaired["original_answer"] == "C"
    assert repaired["answer_repaired"] is True


def test_repair_llm_result_with_options_repairs_q12_integral_constant_option() -> None:
    result = {
        "answer": "C",
        "solution": "C",
        "explanation": "The integral of 2x dx is x^2 + C.",
        "confidence": 0.82,
        "raw_response": "",
        "status": "success",
        "error": None,
    }

    repaired = repair_llm_result_with_options(result, "A) 2x + C\nB) x42 + C\nC) x42")

    assert repaired["answer"] == "B"
    assert repaired["original_answer"] == "C"
    assert repaired["answer_repaired"] is True


def test_repair_llm_result_with_options_repairs_q16_style_numeric_result() -> None:
    result = {
        "answer": "C",
        "solution": "C",
        "explanation": "Area = width * height; 40 = x * 5; x = 40 / 5 = 8",
        "confidence": 0.86,
        "raw_response": "",
        "status": "success",
        "error": None,
    }

    repaired = repair_llm_result_with_options(result, "A)6\nB)8\nC)10\nD)12")

    assert repaired["answer"] == "B"
    assert repaired["original_answer"] == "C"
    assert repaired["answer_repaired"] is True


def test_repair_llm_result_with_options_changes_answer_when_value_matches_option() -> None:
    result = {
        "answer": "D",
        "solution": "D",
        "explanation": "f'(2) = 7",
        "confidence": 0.91,
        "raw_response": "Answer: D\nExplanation: f'(2) = 7",
        "status": "success",
        "error": None,
    }

    repaired = repair_llm_result_with_options(result, "A) 5\nB) 7\nC) 9\nD) 12")

    assert repaired["answer"] == "B"
    assert repaired["original_answer"] == "D"
    assert repaired["answer_repaired"] is True


def test_normalize_answer_variants() -> None:
    assert normalize_answer("B.") == "B"
    assert normalize_answer("B)") == "B"
    assert normalize_answer("Option B") == "B"
    assert normalize_answer("The correct answer is B.") == "B"
    assert normalize_answer("unknown") == "unknown"


def test_normalize_option_value_math_ocr_artifacts() -> None:
    assert normalize_option_value("x42 + C") == "x^2 + c"
    assert normalize_option_value("x^2+C") == "x^2 + c"
    assert normalize_option_value("x\u00b2 + C") == "x^2 + c"
    assert normalize_option_value("x4 2") == "x^2"
    assert normalize_option_value("x 42") == "x^2"
    assert normalize_option_value("x*2") == "x^2"
    assert normalize_option_value("x*24") == "x^2"
    assert normalize_option_value("x*24+C") == "x^2 + c"
    assert normalize_option_value("x²") == "x^2"


def test_llm_service_result_keys(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_mock_mode", True)
    result = solve_text_question("What is 2 + 2?")

    assert set(result.keys()) == {
        "answer",
        "solution",
        "explanation",
        "confidence",
        "raw_response",
        "status",
        "error",
        "latency_ms",
        "provider_mode",
    }


def test_real_mode_missing_model_name_fails_gracefully(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_mock_mode", False)
    monkeypatch.setattr(settings, "llm_model_name", None)
    result = solve_text_question("What is 2 + 2?")

    assert result["status"] == "failed"
    assert "model name" in result["error"].lower()
    assert result["answer"] == "unknown"
    assert result["provider_mode"] == "real"


def test_real_text_mode_calls_litellm_completion(monkeypatch) -> None:
    from types import SimpleNamespace

    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Answer: C\nExplanation: Real text response.\nConfidence: 0.77"
                    )
                )
            ]
        )

    monkeypatch.setattr(settings, "llm_mock_mode", False)
    monkeypatch.setattr(settings, "llm_model_name", "openai/test-model")
    monkeypatch.setattr(settings, "llm_api_key", "test-key")
    monkeypatch.setattr("services.llm_service.completion", fake_completion)

    result = solve_text_question("What is 5 + 4?\nA) 8\nB) 10\nC) 9")

    assert calls
    assert calls[0]["model"] == "openai/test-model"
    assert result["status"] == "success"
    assert result["answer"] == "C"
    assert result["provider_mode"] == "real"
    assert "Real text response" in result["explanation"]


def test_real_vision_mode_calls_litellm_completion(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    from PIL import Image

    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Answer: D\nExplanation: Real vision response.\nConfidence: 0.66"
                    )
                )
            ]
        )

    image_path = tmp_path / "question.png"
    Image.new("RGB", (80, 40), color="white").save(image_path)

    monkeypatch.setattr(settings, "llm_mock_mode", False)
    monkeypatch.setattr(settings, "llm_model_name", "openai/test-vision-model")
    monkeypatch.setattr(settings, "llm_api_key", "test-key")
    monkeypatch.setattr("services.llm_service.completion", fake_completion)

    result = solve_image_question(str(image_path))

    assert calls
    assert calls[0]["model"] == "openai/test-vision-model"
    content = calls[0]["messages"][0]["content"]
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert result["status"] == "success"
    assert result["answer"] == "D"
    assert result["provider_mode"] == "real"


def test_real_api_failure_returns_real_provider_mode_without_mock_explanation(monkeypatch) -> None:
    def fake_completion(**kwargs):
        raise RuntimeError("provider rejected request")

    monkeypatch.setattr(settings, "llm_mock_mode", False)
    monkeypatch.setattr(settings, "llm_model_name", "openai/test-model")
    monkeypatch.setattr(settings, "llm_api_key", "super-secret-key")
    monkeypatch.setattr("services.llm_service.completion", fake_completion)

    result = solve_text_question("What is 2 + 2?")

    assert result["status"] == "failed"
    assert result["answer"] == "unknown"
    assert result["solution"] == ""
    assert result["explanation"] == ""
    assert result["provider_mode"] == "real"
    assert "2 + 2 equals 4" not in result["explanation"]
    assert "Mock mode" not in result["explanation"]
    assert "super-secret-key" not in result["error"]
