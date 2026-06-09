from services.solver_pipeline import (
    run_adaptive_pipeline,
    run_ocr_llm_pipeline,
    run_vision_llm_pipeline,
    solve_question_image,
)


def test_run_ocr_llm_pipeline_returns_expected_keys() -> None:
    result = run_ocr_llm_pipeline("dummy_image.png")
    assert result["pipeline"] == "ocr_llm"
    assert "solution" in result


def test_run_vision_llm_pipeline_returns_expected_keys() -> None:
    result = run_vision_llm_pipeline("dummy_image.png")
    assert result["pipeline"] == "vision_llm"
    assert "solution" in result


def test_solve_question_image_ocr_mode(monkeypatch) -> None:
    def mock_run_ocr_llm_pipeline(image_path: str) -> dict:
        return {
            "pipeline": "ocr_llm",
            "image_path": image_path,
            "ocr_result": {"text": "What is 2 + 2?", "status": "success"},
            "llm_result": {"status": "success"},
            "answer": "B",
            "solution": "B",
            "explanation": "2 + 2 equals 4.",
            "confidence": 0.95,
            "status": "success",
            "error": None,
            "provider_mode": "real",
        }

    monkeypatch.setattr("services.solver_pipeline.run_ocr_llm_pipeline", mock_run_ocr_llm_pipeline)

    result = solve_question_image("question.png", mode="ocr")

    assert result["pipeline"] == "ocr_llm"
    assert result["answer"] == "B"


def test_solve_question_image_vision_mode(monkeypatch) -> None:
    def mock_run_vision_llm_pipeline(image_path: str) -> dict:
        return {
            "pipeline": "vision_llm",
            "image_path": image_path,
            "ocr_result": {},
            "llm_result": {"status": "success"},
            "answer": "B",
            "solution": "B",
            "explanation": "Vision selected B.",
            "confidence": 0.90,
            "status": "success",
            "error": None,
            "provider_mode": "real",
        }

    monkeypatch.setattr("services.solver_pipeline.run_vision_llm_pipeline", mock_run_vision_llm_pipeline)

    result = solve_question_image("question.png", mode="vision")

    assert result["pipeline"] == "vision_llm"
    assert result["answer"] == "B"


def test_solve_question_image_both_mode_returns_recommendation(monkeypatch) -> None:
    def mock_run_ocr_llm_pipeline(image_path: str) -> dict:
        return {
            "pipeline": "ocr_llm",
            "image_path": image_path,
            "ocr_result": {"text": "What is 2 + 2?", "status": "success"},
            "llm_result": {"status": "success"},
            "answer": "B",
            "solution": "B",
            "explanation": "OCR selected B.",
            "confidence": 0.95,
            "status": "success",
            "error": None,
            "provider_mode": "real",
        }

    def mock_run_vision_llm_pipeline(image_path: str) -> dict:
        return {
            "pipeline": "vision_llm",
            "image_path": image_path,
            "ocr_result": {},
            "llm_result": {"status": "success"},
            "answer": "B",
            "solution": "B",
            "explanation": "Vision selected B.",
            "confidence": 0.90,
            "status": "success",
            "error": None,
            "provider_mode": "real",
        }

    monkeypatch.setattr("services.solver_pipeline.run_ocr_llm_pipeline", mock_run_ocr_llm_pipeline)
    monkeypatch.setattr("services.solver_pipeline.run_vision_llm_pipeline", mock_run_vision_llm_pipeline)

    result = solve_question_image("question.png", mode="both")

    assert result["pipeline"] == "both"
    assert result["recommended_pipeline"] == "both_agree"
    assert "comparison_summary" in result
    assert result["answer"] == "B"
    assert result["ocr_provider_mode"] == "real"
    assert result["vision_provider_mode"] == "real"
    assert result["provider_mode"] == "real"


def test_solve_question_image_adaptive_mode_calls_adaptive_path(monkeypatch) -> None:
    def mock_run_adaptive_pipeline(image_path: str) -> dict:
        return {
            "pipeline": "adaptive_ocr_llm",
            "image_path": image_path,
            "ocr_result": {"text": "What is 2 + 2?", "status": "success"},
            "llm_result": {"status": "success"},
            "answer": "B",
            "solution": "B",
            "explanation": "Adaptive selected OCR.",
            "confidence": 0.95,
            "status": "success",
            "error": None,
            "router_decision": {"recommended_mode": "ocr"},
            "adaptive_selected_mode": "ocr",
        }

    monkeypatch.setattr("services.solver_pipeline.run_adaptive_pipeline", mock_run_adaptive_pipeline)

    result = solve_question_image("question.png", mode="adaptive")

    assert result["pipeline"] == "adaptive_ocr_llm"
    assert result["adaptive_selected_mode"] == "ocr"


def test_adaptive_routes_to_ocr_without_duplicate_ocr(monkeypatch) -> None:
    ocr_calls = []

    def mock_ocr_question(image_path: str) -> dict:
        ocr_calls.append(image_path)
        return {"text": "What is 2 + 2?", "status": "success", "error": None}

    def mock_decide_pipeline(ocr_text: str, ocr_confidence=None) -> dict:
        return {
            "detected_subject": "math",
            "detected_question_type": "math_text",
            "recommended_mode": "ocr",
            "reason": "plain text",
            "confidence": 0.8,
        }

    def mock_solve_text_question(text: str) -> dict:
        return {
            "answer": "B",
            "solution": "B",
            "explanation": "2 + 2 equals 4.",
            "confidence": 0.95,
            "raw_response": "Answer: B",
            "status": "success",
            "error": None,
            "provider_mode": "mock",
        }

    monkeypatch.setattr("services.solver_pipeline.ocr_question", mock_ocr_question)
    monkeypatch.setattr("services.solver_pipeline.decide_pipeline", mock_decide_pipeline)
    monkeypatch.setattr("services.solver_pipeline.solve_text_question", mock_solve_text_question)

    result = run_adaptive_pipeline("question.png")

    assert ocr_calls == ["question.png"]
    assert result["pipeline"] == "adaptive_ocr_llm"
    assert result["adaptive_selected_mode"] == "ocr"
    assert result["router_decision"]["recommended_mode"] == "ocr"
    assert result["answer"] == "B"


def test_adaptive_routes_to_vision(monkeypatch) -> None:
    router_decision = {
        "detected_subject": "math",
        "detected_question_type": "geometry",
        "recommended_mode": "vision",
        "reason": "geometry",
        "confidence": 0.9,
    }

    def mock_ocr_question(image_path: str) -> dict:
        return {"text": "Triangle angle question", "status": "success", "error": None}

    def mock_run_vision_llm_pipeline(image_path: str) -> dict:
        return {
            "pipeline": "vision_llm",
            "image_path": image_path,
            "ocr_result": {},
            "llm_result": {"status": "success"},
            "answer": "C",
            "solution": "C",
            "explanation": "Vision selected C.",
            "confidence": 0.89,
            "status": "success",
            "error": None,
        }

    monkeypatch.setattr("services.solver_pipeline.ocr_question", mock_ocr_question)
    monkeypatch.setattr("services.solver_pipeline.decide_pipeline", lambda text, confidence=None: router_decision)
    monkeypatch.setattr("services.solver_pipeline.run_vision_llm_pipeline", mock_run_vision_llm_pipeline)

    result = run_adaptive_pipeline("question.png")

    assert result["pipeline"] == "adaptive_vision_llm"
    assert result["recommended_pipeline"] == "vision_llm"
    assert result["router_decision"] == router_decision
    assert result["adaptive_selected_mode"] == "vision"


def test_adaptive_routes_to_both(monkeypatch) -> None:
    router_decision = {
        "detected_subject": "unknown",
        "detected_question_type": "unknown",
        "recommended_mode": "both",
        "reason": "ambiguous",
        "confidence": 0.7,
    }

    def mock_ocr_question(image_path: str) -> dict:
        return {"text": "", "status": "success", "error": None}

    def mock_run_both_pipelines(image_path: str) -> dict:
        return {
            "pipeline": "both",
            "image_path": image_path,
            "ocr_result": {"text": ""},
            "llm_result": {},
            "ocr_pipeline_result": {},
            "vision_pipeline_result": {},
            "recommended_pipeline": "none",
            "comparison_summary": "Neither pipeline produced a reliable answer.",
            "answer": "unknown",
            "solution": "unknown",
            "explanation": "",
            "confidence": 0.0,
            "status": "success_with_no_reliable_answer",
            "error": "No reliable answer produced by either pipeline.",
        }

    monkeypatch.setattr("services.solver_pipeline.ocr_question", mock_ocr_question)
    monkeypatch.setattr("services.solver_pipeline.decide_pipeline", lambda text, confidence=None: router_decision)
    monkeypatch.setattr("services.solver_pipeline.run_both_pipelines", mock_run_both_pipelines)

    result = run_adaptive_pipeline("question.png")

    assert result["pipeline"] == "adaptive_both"
    assert result["router_decision"] == router_decision
    assert result["adaptive_selected_mode"] == "both"


def test_both_mode_does_not_recommend_agreement_when_both_unknown(monkeypatch) -> None:
    def mock_unknown_pipeline(pipeline: str):
        def _mock(image_path: str) -> dict:
            return {
                "pipeline": pipeline,
                "image_path": image_path,
                "ocr_result": {},
                "llm_result": {"status": "success"},
                "answer": "unknown",
                "solution": "unknown",
                "explanation": "Unknown.",
                "confidence": 0.0,
                "status": "success",
                "error": None,
            }

        return _mock

    monkeypatch.setattr("services.solver_pipeline.run_ocr_llm_pipeline", mock_unknown_pipeline("ocr_llm"))
    monkeypatch.setattr("services.solver_pipeline.run_vision_llm_pipeline", mock_unknown_pipeline("vision_llm"))

    result = solve_question_image("question.png", mode="both")

    assert result["recommended_pipeline"] == "none"
    assert result["recommended_pipeline"] != "both_agree"
    assert result["answer"] == "unknown"
    assert result["confidence"] == 0.0
    assert result["status"] == "success_with_no_reliable_answer"
    assert result["comparison_summary"] == "Neither pipeline produced a reliable answer."


def test_both_mode_recommends_vision_when_ocr_unknown(monkeypatch) -> None:
    def mock_run_ocr_llm_pipeline(image_path: str) -> dict:
        return {
            "pipeline": "ocr_llm",
            "image_path": image_path,
            "ocr_result": {"text": "", "status": "failed"},
            "llm_result": {"status": "success"},
            "answer": "unknown",
            "solution": "unknown",
            "explanation": "Unknown.",
            "confidence": 0.0,
            "status": "success",
            "error": None,
        }

    def mock_run_vision_llm_pipeline(image_path: str) -> dict:
        return {
            "pipeline": "vision_llm",
            "image_path": image_path,
            "ocr_result": {},
            "llm_result": {"status": "success"},
            "answer": "B",
            "solution": "B",
            "explanation": "Vision selected B.",
            "confidence": 0.92,
            "status": "success",
            "error": None,
        }

    monkeypatch.setattr("services.solver_pipeline.run_ocr_llm_pipeline", mock_run_ocr_llm_pipeline)
    monkeypatch.setattr("services.solver_pipeline.run_vision_llm_pipeline", mock_run_vision_llm_pipeline)

    result = solve_question_image("question.png", mode="both")

    assert result["recommended_pipeline"] == "vision_llm"
    assert result["answer"] == "B"
    assert result["confidence"] == 0.92


def test_visual_conflict_prefers_vision_when_confident(monkeypatch) -> None:
    def mock_run_ocr_llm_pipeline(image_path: str) -> dict:
        return {
            "pipeline": "ocr_llm",
            "image_path": image_path,
            "ocr_result": {"text": "Fruit Values chart. Which fruit has the highest value?", "status": "success"},
            "llm_result": {"status": "success", "raw_response": "Answer: A"},
            "answer": "A",
            "solution": "A",
            "explanation": "OCR selected A.",
            "confidence": 0.90,
            "status": "success",
            "error": None,
            "provider_mode": "real",
        }

    def mock_run_vision_llm_pipeline(image_path: str) -> dict:
        return {
            "pipeline": "vision_llm",
            "image_path": image_path,
            "ocr_result": {},
            "llm_result": {"status": "success", "raw_response": "Answer: B"},
            "answer": "B",
            "solution": "B",
            "explanation": "Vision selected B.",
            "confidence": 0.90,
            "status": "success",
            "error": None,
            "provider_mode": "real",
        }

    monkeypatch.setattr("services.solver_pipeline.run_ocr_llm_pipeline", mock_run_ocr_llm_pipeline)
    monkeypatch.setattr("services.solver_pipeline.run_vision_llm_pipeline", mock_run_vision_llm_pipeline)

    result = solve_question_image("data/sample_questions/q05_chart.png", mode="both")

    assert result["recommended_pipeline"] == "vision_llm"
    assert result["answer"] == "B"


def test_both_mode_prefers_repaired_answer_over_unrepaired_conflict(monkeypatch) -> None:
    def mock_run_ocr_llm_pipeline(image_path: str) -> dict:
        return {
            "pipeline": "ocr_llm",
            "image_path": image_path,
            "ocr_result": {"text": "What is 12 / 3 + 2?\nA) 4\nB) 6\nC) 8", "status": "success"},
            "llm_result": {
                "answer": "C",
                "solution": "C",
                "explanation": "12 / 3 + 2 equals 6, so the correct option is B.",
                "confidence": 0.77,
                "raw_response": "Answer: C\nExplanation: 12 / 3 + 2 equals 6, so the correct option is B.",
                "status": "success",
                "error": None,
                "provider_mode": "real",
            },
            "answer": "B",
            "original_answer": "C",
            "solution": "B",
            "explanation": "12 / 3 + 2 equals 6, so the correct option is B.",
            "confidence": 0.8,
            "status": "success",
            "error": None,
            "provider_mode": "real",
            "answer_repaired": True,
            "repair_reason": "Explanation matched option B",
        }

    def mock_run_vision_llm_pipeline(image_path: str) -> dict:
        return {
            "pipeline": "vision_llm",
            "image_path": image_path,
            "ocr_result": {},
            "llm_result": {
                "answer": "D",
                "solution": "D",
                "explanation": "Vision selected D.",
                "confidence": 0.9,
                "raw_response": "Answer: D\nExplanation: Vision selected D.",
                "status": "success",
                "error": None,
                "provider_mode": "real",
            },
            "answer": "D",
            "original_answer": "D",
            "solution": "D",
            "explanation": "Vision selected D.",
            "confidence": 0.9,
            "status": "success",
            "error": None,
            "provider_mode": "real",
            "answer_repaired": False,
            "repair_reason": "",
        }

    monkeypatch.setattr("services.solver_pipeline.run_ocr_llm_pipeline", mock_run_ocr_llm_pipeline)
    monkeypatch.setattr("services.solver_pipeline.run_vision_llm_pipeline", mock_run_vision_llm_pipeline)

    result = solve_question_image("question.png", mode="both")

    assert result["answer"] == "B"
    assert result["recommended_pipeline"] == "ocr_llm"
    assert result["ocr_original_answer"] == "C"
    assert result["ocr_answer_repaired"] is True


def test_non_visual_conflict_chooses_higher_confidence(monkeypatch) -> None:
    def mock_run_ocr_llm_pipeline(image_path: str) -> dict:
        return {
            "pipeline": "ocr_llm",
            "image_path": image_path,
            "ocr_result": {"text": "What is the result of the expression?", "status": "success"},
            "llm_result": {"status": "success", "raw_response": "Answer: A"},
            "answer": "A",
            "solution": "A",
            "explanation": "OCR selected A.",
            "confidence": 0.95,
            "status": "success",
            "error": None,
            "provider_mode": "real",
        }

    def mock_run_vision_llm_pipeline(image_path: str) -> dict:
        return {
            "pipeline": "vision_llm",
            "image_path": image_path,
            "ocr_result": {},
            "llm_result": {"status": "success", "raw_response": "Answer: B"},
            "answer": "B",
            "solution": "B",
            "explanation": "Vision selected B.",
            "confidence": 0.90,
            "status": "success",
            "error": None,
            "provider_mode": "real",
        }

    monkeypatch.setattr("services.solver_pipeline.run_ocr_llm_pipeline", mock_run_ocr_llm_pipeline)
    monkeypatch.setattr("services.solver_pipeline.run_vision_llm_pipeline", mock_run_vision_llm_pipeline)

    result = solve_question_image("plain_text_question.png", mode="both")

    assert result["recommended_pipeline"] == "ocr_llm"
    assert result["answer"] == "A"


def test_both_mode_repairs_vision_result_using_ocr_options(monkeypatch) -> None:
    def mock_run_ocr_llm_pipeline(image_path: str) -> dict:
        return {
            "pipeline": "ocr_llm",
            "image_path": image_path,
            "ocr_result": {
                "text": "A rectangle has area 40 and height 5. What is x?\nA)6\nB)8\nC)10\nD)12",
                "status": "success",
            },
            "llm_result": {
                "answer": "A",
                "solution": "A",
                "explanation": "OCR selected A.",
                "confidence": 0.7,
                "raw_response": "Answer: A",
                "status": "success",
                "error": None,
                "provider_mode": "real",
            },
            "answer": "A",
            "solution": "A",
            "explanation": "OCR selected A.",
            "confidence": 0.7,
            "status": "success",
            "error": None,
            "provider_mode": "real",
            "answer_repaired": False,
            "repair_reason": "",
        }

    def mock_run_vision_llm_pipeline(image_path: str) -> dict:
        return {
            "pipeline": "vision_llm",
            "image_path": image_path,
            "ocr_result": {},
            "llm_result": {
                "answer": "C",
                "solution": "C",
                "explanation": "Area = width * height; 40 = x * 5; x = 40 / 5 = 8",
                "confidence": 0.86,
                "raw_response": "Answer: C",
                "status": "success",
                "error": None,
                "provider_mode": "real",
            },
            "answer": "C",
            "solution": "C",
            "explanation": "Area = width * height; 40 = x * 5; x = 40 / 5 = 8",
            "confidence": 0.86,
            "status": "success",
            "error": None,
            "provider_mode": "real",
            "answer_repaired": False,
            "repair_reason": "",
        }

    monkeypatch.setattr("services.solver_pipeline.run_ocr_llm_pipeline", mock_run_ocr_llm_pipeline)
    monkeypatch.setattr("services.solver_pipeline.run_vision_llm_pipeline", mock_run_vision_llm_pipeline)

    result = solve_question_image("data/sample_questions/q16_mixed_rectangle.png", mode="both")

    assert result["recommended_pipeline"] == "vision_llm"
    assert result["answer"] == "B"
    assert result["vision_original_answer"] == "C"
    assert result["vision_answer_repaired"] is True


def test_solve_question_image_unsupported_mode_returns_failed_status() -> None:
    result = solve_question_image("question.png", mode="invalid")

    assert result["status"] == "failed"
    assert "supported modes" in result["error"].lower()
    assert "adaptive" in result["error"]
