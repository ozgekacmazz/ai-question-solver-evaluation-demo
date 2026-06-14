from pathlib import Path

import pandas as pd

from app.config import settings
from scripts.create_benchmark_questions import create_all_benchmark_questions
from scripts.create_realistic_exam_questions import create_all_realistic_exam_questions
from scripts.create_sample_questions import create_all_sample_questions
from scripts.run_evaluation import get_dataset_configs, run_evaluation
from services.evaluator import (
    evaluate_answer,
    load_ground_truth,
    normalize_answer,
    run_batch_evaluation,
)


def test_normalize_answer_single_letter() -> None:
    assert normalize_answer("B") == "B"


def test_normalize_answer_with_prefix() -> None:
    assert normalize_answer("Answer: B") == "B"


def test_normalize_answer_sentence() -> None:
    assert normalize_answer("The correct answer is C.") == "C"


def test_evaluate_answer_correct() -> None:
    result = evaluate_answer("B", "B")
    assert result["is_correct"] is True
    assert result["needs_manual_review"] is False


def test_evaluate_answer_normalizes_phrased_answers() -> None:
    result = evaluate_answer("The correct answer is B.", "B)")

    assert result["predicted_normalized"] == "B"
    assert result["correct_normalized"] == "B"
    assert result["is_correct"] is True


def test_evaluate_answer_incorrect() -> None:
    result = evaluate_answer("A", "B")
    assert result["is_correct"] is False
    assert result["needs_manual_review"] is True


def test_load_ground_truth_returns_list() -> None:
    records = load_ground_truth("data/ground_truth.json")
    assert isinstance(records, list)
    assert records


def test_sample_question_generation_and_ground_truth_paths() -> None:
    created_paths = create_all_sample_questions(Path("data/sample_questions"))
    records = load_ground_truth("data/ground_truth.json")

    expected_ids = {f"q{index:02d}" for index in range(1, 9)}
    expected_files = {
        "q01_text.png",
        "q02_math.png",
        "q03_equation.png",
        "q04_table.png",
        "q05_chart.png",
        "q06_geometry.png",
        "q07_mixed.png",
        "q08_noisy.png",
    }

    assert {path.name for path in created_paths} == expected_files
    assert len(created_paths) == 8
    assert {record["id"] for record in records} == expected_ids
    assert len(records) == 8

    for path in created_paths:
        assert path.exists()

    for record in records:
        assert Path(record["image_path"]).exists()


def test_benchmark_question_generation_and_ground_truth_paths() -> None:
    created_paths = create_all_benchmark_questions(Path("data/benchmark_questions"))
    records = load_ground_truth("data/benchmark_ground_truth.json")

    expected_ids = {f"q{index:02d}" for index in range(9, 17)}
    expected_files = {
        "q09_parabola_vertex.png",
        "q10_derivative.png",
        "q11_limit.png",
        "q12_integral.png",
        "q13_geometry_angle.png",
        "q14_parabola_graph.png",
        "q15_chart_reasoning.png",
        "q16_mixed_math_visual.png",
    }

    assert {path.name for path in created_paths} == expected_files
    assert len(created_paths) == 8
    assert {record["id"] for record in records} == expected_ids
    assert len(records) == 8

    for path in created_paths:
        assert path.exists()

    for record in records:
        assert Path(record["image_path"]).exists()
        assert record["difficulty"] == "advanced"


def test_realistic_exam_question_generation_and_ground_truth_paths() -> None:
    created_paths = create_all_realistic_exam_questions(Path("data/realistic_exam_questions"))
    records = load_ground_truth("data/realistic_exam_ground_truth.json")

    expected_ids = {f"r{index:02d}" for index in range(1, 13)}

    assert len(created_paths) == 12
    assert {record["id"] for record in records} == expected_ids
    assert len(records) == 12

    for path in created_paths:
        assert path.exists()
        assert path.suffix == ".png"

    for record in records:
        assert Path(record["image_path"]).exists()
        assert record["difficulty"] == "realistic_exam_synthetic"
        assert record["correct_answer"] in {"A", "B", "C", "D", "E"}


def test_run_batch_evaluation_creates_csv(tmp_path: Path, monkeypatch) -> None:
    output_csv = tmp_path / "results.csv"

    import services.evaluator as evaluator_module

    def mock_solve_question_image(image_path: str, mode: str = "ocr"):
        assert mode == "both"
        return {
            "pipeline": "both",
            "ocr_result": {"text": "What is 2 + 2?\nA) 3\nB) 4\nC) 5"},
            "ocr_pipeline_result": {"answer": "B", "confidence": 0.95, "status": "success"},
            "vision_pipeline_result": {"answer": "B", "confidence": 0.90, "status": "success"},
            "recommended_pipeline": "both_agree",
            "answer": "B",
            "explanation": "2 + 2 equals 4, so the correct option is B.",
            "confidence": 0.95,
            "raw_response": "Answer: B\nExplanation: 2 + 2 equals 4.",
            "status": "success",
            "error": None,
        }

    monkeypatch.setattr(evaluator_module, "solve_question_image", mock_solve_question_image)

    results = evaluator_module.run_batch_evaluation("data/ground_truth.json", str(output_csv))
    assert output_csv.exists()
    df = pd.read_csv(output_csv)
    assert "question_id" in df.columns
    assert "id" in df.columns
    assert "recommended_pipeline" in df.columns
    assert "ocr_answer" in df.columns
    assert "vision_answer" in df.columns
    assert "final_confidence" in df.columns
    assert "ocr_original_answer" in df.columns
    assert "vision_original_answer" in df.columns
    assert "ocr_answer_repaired" in df.columns
    assert "vision_answer_repaired" in df.columns
    assert "ocr_repair_reason" in df.columns
    assert "vision_repair_reason" in df.columns
    assert len(results) == len(df)


def test_run_batch_evaluation_can_run_dataset_in_both_mode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_mock_mode", True)
    output_csv = tmp_path / "results.csv"
    create_all_sample_questions(Path("data/sample_questions"))

    results = run_batch_evaluation("data/ground_truth.json", str(output_csv), mode="both")

    assert output_csv.exists()
    assert len(results) == 8
    assert all("recommended_pipeline" in item for item in results)
    assert all("vision_answer" in item for item in results)


def test_run_evaluation_dataset_configs_load_sample_and_benchmark() -> None:
    sample_config = get_dataset_configs("sample")
    benchmark_config = get_dataset_configs("benchmark")
    realistic_config = get_dataset_configs("realistic")

    assert sample_config[0]["ground_truth_path"] == "data/ground_truth.json"
    assert benchmark_config[0]["ground_truth_path"] == "data/benchmark_ground_truth.json"
    assert realistic_config[0]["ground_truth_path"] == "data/realistic_exam_ground_truth.json"
    assert len(load_ground_truth(sample_config[0]["ground_truth_path"])) == 8
    assert len(load_ground_truth(benchmark_config[0]["ground_truth_path"])) == 8


def test_run_evaluation_forwards_cli_mode_and_writes_evaluation_mode(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_evaluation as run_evaluation_module

    seen_modes = []

    monkeypatch.setattr(
        run_evaluation_module,
        "load_ground_truth",
        lambda path: [{"id": "q01", "image_path": "question.png", "correct_answer": "B"}],
    )

    def mock_evaluate_question(record: dict, mode: str = "both") -> dict:
        seen_modes.append(mode)
        return {
            "id": record["id"],
            "question_id": record["id"],
            "predicted_answer": "B",
            "is_correct": True,
            "needs_manual_review": False,
        }

    monkeypatch.setattr(run_evaluation_module, "evaluate_question", mock_evaluate_question)
    monkeypatch.setattr(
        run_evaluation_module,
        "get_output_csv_path",
        lambda dataset: str(tmp_path / f"{dataset}_results.csv"),
    )

    results, output_csv_path = run_evaluation_module.run_evaluation(dataset="sample", mode="adaptive")

    assert seen_modes == ["adaptive"]
    assert results[0]["evaluation_mode"] == "adaptive"
    df = pd.read_csv(output_csv_path)
    assert "evaluation_mode" in df.columns
    assert df.loc[0, "evaluation_mode"] == "adaptive"


def test_evaluate_question_exports_adaptive_metadata(monkeypatch) -> None:
    import services.evaluator as evaluator_module

    def mock_solve_question_image(image_path: str, mode: str = "ocr") -> dict:
        assert mode == "adaptive"
        return {
            "pipeline": "adaptive_ocr_llm",
            "ocr_result": {"text": "What is 7 + 5?\nA) 10\nB) 11\nC) 12"},
            "answer": "C",
            "confidence": 0.88,
            "status": "success",
            "error": None,
            "adaptive_selected_mode": "ocr",
            "adaptive_initial_mode": "vision",
            "adaptive_fallback_mode": "ocr",
            "router_decision": {"recommended_mode": "vision"},
        }

    monkeypatch.setattr(evaluator_module, "solve_question_image", mock_solve_question_image)

    result = evaluator_module.evaluate_question(
        {"id": "q01", "image_path": "question.png", "question_type": "mixed", "correct_answer": "C"},
        mode="adaptive",
    )

    assert result["adaptive_selected_mode"] == "ocr"
    assert result["adaptive_initial_mode"] == "vision"
    assert result["adaptive_fallback_mode"] == "ocr"
    assert result["router_decision"] == {"recommended_mode": "vision"}


def test_parse_args_accepts_mode(monkeypatch) -> None:
    import scripts.run_evaluation as run_evaluation_module

    monkeypatch.setattr(
        "sys.argv",
        ["run_evaluation.py", "--dataset", "expanded", "--mode", "adaptive"],
    )

    args = run_evaluation_module.parse_args()

    assert args.dataset == "expanded"
    assert args.mode == "adaptive"


def test_benchmark_evaluation_runs_without_api_keys(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_mock_mode", True)
    create_all_benchmark_questions(Path("data/benchmark_questions"))

    import scripts.run_evaluation as run_evaluation_module

    monkeypatch.setattr(
        run_evaluation_module,
        "get_output_csv_path",
        lambda dataset: str(tmp_path / f"{dataset}_results.csv"),
    )

    results, output_csv_path = run_evaluation(dataset="benchmark", mode="both")

    assert Path(output_csv_path).exists()
    assert len(results) == 8
    assert all("recommended_pipeline" in item for item in results)
