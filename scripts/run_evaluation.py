import argparse
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.evaluator import evaluate_question, load_ground_truth


DATASET_CONFIGS = {
    "sample": {
        "ground_truth_path": "data/ground_truth.json",
        "output_csv_path": "outputs/results.csv",
    },
    "benchmark": {
        "ground_truth_path": "data/benchmark_ground_truth.json",
        "output_csv_path": "outputs/benchmark_results.csv",
    },
    "expanded": {
        "ground_truth_path": "data/expanded_ground_truth.json",
        "output_csv_path": "outputs/expanded_results.csv",
    },
}


def get_dataset_configs(dataset: str) -> list[dict[str, str]]:
    if dataset == "all":
        return [
            DATASET_CONFIGS["sample"],
            DATASET_CONFIGS["benchmark"],
            DATASET_CONFIGS["expanded"],
        ]
    if dataset not in DATASET_CONFIGS:
        raise ValueError("Dataset must be one of: sample, benchmark, expanded, all.")
    return [DATASET_CONFIGS[dataset]]


def get_output_csv_path(dataset: str) -> str:
    if dataset == "all":
        return "outputs/all_results.csv"
    return DATASET_CONFIGS[dataset]["output_csv_path"]


def run_evaluation(dataset: str = "sample", mode: str = "both") -> tuple[list[dict[str, Any]], str]:
    configs = get_dataset_configs(dataset)
    results = []
    for config in configs:
        records = load_ground_truth(config["ground_truth_path"])
        results.extend(evaluate_question(record, mode=mode) for record in records)

    output_csv_path = get_output_csv_path(dataset)
    output_file = Path(output_csv_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_file, index=False)
    return results, output_csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run question-solver evaluation.")
    parser.add_argument(
        "--dataset",
        choices=["sample", "benchmark", "expanded", "all"],
        default="sample",
        help="Dataset to evaluate. Default: sample.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mode = "both"
    results, output_csv_path = run_evaluation(dataset=args.dataset, mode=mode)
    total = len(results)
    correct = sum(1 for item in results if item.get("is_correct"))
    accuracy = (correct / total * 100) if total else 0.0

    print(f"Dataset: {args.dataset}")
    print(f"Evaluation mode: {mode}")
    print(f"Evaluation complete: {total} questions")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Results saved to {output_csv_path}")
