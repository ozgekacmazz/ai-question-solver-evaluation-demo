from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.evaluator import run_batch_evaluation


if __name__ == "__main__":
    results = run_batch_evaluation()
    total = len(results)
    correct = sum(1 for item in results if item.get("is_correct"))
    accuracy = (correct / total * 100) if total else 0.0

    print(f"Evaluation complete: {total} questions")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("Results saved to outputs/results.csv")
