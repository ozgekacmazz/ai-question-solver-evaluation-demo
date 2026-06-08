from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def test_health_endpoint_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_solve_endpoint_returns_expected_keys(monkeypatch) -> None:
    def mock_solve_question_image(image_path: str, mode: str = "ocr") -> dict:
        return {
            "pipeline": "ocr_llm",
            "image_path": image_path,
            "ocr_result": {"text": "What is 2 + 2?"},
            "llm_result": {
                "answer": "B",
                "explanation": "2 + 2 equals 4, so the correct option is B.",
                "confidence": 0.95,
                "raw_response": "Answer: B\nExplanation: 2 + 2 equals 4.",
                "status": "success",
                "error": None,
                "latency_ms": 1,
            },
            "answer": "B",
            "solution": "B",
            "explanation": "2 + 2 equals 4, so the correct option is B.",
            "confidence": 0.95,
            "status": "success",
            "error": None,
        }

    monkeypatch.setattr("app.main.solve_question_image", mock_solve_question_image)

    image = Image.new("RGB", (640, 240), color="white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    response = client.post(
        "/solve",
        files={"image": ("question.png", buffer, "image/png")},
        data={"mode": "ocr"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    for key in [
        "pipeline",
        "image_path",
        "ocr_result",
        "llm_result",
        "answer",
        "explanation",
        "confidence",
        "error",
    ]:
        assert key in data


def test_solve_endpoint_unsupported_mode_returns_error() -> None:
    image = Image.new("RGB", (640, 240), color="white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    response = client.post(
        "/solve",
        files={"image": ("question.png", buffer, "image/png")},
        data={"mode": "vision"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only mode=ocr is supported for now."
