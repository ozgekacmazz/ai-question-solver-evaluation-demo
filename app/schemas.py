from typing import Any

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    environment: str


class SolveResponse(BaseModel):
    status: str
    pipeline: str
    image_path: str
    ocr_result: dict[str, Any]
    llm_result: dict[str, Any]
    answer: str
    solution: str
    explanation: str
    confidence: float
    error: str | None = None
