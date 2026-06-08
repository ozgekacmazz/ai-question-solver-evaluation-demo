from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    environment: str


class SolveResponse(BaseModel):
    status: str
    pipeline: str
    image_path: str
    ocr_result: dict[str, Any] = Field(default_factory=dict)
    llm_result: dict[str, Any] = Field(default_factory=dict)
    answer: str
    solution: str
    explanation: str
    confidence: float
    error: str | None = None
    ocr_pipeline_result: dict[str, Any] | None = None
    vision_pipeline_result: dict[str, Any] | None = None
    recommended_pipeline: str | None = None
    comparison_summary: str | None = None
