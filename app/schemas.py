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
    provider_mode: str | None = None
    ocr_pipeline_result: dict[str, Any] | None = None
    vision_pipeline_result: dict[str, Any] | None = None
    recommended_pipeline: str | None = None
    comparison_summary: str | None = None
    ocr_provider_mode: str | None = None
    vision_provider_mode: str | None = None
    original_answer: str | None = None
    answer_repaired: bool | None = None
    repair_reason: str | None = None
    ocr_original_answer: str | None = None
    vision_original_answer: str | None = None
    ocr_answer_repaired: bool | None = None
    vision_answer_repaired: bool | None = None
    ocr_repair_reason: str | None = None
    vision_repair_reason: str | None = None
    router_decision: dict[str, Any] | None = None
    adaptive_selected_mode: str | None = None
    langflow_status: str | None = None
    langflow_error: str | None = None
