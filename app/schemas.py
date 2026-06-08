from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    environment: str


class QuestionImageRequest(BaseModel):
    image_path: str
    pipeline: str = "ocr_llm"


class QuestionSolutionResponse(BaseModel):
    question_id: str | None = None
    solution: str | None = None
    pipeline: str
    confidence: float | None = None
