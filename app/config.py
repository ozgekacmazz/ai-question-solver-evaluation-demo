import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DOTENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=DOTENV_PATH)


class Settings:
    """Application configuration loaded from environment variables."""

    def __init__(self) -> None:
        self.project_name: str = os.getenv("PROJECT_NAME", "ai-question-solver-evaluation-demo")
        self.environment: str = os.getenv("ENVIRONMENT", "development")
        self.llm_mock_mode: bool = os.getenv("LLM_MOCK_MODE", "true").strip().lower() in ("true", "1", "yes")
        self.llm_model_name: str | None = os.getenv("LLM_MODEL_NAME")
        self.llm_api_key: str | None = os.getenv("LLM_API_KEY")
        self.llm_api_base: str | None = os.getenv("LLM_API_BASE")
        self.ocr_api_key: str | None = os.getenv("OCR_API_KEY")
        self.langflow_url: str | None = os.getenv("LANGFLOW_URL")


settings = Settings()
