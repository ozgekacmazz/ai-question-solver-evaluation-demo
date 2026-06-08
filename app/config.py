import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DOTENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=DOTENV_PATH)


class Settings:
    """Application configuration loaded from environment variables."""

    def __init__(self) -> None:
        self.environment: str = os.getenv("ENVIRONMENT", "development")
        self.ocr_api_key: str | None = os.getenv("OCR_API_KEY")
        self.llm_api_key: str | None = os.getenv("LLM_API_KEY")
        self.langflow_url: str | None = os.getenv("LANGFLOW_URL")


settings = Settings()
