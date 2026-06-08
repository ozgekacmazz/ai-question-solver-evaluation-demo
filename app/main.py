from fastapi import FastAPI
from app.config import settings

app = FastAPI(title="AI Question Solver Evaluation Demo")


@app.get("/health")
def health_check() -> dict:
    """Health check endpoint for the API."""
    return {
        "status": "ok",
        "service": "ai-question-solver-evaluation-demo",
        "environment": settings.environment,
    }
