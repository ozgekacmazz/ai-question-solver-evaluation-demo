from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.config import settings
from app.schemas import HealthResponse, SolveResponse
from services.solver_pipeline import solve_question_image

app = FastAPI(title="AI Question Solver Evaluation Demo")


@app.get("/health", response_model=HealthResponse)
def health_check() -> dict:
    """Health check endpoint for the API."""
    return {
        "status": "ok",
        "service": "ai-question-solver-evaluation-demo",
        "environment": settings.environment,
    }


@app.post("/solve", response_model=SolveResponse)
def solve_question(
    image: UploadFile = File(...),
    mode: str = Form("ocr"),
) -> Any:
    """Solve a question image with the configured pipeline."""
    supported_modes = {"ocr", "vision", "both"}
    if mode not in supported_modes:
        raise HTTPException(
            status_code=400,
            detail="Unsupported mode. Supported modes are: ocr, vision, both.",
        )

    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)

    file_name = Path(image.filename).name or "uploaded_question.png"
    upload_path = uploads_dir / file_name

    try:
        file_content = image.file.read()
        upload_path.write_bytes(file_content)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.")
    finally:
        image.file.close()

    try:
        result = solve_question_image(str(upload_path), mode=mode)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to process the uploaded image.")

    if result.get("status") != "success" and result.get("error"):
        return JSONResponse(status_code=400, content=result)
    return result
