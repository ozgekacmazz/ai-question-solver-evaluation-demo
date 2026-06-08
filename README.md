# AI Question Solver Evaluation Demo

## Project Purpose

This repository contains a clean Python project skeleton for evaluating two approaches to solving question images:

1. OCR + LLM: use optical character recognition to extract the question text, then solve it with a language model.
2. Direct Vision LLM: send the question image directly to a multimodal language model.

## Problem Definition

The project is designed to compare the performance of text-first and image-first pipelines when solving questions from images. The evaluation will measure how accurately each method interprets the question and arrives at a correct answer.

## Compared Pipelines

- **OCR + LLM**: extract text from an image using OCR, then feed the text to an LLM for reasoning and answer generation.
- **Direct Vision LLM**: provide the image directly to a vision-capable LLM and let the model solve the problem without intermediate text extraction.

## Planned Architecture

- `app/` for API entry points, configuration, and data schemas.
- `services/` for reusable modules covering preprocessing, OCR, LLM interaction, Langflow integration, solver pipelines, and evaluation.
- `ui/` for a Streamlit-based dashboard.
- `data/` for sample questions and ground truth annotations.
- `tests/` for unit tests.
- `reports/` for technical documentation and analysis.

## Tech Stack

- Python
- FastAPI
- Streamlit
- Pydantic
- python-dotenv
- OpenCV
- pytesseract
- requests / httpx
- pandas
- pytest
- litellm

## OCR Pipeline

This repository includes a beginner-friendly OCR pipeline that:

- loads a question image from disk,
- preprocesses it for cleaner OCR with OpenCV,
- extracts text using `pytesseract` and English language support,
- returns a structured result with `text`, `status`, `error`, and `debug_image_path`,
- and saves a debug image when requested.

`pytesseract` is a Python wrapper for the Tesseract OCR engine. Tesseract must also be installed separately on your computer for OCR to work.

To create a sample question image:

```bash
python scripts/create_sample_questions.py
```

To run the OCR service tests:

```bash
python -m pytest tests/test_ocr_service.py
```

## FastAPI API

The project exposes a simple API with a health endpoint and a solver endpoint.

- Run the API:

```bash
uvicorn app.main:app --reload
```

- Health endpoint:

```text
http://127.0.0.1:8000/health
```

- Interactive docs:

```text
http://127.0.0.1:8000/docs
```

- Solve a question image by POSTing a file to `/solve` with `mode=ocr`.

## Streamlit Demo UI

A simple local UI is available for uploading a question image and running the OCR + Mock LLM pipeline directly.

- Run the Streamlit UI:

```bash
streamlit run ui/streamlit_app.py
```

- The UI currently uses OCR + Mock LLM mode.
- Uploaded files are stored temporarily under `uploads/streamlit/` and are ignored by git.

## Setup

1. Clone the repository.
2. Create a Python virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Copy `.env.example` to `.env` and fill in any required environment variables.
5. Run the API:

```bash
uvicorn app.main:app --reload
```

6. Run the Streamlit UI:

```bash
streamlit run ui/streamlit_app.py
```

## Current Status

This project is currently a skeleton implementation ready for future development. It includes a FastAPI health endpoint, configuration support, service placeholders, a Streamlit scaffold, tests, and documentation.
