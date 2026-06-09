# AI Question Solver Evaluation Demo

## Project Purpose

This project compares two approaches for solving question images:

- **OCR + Text LLM**: extract text from the image first, then solve the extracted question with a text language model.
- **Direct Vision / Multimodal LLM**: send the image directly to a vision-capable language model.

The project also includes **Both / Compare** mode, which runs both pipelines and recommends the more reliable result based on answer quality and confidence.

## Why This Project Exists

Question images are not always plain text. They may include mathematical expressions, tables, charts, geometry diagrams, noisy scans, or mixed visual reasoning.

OCR can be enough for text-heavy questions, especially when the scan is clear. Vision models may be better for graph, geometry, chart, or image-heavy questions where important information is visual rather than textual. This project provides a small but practical framework for comparing those approaches.

## Compared Pipelines

1. **OCR + LLM**
   - Preprocess the image with OpenCV.
   - Extract text with Tesseract OCR.
   - Send the text to a mock or real text LLM.

2. **Direct Vision LLM**
   - Send the image directly to a mock or real multimodal LLM.
   - Avoid relying on OCR text extraction.

3. **Both / Compare**
   - Run OCR + LLM and Direct Vision LLM.
   - Compare the results.
   - Recommend OCR, Vision, agreement between both, or no reliable answer.

## Solving Modes and Latest Results

The project compares three solving modes:

- **OCR + Text LLM**
- **Direct Vision / Multimodal LLM**
- **Both / Compare**

Mock mode is used for safe, repeatable infrastructure testing without API keys or provider costs. Real API mode is used for actual LiteLLM/OpenAI-compatible model testing.

Latest real API evaluation with `LLM_MOCK_MODE=false` and `LLM_MODEL_NAME=openai/gpt-4.1-mini`:

- **Sample dataset**: 8/8 correct, 100%
- **Benchmark dataset**: 8/8 correct, 100%

These results are from the included demo datasets only. They are useful for validating this project setup, but they are not a universal guarantee for arbitrary question images.

The latest reliability pass added strict JSON handling, answer normalization, OCR option parsing, answer repair / verification logic, and provider mode visibility.

## Current Features

- FastAPI backend
- Streamlit UI
- OCR preprocessing with OpenCV
- Tesseract OCR integration
- Mock LLM mode by default
- LiteLLM integration prepared for text and vision model calls
- Langflow integration prepared and disabled by default
- Sample smoke-test dataset
- Advanced benchmark dataset
- Batch evaluation workflow
- Pytest test suite

## Project Architecture

- `app/`: FastAPI app, configuration, and response schemas.
- `services/`: OCR, image preprocessing, LLM calls, solver pipelines, evaluator, and Langflow client.
- `ui/`: Streamlit interface for uploading images and choosing a pipeline.
- `scripts/`: dataset generation and evaluation scripts.
- `data/sample_questions/`: basic smoke-test question images.
- `data/benchmark_questions/`: harder benchmark question images.
- `tests/`: automated test suite.
- `configs/`: example LiteLLM configuration.
- `langflow/`: Langflow notes and integration material.
- `reports/`: technical writeups and evaluation notes.
- `screenshots/`: project screenshots.
- `outputs/`: runtime evaluation results and debug outputs.

`outputs/` is for runtime files. CSV results such as `outputs/results.csv` and `outputs/benchmark_results.csv` should not be committed, except for placeholder files such as `.gitkeep` if present.

## Datasets

### Sample Dataset

Located at `data/sample_questions/`.

Includes:

- `q01_text.png`
- `q02_math.png`
- `q03_equation.png`
- `q04_table.png`
- `q05_chart.png`
- `q06_geometry.png`
- `q07_mixed.png`
- `q08_noisy.png`

Purpose: smoke testing and validating the pipeline in mock mode.

### Benchmark Dataset

Located at `data/benchmark_questions/`.

Includes:

- `q09_parabola_vertex.png`
- `q10_derivative.png`
- `q11_limit.png`
- `q12_integral.png`
- `q13_geometry_angle.png`
- `q14_parabola_graph.png`
- `q15_chart_reasoning.png`
- `q16_mixed_math_visual.png`

Purpose: more realistic and harder model evaluation with real API mode.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create a local environment file:

```powershell
copy .env.example .env
```

Tesseract must be installed separately for OCR. `pytesseract` is only the Python wrapper; the Tesseract OCR engine must also be available on your machine.

## Running the API

Start FastAPI:

```powershell
uvicorn app.main:app --reload
```

Useful endpoints:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`

The `/solve` endpoint accepts an uploaded image and supports:

- `mode=ocr`
- `mode=vision`
- `mode=both`

## Running the Streamlit UI

Start the local UI:

```powershell
streamlit run ui/streamlit_app.py
```

The UI lets you upload a question image and choose one of:

- OCR + LLM
- Direct Vision LLM
- Both / Compare

## Generating Question Images

Generate the sample smoke-test dataset:

```powershell
python scripts/create_sample_questions.py
```

Generate the advanced benchmark dataset:

```powershell
python scripts/create_benchmark_questions.py
```

## Running Evaluation

Evaluate the sample dataset:

```powershell
python scripts/run_evaluation.py --dataset sample
```

Evaluate the benchmark dataset:

```powershell
python scripts/run_evaluation.py --dataset benchmark
```

Evaluate both datasets:

```powershell
python scripts/run_evaluation.py --dataset all
```

The sample dataset is expected to perform well in mock mode. The benchmark dataset is intended for real model testing, so mock-mode accuracy may be low by design.

Runtime CSV outputs are saved under `outputs/` and ignored by git.

## Running Tests

Run the full test suite:

```powershell
python -m pytest -q
```

Latest known result:

```text
96 passed, 1 warning
```

## Mock Mode vs Real API Mode

Mock mode is the default. It is safe, does not require API keys, and validates the infrastructure without API costs.

Real API mode requires local `.env` configuration. Never commit real API keys.

Example `.env`:

```env
LLM_MOCK_MODE=true
LLM_MODEL_NAME=
LLM_API_KEY=
LLM_API_BASE=
LITELLM_PROXY_URL=
USE_LANGFLOW=false
LANGFLOW_URL=
LANGFLOW_API_KEY=
LANGFLOW_FLOW_ID=
```

## LiteLLM Integration

LiteLLM support is prepared for real text and vision model calls. In real API mode, the project can call LiteLLM-compatible providers using the configured model name, API key, base URL, or LiteLLM proxy URL.

Mock mode remains the recommended default for local setup, tests, and GitHub-safe demos.

## Langflow Integration

The Langflow client is integrated for the optional `mode=ocr_langflow` pipeline and is disabled by default. It can be enabled locally with `.env` settings such as `USE_LANGFLOW=true`, `LANGFLOW_URL`, `LANGFLOW_API_KEY`, and `LANGFLOW_FLOW_ID`.

This keeps the core Python application usable without Langflow. `mode=adaptive` currently routes among the existing local pipelines and can be extended later to prefer OCR + Langflow for text-heavy questions when configured.

## Current Status

- OCR pipeline works.
- Mock LLM mode works.
- Vision mock mode works.
- Both / Compare mode works.
- Sample evaluation works in mock mode.
- Real API evaluation works on the included sample and benchmark demo datasets.
- Runtime CSV results are generated locally and should not be committed.

## Limitations

- Mock mode does not represent real model intelligence.
- OCR quality depends on Tesseract installation, scan quality, and preprocessing.
- Real API accuracy depends on the selected model, prompt behavior, image quality, and dataset difficulty.
- Generated images are controlled demo samples, not a full educational dataset.

## Future Work

- Real LiteLLM API testing
- Real vision model comparison
- Langflow flow connection
- Richer benchmark set
- Latency and cost comparison
- Report updates and screenshots
