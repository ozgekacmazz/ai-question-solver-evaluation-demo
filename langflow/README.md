# Langflow Integration Notes

Langflow is used in this project as an optional orchestration layer for text-based question solving. The core application works without Langflow, and Langflow is disabled by default.

## OCR + Langflow Pipeline

The current `ocr_langflow` mode follows this flow:

```text
image -> OCR -> OCR text -> Langflow flow -> answer JSON -> FastAPI/Streamlit response
```

The app first extracts text from the uploaded image with OCR. It then sends that OCR text to `services.langflow_client.run_text_solver_flow`, which calls a configured Langflow flow. The flow is expected to return answer data that can be surfaced through the existing FastAPI and Streamlit responses.

If Langflow is not configured, the app fails safely with a clear status/error response. It should not crash and should not require a real Langflow server for normal tests or mock-mode usage.

## Environment Variables

Set these values in your local `.env` when you want to use a real Langflow server:

```env
USE_LANGFLOW=true
LANGFLOW_URL=http://127.0.0.1:7860
LANGFLOW_FLOW_ID=<your-flow-id>
LANGFLOW_API_KEY=<optional>
```

Notes:

- `USE_LANGFLOW` is disabled by default.
- `LANGFLOW_API_KEY` is optional and should only be set if your Langflow server requires authentication.
- `.env` must never be committed.

## Input Contract

The Langflow flow should accept OCR text as the input value:

```text
input_value = OCR text
```

The OCR text may include the question stem and answer options. The flow should treat this as a text-only question.

## Output Contract

The Langflow flow should return a JSON-like object with these fields:

```json
{
  "answer": "A",
  "explanation": "Short reasoning for the selected option.",
  "confidence": 0.85
}
```

Expected field meanings:

- `answer`: selected option, ideally `A`, `B`, `C`, `D`, or `E`
- `explanation`: concise reasoning
- `confidence`: numeric score from `0.0` to `1.0`

See [ocr_question_solver_flow_contract.md](ocr_question_solver_flow_contract.md) for the recommended flow design.
