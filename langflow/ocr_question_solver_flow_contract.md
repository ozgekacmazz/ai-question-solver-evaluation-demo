# OCR Question Solver Flow Contract

This document describes the expected Langflow design for OCR text-based question solving.

The real exported Langflow flow JSON should be added later as:

```text
langflow/ocr_question_solver_flow.json
```

Do not include API keys, tokens, or other secrets in exported flow files.

## Intended Use

This flow is intended for questions where OCR text is sufficient to solve the problem. Visual-heavy questions, including geometry diagrams, charts, tables, and image-dependent math, should use Vision mode or Adaptive mode.

## Recommended Flow Design

Use a simple text-to-JSON flow:

1. Text Input node
2. Prompt Template node
3. LLM node
4. Structured JSON output

## Input

The flow should accept OCR text through Langflow's standard input:

```text
input_value = OCR text
```

The input may include the question stem, choices, and OCR artifacts.

## Prompt Template

Recommended prompt:

```text
You are solving a multiple-choice question from OCR text.

Use only the provided OCR text. If the text is incomplete or ambiguous, choose the best supported answer and lower the confidence.

Return strict JSON only. Do not include markdown, comments, or extra text.

OCR text:
{input_value}

Return this exact JSON shape:
{
  "answer": "A/B/C/D/E",
  "explanation": "Brief reasoning for the selected option.",
  "confidence": 0.0
}

Rules:
- answer must be one of A, B, C, D, or E when possible.
- explanation should be concise.
- confidence must be a number between 0.0 and 1.0.
```

## Expected Output

The flow should return a JSON-like object:

```json
{
  "answer": "B",
  "explanation": "The calculation gives 4, which matches option B.",
  "confidence": 0.9
}
```

The Python client expects these fields:

- `answer`
- `explanation`
- `confidence`

If the Langflow response is plain text, the client will make a best-effort attempt to extract an answer, but structured JSON is preferred.
