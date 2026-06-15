# Model Comparison Final Report

## Accuracy

| Model | Correct | Total | Accuracy |
|---|---:|---:|---:|
| GPT | 20 | 20 | 100.0% |
| Gemini | 19 | 20 | 95.0% |
| Claude | 19 | 20 | 95.0% |
| Our tool | 17 | 20 | 85.0% |

## Summary Table

| System | Correct | Accuracy |
|---|---:|---:|
| GPT Web UI | 20/20 | 100% |
| Gemini Web UI | 19/20 | 95% |
| Claude Web UI | 19/20 | 95% |
| Local Tool | 17/20 | 85% |

## Observations

- GPT answered all 20 questions correctly in this run.
- Gemini missed `cmp_m08`; Claude also missed `cmp_m08`, which is the integral/area graph question.
- Our tool answered 17 of 20 correctly. Its wrong answers were concentrated in math questions: `cmp_m06`, `cmp_m08`, and `cmp_m10`.
- OCR/read issues were most visible on visual or table-heavy questions. `cmp_m08` lost important graph/option detail, and `cmp_m10` OCR text omitted the table values needed for solution.
- `cmp_m06` appears more like a reasoning/answer selection issue than a pure OCR failure: the OCR text was noisy, but the correct option was still present.

## Error Mix

| Row label | Count |
|---|---:|
| correct | 17 |
| model_reasoning_error | 1 |
| ocr_error | 2 |
| format_error | 0 |
