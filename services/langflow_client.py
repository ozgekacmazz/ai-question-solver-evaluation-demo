"""
Langflow client for question solving via Langflow flows.

This module provides safe integration with Langflow if configured,
without requiring Langflow to be running or available.
"""

import re
import time
from typing import Any, Dict

import requests

from app.config import settings


def is_langflow_configured() -> bool:
    """
    Check if Langflow is properly configured.
    
    Returns true only if:
    - USE_LANGFLOW is true
    - LANGFLOW_URL is set
    - LANGFLOW_FLOW_ID is set
    
    Returns:
        bool: True if Langflow is configured, False otherwise.
    """
    return (
        settings.use_langflow
        and bool(settings.langflow_url)
        and bool(settings.langflow_flow_id)
    )


def _extract_answer_from_text(text: str) -> str:
    """Extract a likely answer letter from plain-text Langflow output."""
    if not text:
        return ""

    normalized_text = str(text).strip()
    if not normalized_text:
        return ""

    patterns = [
        r"\b(?:answer|option|choice)\s*(?:is|:)?\s*([A-Za-z0-9])\b",
        r"\b([A-E])\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, normalized_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    return ""


def run_text_solver_flow(question_text: str) -> Dict[str, Any]:
    """
    Send a question to a Langflow text solver flow.
    
    If Langflow is not configured, returns a failed status safely.
    
    Args:
        question_text: The question to solve.
        
    Returns:
        Dict with keys:
        - answer: The predicted answer (e.g., "A", "B", "C", "D", "E")
        - explanation: The reasoning or explanation
        - confidence: Confidence score (0.0 to 1.0)
        - raw_response: Full raw response from Langflow
        - status: "success" or "failed"
        - error: Error message or None
        - latency_ms: Execution time in milliseconds
    """
    start_time = time.time()
    
    # Validate input
    if not question_text or not question_text.strip():
        return {
            "answer": "",
            "explanation": "Question text is empty.",
            "confidence": 0.0,
            "raw_response": "",
            "status": "failed",
            "error": "Empty question text is required.",
            "latency_ms": int((time.time() - start_time) * 1000),
        }
    
    # Check if Langflow is configured
    if not is_langflow_configured():
        return {
            "answer": "",
            "explanation": "Langflow is not configured.",
            "confidence": 0.0,
            "raw_response": "",
            "status": "failed",
            "error": "Langflow is not configured. Set USE_LANGFLOW=true, LANGFLOW_URL, and LANGFLOW_FLOW_ID.",
            "latency_ms": int((time.time() - start_time) * 1000),
        }
    
    # Make request to Langflow
    try:
        url = f"{settings.langflow_url}/api/v1/run/{settings.langflow_flow_id}"
        
        headers = {
            "Content-Type": "application/json",
        }
        
        # Add API key as Bearer token if provided
        if settings.langflow_api_key:
            headers["Authorization"] = f"Bearer {settings.langflow_api_key}"
        
        payload = {
            "input_value": question_text,
        }
        
        # Make request with timeout
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        
        result = response.json()
        
        # Try to extract answer, explanation, confidence from response
        answer = ""
        explanation = ""
        confidence = 0.0
        
        # If result has expected structure, extract fields
        if isinstance(result, dict):
            # Try to get output from common Langflow response structures
            output = result.get("output") or result.get("outputs") or result
            
            if isinstance(output, dict):
                answer = output.get("answer", "")
                explanation = output.get("explanation", "")
                confidence = float(output.get("confidence", 0.0)) if output.get("confidence") else 0.0
            elif isinstance(output, str):
                explanation = output
                answer = _extract_answer_from_text(output)
        elif isinstance(result, str):
            explanation = result
            answer = _extract_answer_from_text(result)
        
        return {
            "answer": str(answer).strip(),
            "explanation": str(explanation).strip(),
            "confidence": max(0.0, min(1.0, confidence)),  # Clamp to [0, 1]
            "raw_response": str(result),
            "status": "success",
            "error": None,
            "latency_ms": int((time.time() - start_time) * 1000),
        }
    
    except requests.exceptions.Timeout:
        return {
            "answer": "",
            "explanation": "Request timed out.",
            "confidence": 0.0,
            "raw_response": "",
            "status": "failed",
            "error": "Langflow request timed out after 30 seconds.",
            "latency_ms": int((time.time() - start_time) * 1000),
        }
    
    except requests.exceptions.ConnectionError as e:
        return {
            "answer": "",
            "explanation": "Connection failed.",
            "confidence": 0.0,
            "raw_response": "",
            "status": "failed",
            "error": "Could not connect to Langflow server. Check LANGFLOW_URL.",
            "latency_ms": int((time.time() - start_time) * 1000),
        }
    
    except requests.exceptions.HTTPError as e:
        error_msg = f"Langflow returned HTTP {e.response.status_code}."
        if e.response.status_code == 401:
            error_msg = "Langflow authentication failed. Check LANGFLOW_API_KEY."
        elif e.response.status_code == 404:
            error_msg = "Langflow flow not found. Check LANGFLOW_FLOW_ID."
        
        return {
            "answer": "",
            "explanation": "HTTP error.",
            "confidence": 0.0,
            "raw_response": "",
            "status": "failed",
            "error": error_msg,
            "latency_ms": int((time.time() - start_time) * 1000),
        }
    
    except ValueError:
        # JSON decode error
        return {
            "answer": "",
            "explanation": "Invalid response.",
            "confidence": 0.0,
            "raw_response": "",
            "status": "failed",
            "error": "Langflow returned invalid JSON.",
            "latency_ms": int((time.time() - start_time) * 1000),
        }
    
    except Exception as e:
        # Catch any other exceptions without exposing stack trace
        return {
            "answer": "",
            "explanation": "Unexpected error.",
            "confidence": 0.0,
            "raw_response": "",
            "status": "failed",
            "error": f"Unexpected error calling Langflow: {type(e).__name__}",
            "latency_ms": int((time.time() - start_time) * 1000),
        }
