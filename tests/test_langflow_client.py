"""
Tests for Langflow client integration.

These tests verify that the Langflow client:
- Returns safe responses when not configured
- Handles empty input gracefully
- Returns correct response structure
- Never requires a real Langflow server to run
"""

import pytest
from unittest.mock import patch, MagicMock
from services.langflow_client import is_langflow_configured, run_text_solver_flow


class TestLangflowConfiguration:
    """Tests for Langflow configuration checking."""

    def test_not_configured_by_default(self):
        """Langflow should not be configured by default."""
        with patch("services.langflow_client.settings") as mock_settings:
            mock_settings.use_langflow = False
            mock_settings.langflow_url = None
            mock_settings.langflow_flow_id = None
            
            assert not is_langflow_configured()

    def test_configured_only_when_all_settings_present(self):
        """Langflow is configured only if all required settings are set."""
        with patch("services.langflow_client.settings") as mock_settings:
            # All settings present
            mock_settings.use_langflow = True
            mock_settings.langflow_url = "http://localhost:7860"
            mock_settings.langflow_flow_id = "flow-id-123"
            
            assert is_langflow_configured()

    def test_not_configured_if_use_langflow_false(self):
        """If USE_LANGFLOW is false, Langflow is not configured."""
        with patch("services.langflow_client.settings") as mock_settings:
            mock_settings.use_langflow = False
            mock_settings.langflow_url = "http://localhost:7860"
            mock_settings.langflow_flow_id = "flow-id-123"
            
            assert not is_langflow_configured()

    def test_not_configured_if_url_missing(self):
        """If LANGFLOW_URL is missing, Langflow is not configured."""
        with patch("services.langflow_client.settings") as mock_settings:
            mock_settings.use_langflow = True
            mock_settings.langflow_url = None
            mock_settings.langflow_flow_id = "flow-id-123"
            
            assert not is_langflow_configured()

    def test_not_configured_if_flow_id_missing(self):
        """If LANGFLOW_FLOW_ID is missing, Langflow is not configured."""
        with patch("services.langflow_client.settings") as mock_settings:
            mock_settings.use_langflow = True
            mock_settings.langflow_url = "http://localhost:7860"
            mock_settings.langflow_flow_id = None
            
            assert not is_langflow_configured()


class TestLangflowTextSolverFlow:
    """Tests for the run_text_solver_flow function."""

    def test_returns_failed_when_not_configured(self):
        """When not configured, should return failed status."""
        with patch("services.langflow_client.settings") as mock_settings:
            mock_settings.use_langflow = False
            mock_settings.langflow_url = None
            mock_settings.langflow_flow_id = None
            
            result = run_text_solver_flow("What is 2 + 2?")
            
            assert result["status"] == "failed"
            assert "not configured" in result["error"].lower()
            assert result["answer"] == ""
            assert result["confidence"] == 0.0

    def test_returns_failed_for_empty_question(self):
        """Empty question should return failed status."""
        result = run_text_solver_flow("")
        
        assert result["status"] == "failed"
        assert "empty" in result["error"].lower()
        assert result["answer"] == ""

    def test_returns_failed_for_whitespace_only_question(self):
        """Whitespace-only question should return failed status."""
        result = run_text_solver_flow("   ")
        
        assert result["status"] == "failed"
        assert result["answer"] == ""

    def test_result_has_required_keys(self):
        """Result should always have required keys."""
        with patch("services.langflow_client.settings") as mock_settings:
            mock_settings.use_langflow = False
            
            result = run_text_solver_flow("What is 2 + 2?")
            
            required_keys = {
                "answer",
                "explanation",
                "confidence",
                "raw_response",
                "status",
                "error",
                "latency_ms",
            }
            assert set(result.keys()) == required_keys

    def test_latency_ms_is_positive_number(self):
        """Latency should be a positive number."""
        result = run_text_solver_flow("test")
        
        assert isinstance(result["latency_ms"], int)
        assert result["latency_ms"] >= 0

    def test_confidence_is_clamped_to_range(self):
        """Confidence should always be between 0.0 and 1.0."""
        with patch("services.langflow_client.settings") as mock_settings:
            mock_settings.use_langflow = True
            mock_settings.langflow_url = "http://localhost:7860"
            mock_settings.langflow_flow_id = "flow-id"
            mock_settings.langflow_api_key = None
            
            with patch("services.langflow_client.requests.post") as mock_post:
                # Response with confidence > 1.0
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "output": {
                        "answer": "B",
                        "explanation": "test",
                        "confidence": 1.5,
                    }
                }
                mock_post.return_value = mock_response
                
                result = run_text_solver_flow("What is 2 + 2?")
                
                # Should be clamped to 1.0
                assert result["confidence"] == 1.0

    def test_handles_timeout_gracefully(self):
        """Should handle request timeout without crashing."""
        with patch("services.langflow_client.settings") as mock_settings:
            mock_settings.use_langflow = True
            mock_settings.langflow_url = "http://localhost:7860"
            mock_settings.langflow_flow_id = "flow-id"
            mock_settings.langflow_api_key = None
            
            with patch("services.langflow_client.requests.post") as mock_post:
                import requests
                mock_post.side_effect = requests.exceptions.Timeout()
                
                result = run_text_solver_flow("What is 2 + 2?")
                
                assert result["status"] == "failed"
                assert "timed out" in result["error"].lower()
                assert result["answer"] == ""

    def test_handles_connection_error_gracefully(self):
        """Should handle connection errors without crashing."""
        with patch("services.langflow_client.settings") as mock_settings:
            mock_settings.use_langflow = True
            mock_settings.langflow_url = "http://localhost:7860"
            mock_settings.langflow_flow_id = "flow-id"
            mock_settings.langflow_api_key = None
            
            with patch("services.langflow_client.requests.post") as mock_post:
                import requests
                mock_post.side_effect = requests.exceptions.ConnectionError()
                
                result = run_text_solver_flow("What is 2 + 2?")
                
                assert result["status"] == "failed"
                assert "connect" in result["error"].lower()
                assert result["answer"] == ""

    def test_handles_401_authentication_error(self):
        """Should provide helpful message for authentication errors."""
        with patch("services.langflow_client.settings") as mock_settings:
            mock_settings.use_langflow = True
            mock_settings.langflow_url = "http://localhost:7860"
            mock_settings.langflow_flow_id = "flow-id"
            mock_settings.langflow_api_key = "invalid-key"
            
            with patch("services.langflow_client.requests.post") as mock_post:
                import requests
                response = MagicMock()
                response.status_code = 401
                mock_post.return_value = response
                mock_post.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError(
                    response=response
                )
                
                result = run_text_solver_flow("What is 2 + 2?")
                
                assert result["status"] == "failed"
                assert "authentication" in result["error"].lower()

    def test_handles_404_flow_not_found(self):
        """Should provide helpful message when flow is not found."""
        with patch("services.langflow_client.settings") as mock_settings:
            mock_settings.use_langflow = True
            mock_settings.langflow_url = "http://localhost:7860"
            mock_settings.langflow_flow_id = "invalid-flow-id"
            mock_settings.langflow_api_key = None
            
            with patch("services.langflow_client.requests.post") as mock_post:
                import requests
                response = MagicMock()
                response.status_code = 404
                mock_post.return_value = response
                mock_post.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError(
                    response=response
                )
                
                result = run_text_solver_flow("What is 2 + 2?")
                
                assert result["status"] == "failed"
                assert "not found" in result["error"].lower()

    def test_handles_invalid_json_response(self):
        """Should handle invalid JSON from Langflow."""
        with patch("services.langflow_client.settings") as mock_settings:
            mock_settings.use_langflow = True
            mock_settings.langflow_url = "http://localhost:7860"
            mock_settings.langflow_flow_id = "flow-id"
            mock_settings.langflow_api_key = None
            
            with patch("services.langflow_client.requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.json.side_effect = ValueError("Invalid JSON")
                mock_post.return_value = mock_response
                
                result = run_text_solver_flow("What is 2 + 2?")
                
                assert result["status"] == "failed"
                assert "invalid json" in result["error"].lower()

    def test_extracts_answer_from_dict_response(self):
        """Should extract answer from dictionary response."""
        with patch("services.langflow_client.settings") as mock_settings:
            mock_settings.use_langflow = True
            mock_settings.langflow_url = "http://localhost:7860"
            mock_settings.langflow_flow_id = "flow-id"
            mock_settings.langflow_api_key = None
            
            with patch("services.langflow_client.requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "output": {
                        "answer": "B",
                        "explanation": "The answer is B because...",
                        "confidence": 0.95,
                    }
                }
                mock_post.return_value = mock_response
                
                result = run_text_solver_flow("What is 2 + 2?")
                
                assert result["status"] == "success"
                assert result["answer"] == "B"
                assert result["explanation"] == "The answer is B because..."
                assert result["confidence"] == 0.95

    def test_extracts_from_outputs_key(self):
        """Should handle response with 'outputs' key."""
        with patch("services.langflow_client.settings") as mock_settings:
            mock_settings.use_langflow = True
            mock_settings.langflow_url = "http://localhost:7860"
            mock_settings.langflow_flow_id = "flow-id"
            mock_settings.langflow_api_key = None
            
            with patch("services.langflow_client.requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "outputs": {
                        "answer": "C",
                        "explanation": "test",
                    }
                }
                mock_post.return_value = mock_response
                
                result = run_text_solver_flow("What is 2 + 2?")
                
                assert result["status"] == "success"
                assert result["answer"] == "C"

    def test_handles_string_response(self):
        """Should handle response that is just a string."""
        with patch("services.langflow_client.settings") as mock_settings:
            mock_settings.use_langflow = True
            mock_settings.langflow_url = "http://localhost:7860"
            mock_settings.langflow_flow_id = "flow-id"
            mock_settings.langflow_api_key = None
            
            with patch("services.langflow_client.requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = "The answer is B"
                mock_post.return_value = mock_response
                
                result = run_text_solver_flow("What is 2 + 2?")
                
                assert result["status"] == "success"
                # String response used as explanation
                assert result["explanation"] == "The answer is B"

    def test_includes_api_key_as_bearer_token(self):
        """Should include API key as Bearer token if provided."""
        with patch("services.langflow_client.settings") as mock_settings:
            mock_settings.use_langflow = True
            mock_settings.langflow_url = "http://localhost:7860"
            mock_settings.langflow_flow_id = "flow-id"
            mock_settings.langflow_api_key = "test-api-key"
            
            with patch("services.langflow_client.requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = {"output": "success"}
                mock_post.return_value = mock_response
                
                run_text_solver_flow("What is 2 + 2?")
                
                # Check that headers include Authorization
                call_kwargs = mock_post.call_args[1]
                assert "Authorization" in call_kwargs["headers"]
                assert call_kwargs["headers"]["Authorization"] == "Bearer test-api-key"

    def test_does_not_include_api_key_if_not_provided(self):
        """Should not include Authorization header if no API key."""
        with patch("services.langflow_client.settings") as mock_settings:
            mock_settings.use_langflow = True
            mock_settings.langflow_url = "http://localhost:7860"
            mock_settings.langflow_flow_id = "flow-id"
            mock_settings.langflow_api_key = None
            
            with patch("services.langflow_client.requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = {"output": "success"}
                mock_post.return_value = mock_response
                
                run_text_solver_flow("What is 2 + 2?")
                
                call_kwargs = mock_post.call_args[1]
                assert "Authorization" not in call_kwargs["headers"]

    def test_sends_input_value_in_payload(self):
        """Should send question as input_value in payload."""
        with patch("services.langflow_client.settings") as mock_settings:
            mock_settings.use_langflow = True
            mock_settings.langflow_url = "http://localhost:7860"
            mock_settings.langflow_flow_id = "flow-id"
            mock_settings.langflow_api_key = None
            
            with patch("services.langflow_client.requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = {"output": "success"}
                mock_post.return_value = mock_response
                
                run_text_solver_flow("What is 2 + 2?")
                
                call_kwargs = mock_post.call_args[1]
                assert call_kwargs["json"]["input_value"] == "What is 2 + 2?"
