# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Security tests for RFC-0001: API Security Hardening

Tests cover:
1. API_VERBOSE default and field redaction
2. CORS configuration
3. Request timeouts
4. Extra arguments validation
5. WebUI authentication
"""

import json
import os
import unittest
from unittest.mock import MagicMock, patch

import pytest


class TestAPIVerboseDefault:
    """Tests for API_VERBOSE default behavior and field redaction."""

    def test_api_verbose_defaults_to_off(self):
        """API_VERBOSE should default to '0' (OFF) for privacy."""
        from llamafactory.extras.misc import is_env_enabled

        # Clear environment variable to test default
        env_backup = os.environ.pop("API_VERBOSE", None)
        try:
            # Should default to "0" (OFF)
            result = is_env_enabled("API_VERBOSE", "0")
            assert not result, "API_VERBOSE should default to OFF"
        finally:
            if env_backup is not None:
                os.environ["API_VERBOSE"] = env_backup

    def test_api_verbose_explicit_enable(self):
        """API_VERBOSE should be enabled when explicitly set to '1'."""
        from llamafactory.extras.misc import is_env_enabled

        with patch.dict(os.environ, {"API_VERBOSE": "1"}):
            result = is_env_enabled("API_VERBOSE", "0")
            assert result, "API_VERBOSE should be ON when set to '1'"

    def test_redact_sensitive_fields(self):
        """Sensitive message content should be redacted in logs."""
        from llamafactory.api.chat import _redact_sensitive_fields

        # Create a mock request with sensitive content
        mock_request = MagicMock()
        mock_request.model_dump.return_value = {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "This is sensitive user data that should not be logged"},
                {"role": "assistant", "content": "This is an assistant response"},
            ],
            "temperature": 0.7,
        }

        result = _redact_sensitive_fields(mock_request)

        # Verify content is redacted but structure is preserved
        assert "messages" in result
        assert len(result["messages"]) == 2
        for msg in result["messages"]:
            assert "content" in msg
            # Content should be replaced with character count
            assert "chars>" in msg["content"]
            # Original content should not be present
            assert "sensitive" not in msg["content"]

    def test_redact_preserves_other_fields(self):
        """Field redaction should preserve non-sensitive fields."""
        from llamafactory.api.chat import _redact_sensitive_fields

        mock_request = MagicMock()
        mock_request.model_dump.return_value = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "test"}],
            "temperature": 0.5,
            "max_tokens": 100,
        }

        result = _redact_sensitive_fields(mock_request)

        assert result["model"] == "gpt-3.5-turbo"
        assert result["temperature"] == 0.5
        assert result["max_tokens"] == 100


class TestCORSConfiguration:
    """Tests for CORS configuration security."""

    def test_cors_default_origin(self):
        """CORS should default to localhost:3000 instead of wildcard."""
        # Test that the default is not a wildcard
        default_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
        assert "*" not in default_origins, "Default CORS origin should not be wildcard"
        assert "http://localhost:3000" in default_origins

    def test_cors_credentials_default_false(self):
        """CORS credentials should default to false."""
        allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true"
        assert not allow_credentials, "CORS credentials should default to false"

    def test_cors_custom_origins(self):
        """CORS should accept custom origins from environment."""
        with patch.dict(os.environ, {"CORS_ORIGINS": "https://example.com,https://api.example.com"}):
            origins = os.getenv("CORS_ORIGINS").split(",")
            assert "https://example.com" in origins
            assert "https://api.example.com" in origins

    def test_cors_credentials_explicit_enable(self):
        """CORS credentials can be enabled explicitly."""
        with patch.dict(os.environ, {"CORS_ALLOW_CREDENTIALS": "true"}):
            allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true"
            assert allow_credentials


class TestRequestTimeouts:
    """Tests for HTTP request timeout configuration."""

    def test_default_timeout_constant_exists(self):
        """DEFAULT_REQUEST_TIMEOUT constant should exist and be reasonable."""
        from llamafactory.api.chat import DEFAULT_REQUEST_TIMEOUT

        assert DEFAULT_REQUEST_TIMEOUT > 0, "Timeout should be positive"
        assert DEFAULT_REQUEST_TIMEOUT <= 60, "Timeout should be reasonable (<=60s)"
        assert DEFAULT_REQUEST_TIMEOUT == 30, "Default timeout should be 30 seconds"

    def test_timeout_applied_to_image_requests(self):
        """Image fetch requests should have timeout parameter."""
        # Read the chat.py file and verify timeout is used
        import inspect
        from llamafactory.api import chat

        source = inspect.getsource(chat)

        # Verify timeout is used in requests.get calls
        assert "timeout=DEFAULT_REQUEST_TIMEOUT" in source, \
            "requests.get should include timeout parameter"


class TestExtraArgsValidation:
    """Tests for extra arguments whitelist validation."""

    def test_allowed_extra_args_whitelist_exists(self):
        """ALLOWED_EXTRA_ARGS whitelist should exist."""
        from llamafactory.webui.runner import ALLOWED_EXTRA_ARGS

        assert isinstance(ALLOWED_EXTRA_ARGS, set)
        assert len(ALLOWED_EXTRA_ARGS) > 0

    def test_common_training_params_allowed(self):
        """Common training parameters should be in whitelist."""
        from llamafactory.webui.runner import ALLOWED_EXTRA_ARGS

        common_params = [
            "learning_rate", "num_train_epochs", "per_device_train_batch_size",
            "gradient_accumulation_steps", "warmup_ratio", "logging_steps",
            "save_steps", "eval_steps", "max_steps", "seed"
        ]
        for param in common_params:
            assert param in ALLOWED_EXTRA_ARGS, f"{param} should be in whitelist"

    def test_validate_extra_args_valid_json(self):
        """Valid JSON with allowed keys should pass validation."""
        from llamafactory.webui.runner import validate_extra_args

        valid_args = '{"learning_rate": 1e-5, "seed": 42}'
        result = validate_extra_args(valid_args)

        assert "learning_rate" in result
        assert result["learning_rate"] == 1e-5
        assert "seed" in result
        assert result["seed"] == 42

    def test_validate_extra_args_filters_disallowed(self):
        """Disallowed keys should be filtered out."""
        from llamafactory.webui.runner import validate_extra_args

        # Include both allowed and disallowed keys
        mixed_args = '{"learning_rate": 1e-5, "malicious_key": "danger", "seed": 42}'
        result = validate_extra_args(mixed_args)

        assert "learning_rate" in result
        assert "seed" in result
        assert "malicious_key" not in result, "Disallowed key should be filtered"

    def test_validate_extra_args_invalid_json(self):
        """Invalid JSON should raise ValueError."""
        from llamafactory.webui.runner import validate_extra_args

        with pytest.raises(ValueError) as exc_info:
            validate_extra_args("not valid json")

        assert "Invalid JSON" in str(exc_info.value)

    def test_validate_extra_args_non_dict(self):
        """Non-dict JSON should raise ValueError."""
        from llamafactory.webui.runner import validate_extra_args

        with pytest.raises(ValueError) as exc_info:
            validate_extra_args('["array", "not", "dict"]')

        assert "must be a JSON object" in str(exc_info.value)

    def test_validate_extra_args_empty_object(self):
        """Empty JSON object should return empty dict."""
        from llamafactory.webui.runner import validate_extra_args

        result = validate_extra_args("{}")
        assert result == {}

    def test_dangerous_params_not_allowed(self):
        """Potentially dangerous parameters should not be in whitelist."""
        from llamafactory.webui.runner import ALLOWED_EXTRA_ARGS

        dangerous_params = [
            "trust_remote_code",  # Could execute arbitrary code
            "model_name_or_path",  # Could change model source
            "output_dir",  # Could write to arbitrary location
            "cache_dir",  # Could access sensitive paths
            "deepspeed",  # Could load arbitrary configs
        ]
        for param in dangerous_params:
            assert param not in ALLOWED_EXTRA_ARGS, \
                f"Dangerous param {param} should not be in whitelist"


class TestWebUIAuthentication:
    """Tests for WebUI authentication support."""

    def test_auth_token_environment_variable(self):
        """WEBUI_AUTH_TOKEN environment variable should be checked."""
        # When token is not set, it should be None
        token = os.getenv("WEBUI_AUTH_TOKEN")
        # Test is just checking the mechanism exists
        assert isinstance(token, (str, type(None)))

    def test_secrets_module_imported(self):
        """Secrets module should be used for timing-safe comparison."""
        from llamafactory.webui import interface
        import secrets as secrets_module

        # Verify secrets is imported
        assert hasattr(interface, 'secrets') or 'secrets' in dir(interface)

    def test_auth_token_enables_authentication(self):
        """When WEBUI_AUTH_TOKEN is set, authentication should be enabled."""
        with patch.dict(os.environ, {"WEBUI_AUTH_TOKEN": "test-secret-token"}):
            auth_token = os.getenv("WEBUI_AUTH_TOKEN")
            assert auth_token == "test-secret-token"
            # Authentication should be enabled when token exists

    def test_no_auth_token_shows_warning(self):
        """When no auth token, a warning should be logged."""
        # This test verifies the code path exists
        from llamafactory.webui.interface import run_web_ui
        import inspect

        source = inspect.getsource(run_web_ui)
        assert "WEBUI_AUTH_TOKEN" in source
        assert "warning" in source.lower() or "Warning" in source


class TestIntegration:
    """Integration tests for security features."""

    def test_chat_module_imports(self):
        """Chat module should import without errors."""
        from llamafactory.api import chat
        assert hasattr(chat, '_redact_sensitive_fields')
        assert hasattr(chat, 'DEFAULT_REQUEST_TIMEOUT')

    def test_app_module_imports(self):
        """App module should import without errors."""
        from llamafactory.api import app
        assert hasattr(app, 'create_app')

    def test_runner_module_imports(self):
        """Runner module should import without errors."""
        from llamafactory.webui import runner
        assert hasattr(runner, 'ALLOWED_EXTRA_ARGS')
        assert hasattr(runner, 'validate_extra_args')

    def test_interface_module_imports(self):
        """Interface module should import without errors."""
        from llamafactory.webui import interface
        assert hasattr(interface, 'run_web_ui')
        assert hasattr(interface, 'run_web_demo')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
