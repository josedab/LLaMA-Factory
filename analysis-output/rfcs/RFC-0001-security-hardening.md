# RFC-0001: API Security Hardening

**Status:** Draft
**Author:** Claude (Automated Analysis)
**Created:** 2025-11-18
**Analysis Commit:** `45f0437`

---

## Summary

Address five critical security vulnerabilities in the API and WebUI components to prevent data leakage, CSRF attacks, and denial of service.

---

## Motivation

Security analysis revealed the following critical issues:

1. **API Verbose Logging (Privacy Risk)**: `API_VERBOSE` defaults to "1", logging all user requests
2. **CORS Misconfiguration (CSRF Risk)**: `allow_origins=["*"]` with credentials enabled
3. **Missing Request Timeouts (DoS Risk)**: HTTP requests without timeout parameter
4. **Extra Args Bypass (Injection Risk)**: JSON args deserialized without validation
5. **No WebUI Authentication**: Zero authentication on training interface

These vulnerabilities expose users to data breaches, unauthorized access, and service disruption.

---

## Detailed Design

### Fix 1: API Verbose Logging Default

**Location:** `src/llamafactory/api/chat.py:83-84`

**Current Code:**
```python
def _process_request(request: "ChatCompletionRequest"):
    if is_env_enabled("API_VERBOSE", "1"):  # Defaults to ON
        logger.info_rank0(f"==== request ====\n{json.dumps(dictify(request), indent=2)}")
```

**Proposed Change:**
```python
def _process_request(request: "ChatCompletionRequest"):
    if is_env_enabled("API_VERBOSE", "0"):  # Default to OFF
        # Redact sensitive fields
        safe_request = _redact_sensitive_fields(request)
        logger.info_rank0(f"==== request ====\n{json.dumps(safe_request, indent=2)}")


def _redact_sensitive_fields(request):
    """Redact potentially sensitive content for logging."""
    safe = dictify(request)
    if "messages" in safe:
        safe["messages"] = [
            {**m, "content": f"<{len(m.get('content', ''))} chars>"}
            for m in safe["messages"]
        ]
    return safe
```

### Fix 2: CORS Configuration

**Location:** `src/llamafactory/api/app.py:72-78`

**Current Code:**
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,  # Dangerous with wildcard origins
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Proposed Change:**
```python
# Add environment configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
CORS_ALLOW_CREDENTIALS = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true"

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,  # Configurable origins
    allow_credentials=CORS_ALLOW_CREDENTIALS,  # Default to false
    allow_methods=["GET", "POST", "OPTIONS"],  # Restrict methods
    allow_headers=["Content-Type", "Authorization"],  # Restrict headers
)
```

### Fix 3: Request Timeouts

**Location:** `src/llamafactory/api/chat.py:128, 141, 154`

**Current Code:**
```python
response = requests.get(url)  # No timeout - can hang indefinitely
```

**Proposed Change:**
```python
DEFAULT_REQUEST_TIMEOUT = 30  # seconds

response = requests.get(url, timeout=DEFAULT_REQUEST_TIMEOUT)
```

Apply to all `requests.get()` and `requests.post()` calls:
- Line 128: Image fetching
- Line 141: Audio fetching
- Line 154: Video fetching

### Fix 4: Extra Arguments Validation

**Location:** `src/llamafactory/webui/runner.py:179`

**Current Code:**
```python
extra_args = json.loads(extra_args_str)  # No validation
args.update(extra_args)  # Direct injection
```

**Proposed Change:**
```python
from pydantic import ValidationError

ALLOWED_EXTRA_ARGS = {
    "learning_rate", "num_train_epochs", "per_device_train_batch_size",
    "gradient_accumulation_steps", "warmup_ratio", "logging_steps",
    "save_steps", "eval_steps", "max_steps", "seed",
    # Add other safe parameters
}

def validate_extra_args(extra_args_str: str) -> dict:
    """Validate and sanitize extra arguments."""
    try:
        extra_args = json.loads(extra_args_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in extra_args: {e}")

    if not isinstance(extra_args, dict):
        raise ValueError("extra_args must be a JSON object")

    # Filter to allowed keys
    validated = {}
    for key, value in extra_args.items():
        if key in ALLOWED_EXTRA_ARGS:
            validated[key] = value
        else:
            logger.warning(f"Ignoring disallowed extra_arg: {key}")

    return validated


# Usage
extra_args = validate_extra_args(extra_args_str)
args.update(extra_args)
```

### Fix 5: WebUI Authentication

**Location:** `src/llamafactory/webui/interface.py:91-97`

**Current Code:**
```python
def create_ui() -> "gr.Blocks":
    # No authentication
    return demo


demo.launch(server_name="0.0.0.0", server_port=port)
```

**Proposed Change:**
```python
import secrets
import os

def create_ui() -> "gr.Blocks":
    # ... existing code ...
    return demo


def launch_ui():
    auth_token = os.getenv("WEBUI_AUTH_TOKEN")

    if auth_token:
        # Token-based authentication
        def auth_check(username, password):
            return secrets.compare_digest(password, auth_token)

        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            auth=auth_check,
            auth_message="Enter any username and the WEBUI_AUTH_TOKEN as password"
        )
    else:
        # Warn if no auth configured
        logger.warning_rank0(
            "WebUI running without authentication. "
            "Set WEBUI_AUTH_TOKEN environment variable for production use."
        )
        demo.launch(server_name="0.0.0.0", server_port=port)
```

---

## Example Usage

### Before

```bash
# Starts with verbose logging, open CORS, no auth
llamafactory-cli api --model_name_or_path llama3
```

### After

```bash
# Production-ready configuration
export API_VERBOSE=0
export CORS_ORIGINS="https://myapp.com,https://admin.myapp.com"
export WEBUI_AUTH_TOKEN=$(openssl rand -hex 32)

llamafactory-cli api --model_name_or_path llama3
```

---

## Implementation Plan

### Phase 1: Immediate Fixes (Day 1)
1. Change `API_VERBOSE` default to "0"
2. Add timeouts to all HTTP requests
3. Implement request field redaction

### Phase 2: CORS & Validation (Day 2)
4. Update CORS configuration
5. Implement extra_args whitelist validation
6. Add environment variable documentation

### Phase 3: WebUI Auth (Day 3)
7. Add optional token authentication to WebUI
8. Update documentation
9. Create migration guide

### Phase 4: Testing & Review (Day 4-5)
10. Write security tests
11. Perform penetration testing
12. Update SECURITY.md

---

## Backwards Compatibility

### Breaking Changes
- Users relying on verbose logging must explicitly enable it
- Users with `allow_origins=["*"]` CORS need to configure explicitly
- Extra args not in whitelist will be ignored

### Migration Path
```bash
# For users who want previous behavior
export API_VERBOSE=1
export CORS_ORIGINS="*"
export CORS_ALLOW_CREDENTIALS=true
```

---

## Alternatives Considered

### 1. Rate Limiting Instead of Timeouts
- Rejected: Timeouts are simpler and address the immediate DoS risk
- Consider adding rate limiting as a separate RFC

### 2. OAuth2 for WebUI
- Rejected for MVP: Too complex for first iteration
- Consider for future enhancement

### 3. Strict Argument Schema
- Rejected: Would require maintaining parallel schema definitions
- Whitelist approach is simpler and sufficient

---

## Open Questions

1. **Should we add rate limiting to the API?**
   - Suggested: Yes, as follow-up RFC

2. **What authentication methods for WebUI?**
   - MVP: Simple token auth
   - Future: OAuth2, LDAP integration

3. **Should CORS origins support wildcards?**
   - Suggested: Support `*.example.com` patterns

---

## Success Criteria

- [ ] Zero critical security findings in automated scan
- [ ] All HTTP requests have timeout configured
- [ ] CORS only allows configured origins
- [ ] WebUI has authentication option
- [ ] Extra args validated against whitelist
- [ ] Security documentation updated

---

## Effort Estimation

**Total: 3-5 dev-days**

| Task | Effort |
|------|--------|
| Code changes | 1 day |
| Tests | 1 day |
| Documentation | 0.5 day |
| Security review | 0.5-1 day |
| Integration testing | 0.5-1 day |

---

## Stakeholder Approvals

- [ ] Project maintainers
- [ ] Security reviewer
- [ ] Documentation maintainer
