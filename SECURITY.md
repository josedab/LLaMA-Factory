# Security Policy

This document describes the security features and best practices for deploying LLaMA-Factory in production environments.

## Security Features

LLaMA-Factory includes several security hardening features to protect against common vulnerabilities:

### 1. API Request Logging Privacy

**Default Behavior:** Request logging is disabled by default to protect user privacy.

**Configuration:**
```bash
# Enable verbose logging (for debugging only)
export API_VERBOSE=1

# Disable verbose logging (default)
export API_VERBOSE=0
```

When verbose logging is enabled, sensitive message content is redacted and replaced with character counts to prevent data leakage.

### 2. CORS Configuration

**Default Behavior:** CORS is restricted to localhost with credentials disabled.

**Configuration:**
```bash
# Configure allowed origins (comma-separated)
export CORS_ORIGINS="https://myapp.com,https://admin.myapp.com"

# Enable credentials (only when needed)
export CORS_ALLOW_CREDENTIALS=true
```

**Security Notes:**
- Never use `CORS_ORIGINS="*"` with `CORS_ALLOW_CREDENTIALS=true` in production
- Restrict origins to only the domains that need access
- The default restricts access to `http://localhost:3000`

### 3. HTTP Request Timeouts

All external HTTP requests (for fetching images, videos, audio) include a 30-second timeout to prevent denial-of-service attacks from slow or unresponsive URLs.

### 4. Extra Arguments Validation

WebUI training extra arguments are validated against a whitelist to prevent injection attacks.

**Allowed Parameters:**
- Training: `learning_rate`, `num_train_epochs`, `per_device_train_batch_size`, `gradient_accumulation_steps`, `warmup_ratio`, `warmup_steps`, `logging_steps`, `save_steps`, `eval_steps`, `max_steps`, `seed`, `weight_decay`, `adam_beta1`, `adam_beta2`, `adam_epsilon`, `max_grad_norm`, `lr_scheduler_type`, `num_cycles`
- Evaluation: `per_device_eval_batch_size`, `eval_accumulation_steps`
- Generation: `max_new_tokens`, `top_p`, `top_k`, `temperature`, `do_sample`, `num_beams`, `repetition_penalty`, `length_penalty`
- Data: `max_samples`, `val_size`, `cutoff_len`, `preprocessing_num_workers`
- Other: `report_to`, `save_total_limit`, `save_on_each_node`, `no_cuda`, `dataloader_num_workers`, `dataloader_pin_memory`, `gradient_checkpointing`, `optim`, `group_by_length`

Any unrecognized parameters will be logged and ignored.

### 5. WebUI Authentication

**Default Behavior:** WebUI runs without authentication (warning is displayed).

**Configuration:**
```bash
# Set authentication token for production
export WEBUI_AUTH_TOKEN=$(openssl rand -hex 32)
```

When `WEBUI_AUTH_TOKEN` is set:
- Users must enter the token as password to access the WebUI
- Username can be anything (only password is validated)
- Token comparison uses timing-safe algorithm to prevent timing attacks

### 6. API Key Authentication

The API supports bearer token authentication:

```bash
# Set API key for production
export API_KEY=your-secure-api-key
```

Clients must include the API key in requests:
```bash
curl -H "Authorization: Bearer your-secure-api-key" \
     http://localhost:8000/v1/models
```

## Production Deployment Checklist

Before deploying to production:

- [ ] Set `API_VERBOSE=0` or leave unset (default is OFF)
- [ ] Configure specific `CORS_ORIGINS` instead of wildcard
- [ ] Set `CORS_ALLOW_CREDENTIALS=false` unless specifically needed
- [ ] Set `WEBUI_AUTH_TOKEN` for WebUI authentication
- [ ] Set `API_KEY` for API authentication
- [ ] Use HTTPS/TLS for all external access
- [ ] Configure firewall rules to restrict access
- [ ] Review and restrict `SAFE_MEDIA_PATH` for local file access
- [ ] Set `ALLOW_LOCAL_FILES=0` to disable local file access if not needed

## Example Production Configuration

```bash
# Security environment variables for production
export API_VERBOSE=0
export CORS_ORIGINS="https://myapp.com"
export CORS_ALLOW_CREDENTIALS=false
export WEBUI_AUTH_TOKEN=$(openssl rand -hex 32)
export API_KEY=$(openssl rand -hex 32)
export ALLOW_LOCAL_FILES=0

# Start the API
llamafactory-cli api --model_name_or_path llama3
```

## Reporting Security Vulnerabilities

If you discover a security vulnerability, please report it responsibly:

1. **Do not** create a public GitHub issue
2. Email security concerns to the maintainers
3. Include detailed steps to reproduce
4. Allow reasonable time for a fix before public disclosure

## Security Updates

Security updates are released as needed. We recommend:

1. Subscribing to release notifications
2. Regularly updating to the latest version
3. Reviewing changelog for security-related fixes

## Existing Security Features

LLaMA-Factory already includes several security protections:

- **SSRF Protection:** URL validation prevents access to private/internal IPs
- **LFI Protection:** Local file access is restricted to designated safe directories
- **Input Validation:** Request parameters are validated before processing

## Migration Guide

If you were relying on previous behavior:

### API Verbose Logging
```bash
# Restore previous behavior (logging enabled)
export API_VERBOSE=1
```

### CORS Wildcard
```bash
# Restore previous behavior (not recommended)
export CORS_ORIGINS="*"
export CORS_ALLOW_CREDENTIALS=true
```

### Extra Arguments
If you were using extra arguments that are now blocked, either:
1. Check if they're in the allowed list
2. Request the parameter be added via GitHub issue
3. Modify the `ALLOWED_EXTRA_ARGS` set in `src/llamafactory/webui/runner.py`

## Version History

- **v0.9.x**: Initial security hardening (RFC-0001)
  - Changed API_VERBOSE default to OFF
  - Restricted CORS configuration
  - Added HTTP request timeouts
  - Implemented extra arguments whitelist
  - Added WebUI authentication option
