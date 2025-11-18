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

"""Provide common utilities and security validation for the API module.

This module contains shared utility functions for data serialization and
security validation used throughout the API package. It provides Pydantic
model conversion helpers and security checks to prevent Local File Inclusion
(LFI) and Server-Side Request Forgery (SSRF) vulnerabilities when handling
user-provided file paths and URLs.

Security features are configurable via environment variables:
    - SAFE_MEDIA_PATH: Directory for allowed local file access
    - ALLOW_LOCAL_FILES: Enable/disable local file access (default: "1")

Key Functions:
    dictify: Convert Pydantic model to dictionary with unset fields excluded.
    jsonify: Convert Pydantic model to JSON string with unset fields excluded.
    check_lfi_path: Validate file paths against LFI attacks.
    check_ssrf_url: Validate URLs against SSRF attacks.

Key Constants:
    SAFE_MEDIA_PATH: Base directory for safe local file access.
    ALLOW_LOCAL_FILES: Flag to enable/disable local file access.

Example:
    Serialize Pydantic models::

        from llamafactory.api.common import dictify, jsonify
        from llamafactory.api.protocol import ChatCompletionResponse

        response = ChatCompletionResponse(...)
        response_dict = dictify(response)  # For internal use
        response_json = jsonify(response)  # For API response

    Validate user-provided paths and URLs::

        from llamafactory.api.common import check_lfi_path, check_ssrf_url

        # Raises HTTPException if path is outside safe directory
        check_lfi_path("/path/to/image.png")

        # Raises HTTPException if URL points to private IP
        check_ssrf_url("https://example.com/image.png")

See Also:
    - :mod:`llamafactory.api.chat`: Uses these utilities for request processing.
    - :mod:`llamafactory.api.protocol`: Pydantic models serialized by these functions.
    - OWASP SSRF Prevention: https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html
"""

import ipaddress
import json
import os
import socket
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from ..extras.misc import is_env_enabled
from ..extras.packages import is_fastapi_available


if is_fastapi_available():
    from fastapi import HTTPException, status


if TYPE_CHECKING:
    from pydantic import BaseModel


SAFE_MEDIA_PATH = os.environ.get("SAFE_MEDIA_PATH", os.path.join(os.path.dirname(__file__), "safe_media"))
ALLOW_LOCAL_FILES = is_env_enabled("ALLOW_LOCAL_FILES", "1")


def dictify(data: "BaseModel") -> dict[str, Any]:
    try:  # pydantic v2
        return data.model_dump(exclude_unset=True)
    except AttributeError:  # pydantic v1
        return data.dict(exclude_unset=True)


def jsonify(data: "BaseModel") -> str:
    try:  # pydantic v2
        return json.dumps(data.model_dump(exclude_unset=True), ensure_ascii=False)
    except AttributeError:  # pydantic v1
        return data.json(exclude_unset=True, ensure_ascii=False)


def check_lfi_path(path: str) -> None:
    """Checks if a given path is vulnerable to LFI. Raises HTTPException if unsafe."""
    if not ALLOW_LOCAL_FILES:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Local file access is disabled.")

    try:
        os.makedirs(SAFE_MEDIA_PATH, exist_ok=True)
        real_path = os.path.realpath(path)
        safe_path = os.path.realpath(SAFE_MEDIA_PATH)

        if not real_path.startswith(safe_path):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="File access is restricted to the safe media directory."
            )
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or inaccessible file path.")


def check_ssrf_url(url: str) -> None:
    """Checks if a given URL is vulnerable to SSRF. Raises HTTPException if unsafe."""
    try:
        parsed_url = urlparse(url)
        if parsed_url.scheme not in ["http", "https"]:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only HTTP/HTTPS URLs are allowed.")

        hostname = parsed_url.hostname
        if not hostname:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid URL hostname.")

        ip_info = socket.getaddrinfo(hostname, parsed_url.port)
        ip_address_str = ip_info[0][4][0]
        ip = ipaddress.ip_address(ip_address_str)

        if not ip.is_global:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access to private or reserved IP addresses is not allowed.",
            )

    except socket.gaierror:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not resolve hostname: {parsed_url.hostname}"
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid URL: {e}")
