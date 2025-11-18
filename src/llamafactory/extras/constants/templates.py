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

"""Template-related constants.

This module provides access to template-related constants.
The DEFAULT_TEMPLATE is populated by model registrations in models.py.
"""

from .models import DEFAULT_TEMPLATE


def get_template(model_name: str) -> str:
    """Get the template name for a given model.

    Args:
        model_name: The name of the model.

    Returns:
        The template name, or empty string if not found.
    """
    return DEFAULT_TEMPLATE.get(model_name, "")


__all__ = [
    "DEFAULT_TEMPLATE",
    "get_template",
]
