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

"""Provide the Gradio-based web user interface for LLaMA Factory.

This package implements a comprehensive web interface for fine-tuning and
deploying large language models. It provides an intuitive UI for model
selection, training configuration, evaluation, inference, and model export,
all built on top of Gradio components.

Key Modules:
    interface: Main entry points for creating and launching the WebUI.
    engine: Central engine managing UI state and component coordination.
    manager: Gradio component registration and lookup system.
    runner: Training/evaluation process execution and monitoring.
    chatter: Chat model management and streaming inference.
    control: UI callback functions for dynamic component updates.
    common: Shared utilities for configuration and path management.
    locales: Internationalization strings for multi-language support.
    css: Custom CSS styles for UI appearance.
    components: Subpackage with Gradio component builders for each tab.

Example:
    >>> # Launch the full WebUI
    >>> from llamafactory.webui.interface import run_web_ui
    >>> run_web_ui()
    >>> # Or create custom interface
    >>> from llamafactory.webui.interface import create_ui
    >>> demo = create_ui(demo_mode=False)
    >>> demo.queue().launch(server_name="0.0.0.0", share=True)
    >>> # For chat-only demo
    >>> from llamafactory.webui.interface import run_web_demo
    >>> run_web_demo()

See Also:
    llamafactory.chat: Core chat functionality used by WebUI.
    llamafactory.train: Training implementations invoked by WebUI.
    llamafactory.data: Data processing used for dataset handling.
"""
