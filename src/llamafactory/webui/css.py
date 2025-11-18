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

"""Define custom CSS styles for the LLaMA Factory WebUI.

This module contains the CSS stylesheet string that customizes the appearance
of Gradio components in the web interface. It includes styles for buttons,
thinking/reasoning display blocks, modal dialogs, and dark mode support.

Key Variables:
    CSS: Complete CSS stylesheet string for WebUI styling, including:
        - Duplicate button styling for Hugging Face Spaces
        - Thinking summary and container styles for reasoning models
        - Modal box positioning and appearance
        - Dark mode theme overrides

Example:
    >>> from llamafactory.webui.css import CSS
    >>> import gradio as gr
    >>> # Apply custom CSS to Gradio Blocks
    >>> with gr.Blocks(css=CSS) as demo:
    ...     gr.Markdown("# LLaMA Factory")
    ...     # Add components here
    >>> demo.launch()

See Also:
    llamafactory.webui.interface: Main interface that applies this CSS.
    llamafactory.webui.chatter: Uses thinking-container classes for responses.
"""

CSS = r"""
.duplicate-button {
  margin: auto !important;
  color: white !important;
  background: black !important;
  border-radius: 100vh !important;
}

.thinking-summary {
  padding: 8px !important;
}

.thinking-summary span {
  border-radius: 4px !important;
  padding: 4px !important;
  cursor: pointer !important;
  font-size: 14px !important;
  background: rgb(245, 245, 245) !important;
}

.dark .thinking-summary span {
  background: rgb(73, 73, 73) !important;
}

.thinking-container {
  border-left: 2px solid #a6a6a6 !important;
  padding-left: 10px !important;
  margin: 4px 0 !important;
}

.thinking-container p {
  color: #a6a6a6 !important;
}

.modal-box {
  position: fixed !important;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%); /* center horizontally */
  max-width: 1000px;
  max-height: 750px;
  overflow-y: auto;
  background-color: var(--input-background-fill);
  flex-wrap: nowrap !important;
  border: 2px solid black !important;
  z-index: 1000;
  padding: 10px;
}

.dark .modal-box {
  border: 2px solid white !important;
}
"""
