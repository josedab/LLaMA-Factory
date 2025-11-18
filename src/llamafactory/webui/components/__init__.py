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

"""Export Gradio component builders for WebUI tabs and sections.

This package provides factory functions that create Gradio UI components
for each section of the LLaMA Factory web interface. Each builder returns
a dictionary of components that can be registered with the Manager for
event handling and dynamic updates.

Key Functions:
    create_chat_box: Build chat interface with message input and controls.
    create_eval_tab: Build evaluation and prediction configuration tab.
    create_export_tab: Build model export configuration tab.
    create_footer: Build footer with device memory display.
    create_infer_tab: Build inference tab with model loading and chat.
    create_top: Build top section with model and configuration selection.
    create_train_tab: Build training configuration tab with all options.

Example:
    >>> from llamafactory.webui.components import create_top, create_train_tab
    >>> from llamafactory.webui.engine import Engine
    >>> engine = Engine()
    >>> # Create top section components
    >>> top_elems = create_top()
    >>> engine.manager.add_elems("top", top_elems)
    >>> # Create training tab
    >>> train_elems = create_train_tab(engine)
    >>> engine.manager.add_elems("train", train_elems)

See Also:
    llamafactory.webui.interface: Uses these builders to construct the UI.
    llamafactory.webui.manager.Manager: Registers returned component dicts.
    llamafactory.webui.engine.Engine: Passed to builders requiring engine access.
"""

from .chatbot import create_chat_box
from .eval import create_eval_tab
from .export import create_export_tab
from .footer import create_footer
from .infer import create_infer_tab
from .top import create_top
from .train import create_train_tab


__all__ = [
    "create_chat_box",
    "create_eval_tab",
    "create_export_tab",
    "create_footer",
    "create_infer_tab",
    "create_top",
    "create_train_tab",
]
