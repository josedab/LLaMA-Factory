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

"""Provide the command-line interface launcher for LLaMA Factory v1.

This module implements the main entry point for the LLaMA Factory v1 CLI,
handling command parsing and routing to appropriate training or utility
functions. It supports various commands including supervised fine-tuning,
environment information display, and version checking.

The launcher provides a streamlined interface for accessing v1 functionality
while maintaining backward compatibility with the existing LLaMA Factory
command structure.

Key Functions:
    launch: Main entry point that parses CLI commands and dispatches to
        appropriate handlers.

Example:
    Run SFT training from command line::

        llamafactory-cli sft config.yaml

    Display version information::

        llamafactory-cli version

    Show environment details::

        llamafactory-cli env

    Display usage help::

        llamafactory-cli help

See Also:
    llamafactory.v1.trainers.sft_trainer: SFT training implementation.
    llamafactory.extras.env: Environment information utilities.
"""

import sys

from ..extras.env import VERSION, print_env


USAGE = (
    "-" * 70
    + "\n"
    + "| Usage:                                                             |\n"
    + "|   llamafactory-cli sft -h: train models                            |\n"
    + "|   llamafactory-cli version: show version info                      |\n"
    + "| Hint: You can use `lmf` as a shortcut for `llamafactory-cli`.      |\n"
    + "-" * 70
)


WELCOME = (
    "-" * 58
    + "\n"
    + f"| Welcome to LLaMA Factory, version {VERSION}"
    + " " * (21 - len(VERSION))
    + "|\n|"
    + " " * 56
    + "|\n"
    + "| Project page: https://github.com/hiyouga/LLaMA-Factory |\n"
    + "-" * 58
)


def launch():
    command = sys.argv.pop(1) if len(sys.argv) > 1 else "help"

    if command == "sft":
        from .trainers.sft_trainer import run_sft

        run_sft()

    elif command == "env":
        print_env()

    elif command == "version":
        print(WELCOME)

    elif command == "help":
        print(USAGE)

    else:
        print(f"Unknown command: {command}.\n{USAGE}")


if __name__ == "__main__":
    pass
