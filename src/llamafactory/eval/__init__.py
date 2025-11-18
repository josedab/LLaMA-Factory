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

"""Provide evaluation utilities for benchmarking language models.

This module contains the evaluation framework for assessing language model
performance on standardized benchmarks such as MMLU (Massive Multitask Language
Understanding). It provides utilities for loading evaluation datasets, formatting
prompts according to language-specific templates, and computing accuracy metrics
across different subject categories.

Key Classes:
    Evaluator: Main evaluation class that handles model loading, batch inference,
        and results aggregation.
    EvalTemplate: Template class for formatting evaluation prompts and parsing
        examples.

Key Functions:
    run_eval: Entry point function to run the complete evaluation pipeline.
    get_eval_template: Retrieve a registered evaluation template by language name.

Example:
    Run evaluation from command line::

        from llamafactory.eval import run_eval
        run_eval()

    Or programmatically with custom arguments::

        from llamafactory.eval.evaluator import Evaluator
        evaluator = Evaluator(args={"model_name_or_path": "path/to/model"})
        evaluator.eval()

See Also:
    llamafactory.eval.evaluator: Contains the Evaluator class implementation.
    llamafactory.eval.template: Contains evaluation template definitions.
    llamafactory.hparams: Hyperparameter configurations for evaluation.
"""
