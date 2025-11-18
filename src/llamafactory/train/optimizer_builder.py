# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the original GaLore's implementation: https://github.com/jiaweizzhao/GaLore
# and the original APOLLO's implementation: https://github.com/zhuhanqing/APOLLO
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

"""Optimizer builder pattern for GaLore and APOLLO optimizers.

This module provides a unified builder pattern for creating memory-efficient optimizers
like GaLore and APOLLO, reducing code duplication and making it easier to add new optimizers.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch
from transformers import Trainer
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names

from ..extras import logging
from ..model import find_all_linear_modules


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ..hparams import FinetuningArguments, TrainingArguments


logger = logging.get_logger(__name__)


class DummyOptimizer(torch.optim.Optimizer):
    r"""A dummy optimizer used for layerwise GaLore or APOLLO algorithms."""

    def __init__(
        self, lr: float = 1e-3, optimizer_dict: dict["torch.nn.Parameter", "torch.optim.Optimizer"] | None = None
    ) -> None:
        dummy_tensor = torch.randn(1, 1)
        self.optimizer_dict = optimizer_dict
        super().__init__([dummy_tensor], {"lr": lr})

    def zero_grad(self, set_to_none: bool = True) -> None:
        pass

    def step(self, closure: Any = None) -> None:
        pass


def _get_decay_parameter_names(model: "PreTrainedModel") -> list[str]:
    r"""Return a list of names of parameters with weight decay. (weights in non-layernorm layers)."""
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    return decay_parameters


class LowRankOptimizerBuilder(ABC):
    """Base class for building low-rank optimizers like GaLore and APOLLO.

    This builder pattern abstracts the common logic for:
    1. Finding target parameters based on module names
    2. Separating parameters into decay/nodecay groups
    3. Applying optimizer-specific settings
    4. Supporting both standard and layerwise optimization

    Subclasses need to implement:
    - _get_targets(): Returns the list of target module names
    - _get_optimizer_kwargs(): Returns optimizer-specific keyword arguments
    - _get_optimizer_class(): Returns the optimizer class to use
    - _is_layerwise(): Returns whether to use layerwise optimization
    - _get_optimizer_name(): Returns the name for logging
    """

    def __init__(
        self,
        model: "PreTrainedModel",
        training_args: "TrainingArguments",
        finetuning_args: "FinetuningArguments",
    ):
        self.model = model
        self.training_args = training_args
        self.finetuning_args = finetuning_args

    def build(self) -> "torch.optim.Optimizer":
        """Build and return the optimizer.

        Returns:
            The configured optimizer instance.
        """
        # Get target parameters
        targets = self._get_targets()
        target_params = self._collect_target_params(targets)

        # Get optimizer-specific kwargs
        optimizer_kwargs = self._get_optimizer_kwargs()

        # Separate parameters into groups
        nodecay_params, decay_params, trainable_params = self._separate_params(target_params)

        # Get optimizer class and base kwargs
        _, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.training_args)
        optim_class = self._get_optimizer_class()

        # Build the optimizer
        if self._is_layerwise():
            optimizer = self._build_layerwise_optimizer(
                nodecay_params, decay_params, target_params, trainable_params,
                optim_class, optim_kwargs, optimizer_kwargs
            )
        else:
            optimizer = self._build_standard_optimizer(
                nodecay_params, decay_params, target_params,
                optim_class, optim_kwargs, optimizer_kwargs
            )

        self._log_optimizer_info(optimizer_kwargs)
        return optimizer

    def _collect_target_params(self, targets: list[str]) -> list[torch.nn.Parameter]:
        """Collect parameters that match the target modules.

        Args:
            targets: List of module name patterns to target.

        Returns:
            List of parameters from matching modules.
        """
        target_params: list[torch.nn.Parameter] = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear) and any(target in name for target in targets):
                for param in module.parameters():
                    if param.requires_grad and len(param.shape) > 1:
                        target_params.append(param)
        return target_params

    def _separate_params(
        self, target_params: list[torch.nn.Parameter]
    ) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter], list[torch.nn.Parameter]]:
        """Separate model parameters into decay, nodecay, and target groups.

        Args:
            target_params: Parameters that will use the low-rank optimizer.

        Returns:
            Tuple of (nodecay_params, decay_params, trainable_params).
        """
        id_target_params = {id(param) for param in target_params}
        decay_params: list[torch.nn.Parameter] = []
        nodecay_params: list[torch.nn.Parameter] = []
        trainable_params: list[torch.nn.Parameter] = []
        decay_param_names = _get_decay_parameter_names(self.model)

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
                if id(param) not in id_target_params:
                    if name in decay_param_names:
                        decay_params.append(param)
                    else:
                        nodecay_params.append(param)

        return nodecay_params, decay_params, trainable_params

    def _build_layerwise_optimizer(
        self,
        nodecay_params: list[torch.nn.Parameter],
        decay_params: list[torch.nn.Parameter],
        target_params: list[torch.nn.Parameter],
        trainable_params: list[torch.nn.Parameter],
        optim_class: type,
        optim_kwargs: dict[str, Any],
        optimizer_kwargs: dict[str, Any],
    ) -> "torch.optim.Optimizer":
        """Build a layerwise optimizer that updates parameters independently.

        Args:
            nodecay_params: Parameters without weight decay.
            decay_params: Parameters with weight decay.
            target_params: Parameters using low-rank optimization.
            trainable_params: All trainable parameters.
            optim_class: The optimizer class to use.
            optim_kwargs: Base optimizer keyword arguments.
            optimizer_kwargs: Low-rank optimizer specific kwargs.

        Returns:
            A DummyOptimizer wrapping individual optimizers per parameter.
        """
        self._validate_layerwise_config()

        optimizer_dict: dict[torch.Tensor, torch.optim.Optimizer] = {}

        # Create optimizer for each parameter
        for param in nodecay_params:
            param_groups = [dict(params=[param], weight_decay=0.0)]
            optimizer_dict[param] = optim_class(param_groups, **optim_kwargs)

        for param in decay_params:
            param_groups = [dict(params=[param], weight_decay=self.training_args.weight_decay)]
            optimizer_dict[param] = optim_class(param_groups, **optim_kwargs)

        for param in target_params:
            param_groups = [dict(
                params=[param],
                weight_decay=self.training_args.weight_decay,
                **optimizer_kwargs
            )]
            optimizer_dict[param] = optim_class(param_groups, **optim_kwargs)

        # Register hooks for automatic updates
        def optimizer_hook(param: "torch.nn.Parameter"):
            if param.grad is not None:
                optimizer_dict[param].step()
                optimizer_dict[param].zero_grad()

        for param in trainable_params:
            param.register_post_accumulate_grad_hook(optimizer_hook)

        return DummyOptimizer(lr=self.training_args.learning_rate, optimizer_dict=optimizer_dict)

    def _build_standard_optimizer(
        self,
        nodecay_params: list[torch.nn.Parameter],
        decay_params: list[torch.nn.Parameter],
        target_params: list[torch.nn.Parameter],
        optim_class: type,
        optim_kwargs: dict[str, Any],
        optimizer_kwargs: dict[str, Any],
    ) -> "torch.optim.Optimizer":
        """Build a standard optimizer with parameter groups.

        Args:
            nodecay_params: Parameters without weight decay.
            decay_params: Parameters with weight decay.
            target_params: Parameters using low-rank optimization.
            optim_class: The optimizer class to use.
            optim_kwargs: Base optimizer keyword arguments.
            optimizer_kwargs: Low-rank optimizer specific kwargs.

        Returns:
            The configured optimizer.
        """
        param_groups = [
            dict(params=nodecay_params, weight_decay=0.0),
            dict(params=decay_params, weight_decay=self.training_args.weight_decay),
            dict(params=target_params, weight_decay=self.training_args.weight_decay, **optimizer_kwargs),
        ]
        return optim_class(param_groups, **optim_kwargs)

    def _validate_layerwise_config(self) -> None:
        """Validate configuration for layerwise optimization."""
        optimizer_name = self._get_optimizer_name()
        logger.warning_rank0(f"The displayed gradient norm will be all zeros in layerwise {optimizer_name}.")
        if self.training_args.gradient_accumulation_steps != 1:
            raise ValueError(f"Per-layer {optimizer_name} does not support gradient accumulation.")

    @abstractmethod
    def _get_targets(self) -> list[str]:
        """Get the list of target module name patterns.

        Returns:
            List of strings to match against module names.
        """
        pass

    @abstractmethod
    def _get_optimizer_kwargs(self) -> dict[str, Any]:
        """Get optimizer-specific keyword arguments.

        Returns:
            Dictionary of kwargs to pass to parameter groups.
        """
        pass

    @abstractmethod
    def _get_optimizer_class(self) -> type:
        """Get the optimizer class to use.

        Returns:
            The optimizer class.
        """
        pass

    @abstractmethod
    def _is_layerwise(self) -> bool:
        """Check if layerwise optimization is enabled.

        Returns:
            True if using layerwise optimization.
        """
        pass

    @abstractmethod
    def _get_optimizer_name(self) -> str:
        """Get the name of the optimizer for logging.

        Returns:
            The optimizer name.
        """
        pass

    @abstractmethod
    def _log_optimizer_info(self, optimizer_kwargs: dict[str, Any]) -> None:
        """Log information about the optimizer configuration.

        Args:
            optimizer_kwargs: The optimizer-specific kwargs to log.
        """
        pass


class GaLoreOptimizerBuilder(LowRankOptimizerBuilder):
    """Builder for GaLore (Gradient Low-Rank Projection) optimizer."""

    def _get_targets(self) -> list[str]:
        if len(self.finetuning_args.galore_target) == 1 and self.finetuning_args.galore_target[0] == "all":
            return find_all_linear_modules(self.model, self.finetuning_args.freeze_vision_tower)
        return self.finetuning_args.galore_target

    def _get_optimizer_kwargs(self) -> dict[str, Any]:
        return {
            "rank": self.finetuning_args.galore_rank,
            "update_proj_gap": self.finetuning_args.galore_update_interval,
            "scale": self.finetuning_args.galore_scale,
            "proj_type": self.finetuning_args.galore_proj_type,
        }

    def _get_optimizer_class(self) -> type:
        from galore_torch import GaLoreAdafactor, GaLoreAdamW, GaLoreAdamW8bit  # type: ignore

        if self.training_args.optim == "adamw_torch":
            return GaLoreAdamW
        elif self.training_args.optim in ["adamw_bnb_8bit", "adamw_8bit", "paged_adamw_8bit"]:
            return GaLoreAdamW8bit
        elif self.training_args.optim == "adafactor":
            return GaLoreAdafactor
        else:
            raise NotImplementedError(f"Unknown optim: {self.training_args.optim}.")

    def _is_layerwise(self) -> bool:
        return self.finetuning_args.galore_layerwise

    def _get_optimizer_name(self) -> str:
        return "GaLore"

    def _log_optimizer_info(self, optimizer_kwargs: dict[str, Any]) -> None:
        logger.info_rank0(
            f"Using GaLore optimizer with args: {optimizer_kwargs}. "
            "It may cause hanging at the start of training, wait patiently."
        )


class APOLLOOptimizerBuilder(LowRankOptimizerBuilder):
    """Builder for APOLLO (Approximated Gradient Scaling for Memory Efficient LLM Optimization) optimizer."""

    def _get_targets(self) -> list[str]:
        if len(self.finetuning_args.apollo_target) == 1 and self.finetuning_args.apollo_target[0] == "all":
            return find_all_linear_modules(self.model, self.finetuning_args.freeze_vision_tower)
        return self.finetuning_args.apollo_target

    def _get_optimizer_kwargs(self) -> dict[str, Any]:
        return {
            "rank": self.finetuning_args.apollo_rank,
            "proj": self.finetuning_args.apollo_proj,
            "proj_type": self.finetuning_args.apollo_proj_type,
            "update_proj_gap": self.finetuning_args.apollo_update_interval,
            "scale": self.finetuning_args.apollo_scale,
            "scale_type": self.finetuning_args.apollo_scale_type,
            "scale_front": self.finetuning_args.apollo_scale_front,
        }

    def _get_optimizer_class(self) -> type:
        from apollo_torch import APOLLOAdamW  # type: ignore

        if self.training_args.optim == "adamw_torch":
            return APOLLOAdamW
        else:
            raise NotImplementedError(f"Unknown optim: {self.training_args.optim}.")

    def _is_layerwise(self) -> bool:
        return self.finetuning_args.apollo_layerwise

    def _get_optimizer_name(self) -> str:
        return "APOLLO"

    def _log_optimizer_info(self, optimizer_kwargs: dict[str, Any]) -> None:
        logger.info_rank0(f"Using APOLLO optimizer with args: {optimizer_kwargs}.")


def create_galore_optimizer(
    model: "PreTrainedModel",
    training_args: "TrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> "torch.optim.Optimizer":
    """Create a GaLore optimizer using the builder pattern.

    Args:
        model: The model to optimize.
        training_args: Training configuration.
        finetuning_args: Fine-tuning configuration.

    Returns:
        The configured GaLore optimizer.
    """
    builder = GaLoreOptimizerBuilder(model, training_args, finetuning_args)
    return builder.build()


def create_apollo_optimizer(
    model: "PreTrainedModel",
    training_args: "TrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> "torch.optim.Optimizer":
    """Create an APOLLO optimizer using the builder pattern.

    Args:
        model: The model to optimize.
        training_args: Training configuration.
        finetuning_args: Fine-tuning configuration.

    Returns:
        The configured APOLLO optimizer.
    """
    builder = APOLLOOptimizerBuilder(model, training_args, finetuning_args)
    return builder.build()
