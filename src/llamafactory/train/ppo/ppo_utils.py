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

"""Provide utility functions for PPO training operations.

This module contains helper functions for PPO training including API-based
reward fetching, model adapter switching between default and reward heads,
and layernorm parameter management for numerical stability.

Key Functions:
    get_rewards_from_server: Fetch reward scores from an external API server.
    replace_model: Switch between default and reward adapters/valueheads.
    dump_layernorm: Save layernorm parameters before dtype conversion.
    restore_layernorm: Restore layernorm parameters after generation.

Example:
    >>> from llamafactory.train.ppo.ppo_utils import replace_model, get_rewards_from_server
    >>> # Switch to reward model adapter
    >>> replace_model(unwrapped_model, target="reward")
    >>> rewards = get_rewards_from_server(server_url, messages)
    >>> # Switch back to default adapter
    >>> replace_model(unwrapped_model, target="default")

See Also:
    - llamafactory.train.ppo.trainer: PPO trainer using these utilities.
    - llamafactory.train.trainer_utils: Model creation utilities.
"""

import json
from contextlib import nullcontext
from typing import TYPE_CHECKING, Literal, Optional

import torch
from transformers.integrations import is_deepspeed_zero3_enabled

from ...extras.packages import is_requests_available


if is_requests_available():
    import requests


if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from trl import AutoModelForCausalLMWithValueHead


def get_rewards_from_server(server_url: str, messages: list[str]) -> list["torch.Tensor"]:
    r"""Get reward scores from the API server."""
    headers = {"Content-Type": "application/json"}
    payload = {"model": "model", "messages": messages}
    response = requests.post(server_url, json=payload, headers=headers)
    rewards = json.loads(response.text)["scores"]
    return torch.Tensor(rewards)


def replace_model(model: "AutoModelForCausalLMWithValueHead", target: Literal["default", "reward"]) -> None:
    r"""Replace the default/reward modules in the model. The model is already unwrapped."""
    v_head_layer = model.v_head.summary
    if is_deepspeed_zero3_enabled():
        import deepspeed  # type: ignore

        params = [v_head_layer.weight, v_head_layer.bias]
        context_maybe_zero3 = deepspeed.zero.GatheredParameters(params, modifier_rank=0)
    else:
        context_maybe_zero3 = nullcontext()

    model.pretrained_model.set_adapter(target)  # set the LoRA adapter to be active
    with context_maybe_zero3:
        if target == "reward":  # save default head temporarily
            setattr(model, "default_head_weight", v_head_layer.weight.data.detach().clone())
            setattr(model, "default_head_bias", v_head_layer.bias.data.detach().clone())

        device = v_head_layer.weight.device
        v_head_layer.weight.data = model.get_buffer(f"{target}_head_weight").detach().clone().to(device)
        v_head_layer.bias.data = model.get_buffer(f"{target}_head_bias").detach().clone().to(device)


def dump_layernorm(model: "PreTrainedModel") -> dict[str, "torch.Tensor"]:
    r"""Dump the layernorm parameters in the model. The model is already unwrapped (and gathered)."""
    layer_norm_params = {}
    for name, param in model.named_parameters():
        if param.data.dtype == torch.float32:
            layer_norm_params[name] = param.data.detach().clone()
            param.data = param.data.to(model.config.torch_dtype)

    return layer_norm_params


def restore_layernorm(model: "PreTrainedModel", layernorm_params: Optional[dict[str, "torch.Tensor"]] = None) -> None:
    r"""Restore the layernorm parameters in the model. The model is already unwrapped (and gathered)."""
    for name, param in model.named_parameters():
        if name in layernorm_params:
            param.data = layernorm_params[name]
