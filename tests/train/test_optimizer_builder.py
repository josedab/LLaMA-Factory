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

"""Tests for the optimizer builder pattern."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from llamafactory.extras.packages import is_apollo_available, is_galore_available
from llamafactory.train.optimizer_builder import (
    APOLLOOptimizerBuilder,
    DummyOptimizer,
    GaLoreOptimizerBuilder,
    LowRankOptimizerBuilder,
    _get_decay_parameter_names,
    create_apollo_optimizer,
    create_galore_optimizer,
)


class SimpleModel(nn.Module):
    """A simple model for testing optimizer builders."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 16)
        self.linear1 = nn.Linear(16, 32)
        self.norm = nn.LayerNorm(32)
        self.linear2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.embed(x)
        x = self.linear1(x)
        x = self.norm(x)
        x = self.linear2(x)
        return x


class TestDummyOptimizer:
    """Tests for the DummyOptimizer class."""

    def test_init_default(self):
        """Test DummyOptimizer initialization with default values."""
        optimizer = DummyOptimizer()
        assert optimizer.defaults["lr"] == 1e-3
        assert optimizer.optimizer_dict is None

    def test_init_custom_lr(self):
        """Test DummyOptimizer initialization with custom learning rate."""
        optimizer = DummyOptimizer(lr=0.01)
        assert optimizer.defaults["lr"] == 0.01

    def test_init_with_optimizer_dict(self):
        """Test DummyOptimizer initialization with optimizer dict."""
        param = nn.Parameter(torch.randn(3, 3))
        inner_optimizer = torch.optim.Adam([param])
        optimizer_dict = {param: inner_optimizer}

        optimizer = DummyOptimizer(lr=0.001, optimizer_dict=optimizer_dict)
        assert optimizer.optimizer_dict is optimizer_dict
        assert len(optimizer.optimizer_dict) == 1

    def test_zero_grad_does_nothing(self):
        """Test that zero_grad is a no-op."""
        optimizer = DummyOptimizer()
        # Should not raise any exceptions
        optimizer.zero_grad()
        optimizer.zero_grad(set_to_none=False)

    def test_step_does_nothing(self):
        """Test that step is a no-op."""
        optimizer = DummyOptimizer()
        # Should not raise any exceptions
        result = optimizer.step()
        assert result is None


class TestGetDecayParameterNames:
    """Tests for the _get_decay_parameter_names function."""

    def test_excludes_bias_parameters(self):
        """Test that bias parameters are excluded from decay."""
        model = SimpleModel()
        decay_names = _get_decay_parameter_names(model)

        for name in decay_names:
            assert "bias" not in name

    def test_excludes_layernorm_parameters(self):
        """Test that LayerNorm parameters are excluded from decay."""
        model = SimpleModel()
        decay_names = _get_decay_parameter_names(model)

        # LayerNorm weight and bias should not be in decay list
        for name in decay_names:
            assert "norm" not in name or "weight" not in name

    def test_includes_linear_weights(self):
        """Test that linear layer weights are included for decay."""
        model = SimpleModel()
        decay_names = _get_decay_parameter_names(model)

        # Linear weights should be in the decay list
        assert any("linear1.weight" in name for name in decay_names)
        assert any("linear2.weight" in name for name in decay_names)


class TestLowRankOptimizerBuilder:
    """Tests for the LowRankOptimizerBuilder base class."""

    def test_collect_target_params(self):
        """Test that target parameters are correctly collected."""
        model = SimpleModel()

        # Create a concrete implementation for testing
        class TestBuilder(LowRankOptimizerBuilder):
            def _get_targets(self):
                return ["linear1"]

            def _get_optimizer_kwargs(self):
                return {}

            def _get_optimizer_class(self):
                return torch.optim.Adam

            def _is_layerwise(self):
                return False

            def _get_optimizer_name(self):
                return "Test"

            def _log_optimizer_info(self, kwargs):
                pass

        builder = TestBuilder(model, MagicMock(), MagicMock())
        target_params = builder._collect_target_params(["linear1"])

        # Should only collect linear1's weight (shape > 1)
        assert len(target_params) == 1
        assert target_params[0].shape == torch.Size([32, 16])

    def test_separate_params(self):
        """Test that parameters are correctly separated into groups."""
        model = SimpleModel()

        class TestBuilder(LowRankOptimizerBuilder):
            def _get_targets(self):
                return ["linear1"]

            def _get_optimizer_kwargs(self):
                return {}

            def _get_optimizer_class(self):
                return torch.optim.Adam

            def _is_layerwise(self):
                return False

            def _get_optimizer_name(self):
                return "Test"

            def _log_optimizer_info(self, kwargs):
                pass

        builder = TestBuilder(model, MagicMock(), MagicMock())
        target_params = builder._collect_target_params(["linear1"])
        nodecay, decay, trainable = builder._separate_params(target_params)

        # Check that all trainable params are accounted for
        total_params = len(nodecay) + len(decay) + len(target_params)
        assert len(trainable) == total_params

        # Check that no param appears in multiple groups
        all_ids = [id(p) for p in nodecay] + [id(p) for p in decay] + [id(p) for p in target_params]
        assert len(all_ids) == len(set(all_ids))


@pytest.mark.skipif(not is_galore_available(), reason="GaLore is not available")
class TestGaLoreOptimizerBuilder:
    """Tests for the GaLoreOptimizerBuilder class."""

    @pytest.fixture
    def mock_args(self):
        """Create mock training and finetuning args."""
        training_args = MagicMock()
        training_args.optim = "adamw_torch"
        training_args.learning_rate = 1e-4
        training_args.weight_decay = 0.01
        training_args.gradient_accumulation_steps = 1

        finetuning_args = MagicMock()
        finetuning_args.galore_target = ["linear1", "linear2"]
        finetuning_args.galore_rank = 8
        finetuning_args.galore_update_interval = 200
        finetuning_args.galore_scale = 1.0
        finetuning_args.galore_proj_type = "std"
        finetuning_args.galore_layerwise = False
        finetuning_args.freeze_vision_tower = True

        return training_args, finetuning_args

    def test_get_targets_specific(self, mock_args):
        """Test getting specific target modules."""
        model = SimpleModel()
        training_args, finetuning_args = mock_args

        builder = GaLoreOptimizerBuilder(model, training_args, finetuning_args)
        targets = builder._get_targets()

        assert targets == ["linear1", "linear2"]

    def test_get_optimizer_kwargs(self, mock_args):
        """Test getting optimizer-specific kwargs."""
        model = SimpleModel()
        training_args, finetuning_args = mock_args

        builder = GaLoreOptimizerBuilder(model, training_args, finetuning_args)
        kwargs = builder._get_optimizer_kwargs()

        assert kwargs["rank"] == 8
        assert kwargs["update_proj_gap"] == 200
        assert kwargs["scale"] == 1.0
        assert kwargs["proj_type"] == "std"

    def test_get_optimizer_class_adamw(self, mock_args):
        """Test getting AdamW optimizer class."""
        model = SimpleModel()
        training_args, finetuning_args = mock_args

        builder = GaLoreOptimizerBuilder(model, training_args, finetuning_args)
        optim_class = builder._get_optimizer_class()

        from galore_torch import GaLoreAdamW
        assert optim_class is GaLoreAdamW

    def test_get_optimizer_name(self, mock_args):
        """Test getting optimizer name."""
        model = SimpleModel()
        training_args, finetuning_args = mock_args

        builder = GaLoreOptimizerBuilder(model, training_args, finetuning_args)
        assert builder._get_optimizer_name() == "GaLore"

    def test_build_standard_optimizer(self, mock_args):
        """Test building a standard (non-layerwise) optimizer."""
        model = SimpleModel()
        training_args, finetuning_args = mock_args

        with patch.object(GaLoreOptimizerBuilder, '_log_optimizer_info'):
            optimizer = create_galore_optimizer(model, training_args, finetuning_args)

        assert optimizer is not None
        # Check that param groups are created
        assert len(optimizer.param_groups) == 3

    def test_build_layerwise_optimizer(self, mock_args):
        """Test building a layerwise optimizer."""
        model = SimpleModel()
        training_args, finetuning_args = mock_args
        finetuning_args.galore_layerwise = True

        with patch.object(GaLoreOptimizerBuilder, '_log_optimizer_info'):
            optimizer = create_galore_optimizer(model, training_args, finetuning_args)

        assert isinstance(optimizer, DummyOptimizer)
        assert optimizer.optimizer_dict is not None


@pytest.mark.skipif(not is_apollo_available(), reason="APOLLO is not available")
class TestAPOLLOOptimizerBuilder:
    """Tests for the APOLLOOptimizerBuilder class."""

    @pytest.fixture
    def mock_args(self):
        """Create mock training and finetuning args."""
        training_args = MagicMock()
        training_args.optim = "adamw_torch"
        training_args.learning_rate = 1e-4
        training_args.weight_decay = 0.01
        training_args.gradient_accumulation_steps = 1

        finetuning_args = MagicMock()
        finetuning_args.apollo_target = ["linear1", "linear2"]
        finetuning_args.apollo_rank = 8
        finetuning_args.apollo_update_interval = 200
        finetuning_args.apollo_scale = 32.0
        finetuning_args.apollo_proj = "random"
        finetuning_args.apollo_proj_type = "std"
        finetuning_args.apollo_scale_type = "channel"
        finetuning_args.apollo_scale_front = False
        finetuning_args.apollo_layerwise = False
        finetuning_args.freeze_vision_tower = True

        return training_args, finetuning_args

    def test_get_targets_specific(self, mock_args):
        """Test getting specific target modules."""
        model = SimpleModel()
        training_args, finetuning_args = mock_args

        builder = APOLLOOptimizerBuilder(model, training_args, finetuning_args)
        targets = builder._get_targets()

        assert targets == ["linear1", "linear2"]

    def test_get_optimizer_kwargs(self, mock_args):
        """Test getting optimizer-specific kwargs."""
        model = SimpleModel()
        training_args, finetuning_args = mock_args

        builder = APOLLOOptimizerBuilder(model, training_args, finetuning_args)
        kwargs = builder._get_optimizer_kwargs()

        assert kwargs["rank"] == 8
        assert kwargs["update_proj_gap"] == 200
        assert kwargs["scale"] == 32.0
        assert kwargs["proj"] == "random"
        assert kwargs["proj_type"] == "std"
        assert kwargs["scale_type"] == "channel"
        assert kwargs["scale_front"] is False

    def test_get_optimizer_name(self, mock_args):
        """Test getting optimizer name."""
        model = SimpleModel()
        training_args, finetuning_args = mock_args

        builder = APOLLOOptimizerBuilder(model, training_args, finetuning_args)
        assert builder._get_optimizer_name() == "APOLLO"

    def test_build_standard_optimizer(self, mock_args):
        """Test building a standard (non-layerwise) optimizer."""
        model = SimpleModel()
        training_args, finetuning_args = mock_args

        with patch.object(APOLLOOptimizerBuilder, '_log_optimizer_info'):
            optimizer = create_apollo_optimizer(model, training_args, finetuning_args)

        assert optimizer is not None
        # Check that param groups are created
        assert len(optimizer.param_groups) == 3


class TestFactoryFunctions:
    """Tests for the factory functions."""

    def test_create_galore_optimizer_function_exists(self):
        """Test that create_galore_optimizer function is accessible."""
        assert callable(create_galore_optimizer)

    def test_create_apollo_optimizer_function_exists(self):
        """Test that create_apollo_optimizer function is accessible."""
        assert callable(create_apollo_optimizer)


class TestBackwardCompatibility:
    """Tests for backward compatibility with trainer_utils."""

    def test_dummy_optimizer_import_from_trainer_utils(self):
        """Test that DummyOptimizer can be imported from trainer_utils."""
        from llamafactory.train.trainer_utils import DummyOptimizer as DummyOptimizerFromUtils
        assert DummyOptimizerFromUtils is DummyOptimizer

    def test_create_galore_optimizer_in_trainer_utils(self):
        """Test that _create_galore_optimizer exists in trainer_utils."""
        from llamafactory.train.trainer_utils import _create_galore_optimizer
        assert callable(_create_galore_optimizer)

    def test_create_apollo_optimizer_in_trainer_utils(self):
        """Test that _create_apollo_optimizer exists in trainer_utils."""
        from llamafactory.train.trainer_utils import _create_apollo_optimizer
        assert callable(_create_apollo_optimizer)
