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

"""Provide Muon optimizer for momentum-orthogonalized training.

This module contains the Muon optimizer implementation, which combines SGD
momentum with Newton-Schulz orthogonalization to achieve better convergence
properties for training large language models.

Key Classes:
    Muon: Main optimizer class that internally uses SGD-momentum with
        orthogonalization post-processing for 2D parameters, with AdamW
        fallback for other parameters.

Example:
    Create and use Muon optimizer::

        from llamafactory.third_party.muon import Muon

        optimizer = Muon(
            lr=0.02,
            muon_params=model.get_2d_params(),
            adamw_params=model.get_other_params()
        )

        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

See Also:
    llamafactory.third_party.muon.muon: Implementation details.
"""

from .muon import Muon


__all__ = ["Muon"]
