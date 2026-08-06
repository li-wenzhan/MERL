# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from importlib import import_module

from .base import BaseRollout
from .hf_rollout import HFRollout
from .naive import NaiveRollout

__all__ = [
    "BaseRollout",
    "NaiveRollout",
    "HFRollout",
    "RobHFRollout",
    "RobWMHFRollout",
    "RobWMHFRolloutPro",
]


def __getattr__(name):
    if name == "RobHFRollout":
        return import_module(".rob_rollout", __name__).RobHFRollout
    if name == "RobWMHFRollout":
        return import_module(".old.rob_rollout_wm", __name__).RobWMHFRollout
    if name == "RobWMHFRolloutPro":
        return import_module(".rob_rollout_wm_pro", __name__).RobWMHFRolloutPro
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
