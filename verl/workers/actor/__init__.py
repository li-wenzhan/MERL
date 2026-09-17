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

from .base import BasePPOActor

__all__ = ["BasePPOActor", "DataParallelPPOActor", "DataParallelPRIME","RobDataParallelPPOActor"]


def __getattr__(name):
    """Load only the requested actor and its optional backend dependencies."""
    from importlib import import_module

    modules = {"DataParallelPPOActor": ".dp_actor", "DataParallelPRIME": ".dp_prime",
               "RobDataParallelPPOActor": ".dp_rob"}
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
