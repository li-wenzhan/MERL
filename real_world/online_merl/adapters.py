from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class RealRobotStep:
    image: Any
    action: Any
    task_instruction: str
    timestamp: float
    metadata: Dict[str, Any]


class OnlineMERLAdapter:
    """Placeholder interface for the later real-robot online MERL loop."""

    def __init__(self, world_model_predictor):
        self.world_model_predictor = world_model_predictor

    def predict_world_model(self, request):
        return self.world_model_predictor.predict(request)

