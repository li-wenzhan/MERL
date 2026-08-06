# Online MERL Real-Robot Integration

This directory is reserved for the later online MERL loop.

Stage 1 does not control a robot and does not run PPO. The only dependency to keep stable now is the world-model predictor:

```python
from real_world.world_model import RealWorldWMPredictor
```

The future online loop should feed camera frames, policy-proposed actions, and task text into that predictor, then write the result into replay or diagnostics.

