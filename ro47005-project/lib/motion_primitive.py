from dataclasses import dataclass

import numpy as np


@dataclass
class MotionPrimitive:
    name: str
    forward_speed: float
    steering_angle: float
    n_seconds: float
    total_length: float = 0.
    points: np.ndarray = np.array([])
