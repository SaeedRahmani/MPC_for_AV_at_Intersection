import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np


@dataclass
class MotionPrimitive:
    name: str
    forward_speed: float
    steering_angle: float
    n_seconds: float

    total_length: float = 0.
    """The length of the motion primitive as a whole (i.e. sum of all distances between consecutive points)"""

    points: np.ndarray = np.array([])
    """The motion primitive (i.e. array of points) itself"""


def load_motion_primitives() -> Dict[str, MotionPrimitive]:
    # DIR = Path('../data/motion_primitives_prius')
    DIR = Path('../data/motion_primitives_bicycle_model')

    mps = {}

    for filename in DIR.glob("*.pkl"):
        with open(filename, 'rb') as file:
            mp: MotionPrimitive = pickle.load(file)
            mps[mp.name] = mp

    return mps
