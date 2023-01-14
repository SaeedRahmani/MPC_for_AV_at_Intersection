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


def load_motion_primitives(version="prius") -> Dict[str, MotionPrimitive]:
    if version == "prius":
        DIR = Path('data/motion_primitives_prius')
    elif version == "bicycle_model":
        DIR = Path('../data/motion_primitives_bicycle_model')
    elif version == "bicycle_model_real_size":
        DIR = Path('../data/motion_primitives_bicycle_model_real_size')
    else:
        raise"Motion primitives version not recognized!"

    mps = {}

    files = list(DIR.glob("*.pkl"))

    if len(files) == 0:
        raise Exception("No motion primitives found.")

    for filename in files:
        with open(filename, 'rb') as file:
            mp: MotionPrimitive = pickle.load(file)
            mps[mp.name] = mp

    return mps
