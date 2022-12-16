import pickle
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable, Callable
from pathlib import Path

import numpy as np

from lib.obstacles import Obstacle, Scenario


@dataclass
class MotionPrimitive:
    name: str
    forward_speed: float
    steering_angle: float
    n_seconds: float
    total_length: float = 0.
    points: np.ndarray = np.array([])


def load_motion_primitives() -> Dict[str, MotionPrimitive]:
    DIR = Path('../data/motion_primitives')

    mps = {}

    for filename in DIR.glob("*.pkl"):
        print(filename)
        with open(filename, 'rb') as file:
            mp: MotionPrimitive = pickle.load(file)
            mps[mp.name] = mp

    return mps


NodeType = Tuple[float, float, float]


def create_transform_mtx(x: float, y: float, theta: float) -> np.ndarray:
    return np.array([[np.cos(theta), -np.sin(theta), x],
                     [np.sin(theta), np.cos(theta), y],
                     [0, 0, 1]])


def transform_pts(theta: float, mtx: np.ndarray, points: np.ndarray) -> np.ndarray:
    points2 = points.copy()
    points2[:, 2] = 1
    points2 = points2 @ mtx.T
    points2[:, 2] = points[:, 2] + theta
    return points2


class MotionPrimitiveSearchMetadata:
    def __init__(self, scenario: Scenario, mps: Dict[str, MotionPrimitive]):
        self.scenario = scenario
        self.mps = mps
        self.points_map: Dict[Tuple[NodeType, NodeType], np.ndarray] = {}

    def mp_neighbor_function(self, node: NodeType) -> Iterable[Tuple[float, NodeType]]:
        mtx = create_transform_mtx(*node)

        for mp in self.mps.values():
            points = transform_pts(node[2], mtx, mp.points)

            neighbor = tuple(points[-1, :].tolist())
            cost = mp.total_length

            self.points_map[node, neighbor] = points

            yield cost, neighbor

    def augment_path(self, path: List[NodeType]) -> np.ndarray:
        points: List[np.ndarray] = []

        for p1, p2 in zip(path[:-1], path[1:]):
            this_points = self.points_map[p1, p2]
            points.append(this_points)

        return np.concatenate(points, axis=0)
