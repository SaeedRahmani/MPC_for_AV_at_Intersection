import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Iterable

import numpy as np

from lib.a_star import AStar
from lib.math import normalize_angle
from lib.obstacles import Scenario, check_collision, obstacle_to_convex


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


class MotionPrimitiveSearch:
    def __init__(self, scenario: Scenario, mps: Dict[str, MotionPrimitive], margin: Tuple[float, float]):
        self._scenario = scenario
        self._mps = mps
        self._points_map: Dict[Tuple[NodeType, NodeType], np.ndarray] = {}

        self._goal_area_hp = obstacle_to_convex(self._scenario.goal_area, margin=margin)
        self._obstacles_hp: List[np.ndarray] = [obstacle_to_convex(o, margin=margin) for o in self._scenario.obstacles]
        self._gx, self._gy, self._gtheta = self._scenario.goal_point

        self._a_star: AStar[NodeType] = AStar(neighbor_function=self.neighbor_function)

    def run(self, debug=False) -> Tuple[float, np.ndarray]:
        cost, path = self._a_star.run(self._scenario.start, is_goal_function=self.is_goal,
                                      heuristic_function=self.distance_to_goal, debug=debug)
        trajectory = self.path_to_full_trajectory(path)
        return cost, trajectory

    @property
    def debug_data(self):
        return self._a_star.debug_data

    def is_goal(self, node: Tuple[float, float, float]) -> bool:
        x, y, theta = node

        result = check_collision(self._goal_area_hp, np.atleast_2d([x, y]).T) \
                 and abs(theta - self._gtheta) <= np.pi / 8

        return result

    def distance_to_goal(self, node: Tuple[float, float, float]) -> float:
        x, y, theta = node
        return np.sqrt((self._gx - x) ** 2 + (self._gy - y) ** 2) + 2.425 * abs(theta - self._gtheta)

    def neighbor_function(self, node: NodeType) -> Iterable[Tuple[float, NodeType]]:
        mtx = create_transform_mtx(*node)

        for mp in self._mps.values():
            mp_points = transform_pts(node[2], mtx, mp.points)
            mp_points_xy = mp_points[:, :2].T

            # checks collision with every obstacle
            # since it is not a list but an iterator, it doesn't check all of them if it doesn't have to,
            # but breaks once the first colliding obstacle is found.
            # Important: If the () parentheses were replaced with [] in the line below, then it would check all,
            # which we don't want.
            #
            # If no colliding obstacles are found, then it will have checked all of them.
            collides = any((check_collision(o, mp_points_xy) for o in self._obstacles_hp))

            if not collides:
                # we can yield this obstacle because it has now been properly collision-checked

                x, y, theta = tuple(mp_points[-1, :].tolist())
                neighbor = x, y, normalize_angle(theta)

                cost = mp.total_length

                self._points_map[node, neighbor] = mp_points

                yield cost, neighbor

    def path_to_full_trajectory(self, path: List[NodeType]) -> np.ndarray:
        points: List[np.ndarray] = []

        for p1, p2 in zip(path[:-1], path[1:]):
            this_points = self._points_map[p1, p2]
            points.append(this_points)

        return np.concatenate(points, axis=0)
