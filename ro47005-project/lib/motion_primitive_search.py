from typing import Dict, Tuple, List, Iterable

import numpy as np

from lib.a_star import AStar
from lib.linalg import create_2d_transform_mtx, transform_2d_pts
from lib.math import normalize_angle
from lib.motion_primitive import MotionPrimitive
from lib.obstacles import check_collision
from lib.scenario import Scenario

NodeType = Tuple[float, float, float]


class MotionPrimitiveSearch:
    def __init__(self, scenario: Scenario, mps: Dict[str, MotionPrimitive], margin: Tuple[float, float]):
        self._mps = mps
        self._points_map: Dict[Tuple[NodeType, NodeType], np.ndarray] = {}

        self._start = scenario.start
        self._goal_area = scenario.goal_area
        self._allowed_goal_theta_difference = scenario.allowed_goal_theta_difference
        self._goal_area_hp = self._goal_area.to_convex(margin=(0., 0.))  # goal area already defines its own margin!
        self._obstacles_hp: List[np.ndarray] = [o.to_convex(margin=margin) for o in scenario.obstacles]
        self._gx, self._gy, self._gtheta = scenario.goal_point

        self._a_star: AStar[NodeType] = AStar(neighbor_function=self.neighbor_function)

    def run(self, debug=False) -> Tuple[float, np.ndarray]:
        cost, path = self._a_star.run(self._start, is_goal_function=self.is_goal,
                                      heuristic_function=self.distance_to_goal, debug=debug)
        trajectory = self.path_to_full_trajectory(path)
        return cost, trajectory

    @property
    def debug_data(self):
        return self._a_star.debug_data

    def is_goal(self, node: Tuple[float, float, float]) -> bool:
        _, _, theta = node
        result = self._goal_area.distance_to_point(node[:2]) <= 1e-5 \
                 and abs(theta - self._gtheta) <= self._allowed_goal_theta_difference

        return result

    def distance_to_goal(self, node: Tuple[float, float, float]) -> float:
        x, y, theta = node
        distance_xy = self._goal_area.distance_to_point(node[:2])
        distance_theta = max(0., abs(theta - self._gtheta) - self._allowed_goal_theta_difference)
        return distance_xy + 2.7 * distance_theta

    def neighbor_function(self, node: NodeType) -> Iterable[Tuple[float, NodeType]]:
        mtx = create_2d_transform_mtx(*node)

        for mp in self._mps.values():
            mp_points = transform_2d_pts(node[2], mtx, mp.points)
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
