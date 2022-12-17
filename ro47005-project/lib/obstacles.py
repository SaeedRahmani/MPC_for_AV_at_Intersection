from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Obstacle:
    type: str
    dim: List[float]
    pos: List[float]


@dataclass
class Scenario:
    start: Tuple[float, float, float]
    obstacles: List[Obstacle]
    goal_area: Obstacle
    goal_point: Tuple[float, float, float]


def add_obstacles_to_env(obstacles: List[Obstacle], env):
    for obstacle in obstacles:
        env.add_shapes(shape_type=obstacle.type, dim=obstacle.dim, poses_2d=[obstacle.pos])


def draw_obstacle(obstacle: Obstacle, ax, color=None):
    import matplotlib.pylab as plt
    from matplotlib.patches import Rectangle, Circle

    if obstacle.type == "GEOM_BOX":
        x, y = obstacle.pos[0], obstacle.pos[1]
        width, height, orientation = obstacle.dim[0], obstacle.dim[1], obstacle.dim[2]
        x, y = x - width / 2, y - height / 2
        # assert orientation == 0.
        ax.add_patch(Rectangle((x, y), width, height, edgecolor=color, facecolor='none'))
    elif obstacle.type == "GEOM_CYLINDER":
        radius = obstacle.dim[0]
        x, y = obstacle.pos[0], obstacle.pos[1]
        ax.add_patch(Circle((x, y), radius, edgecolor=color, facecolor='none'))


def check_collision(x: float, y: float, obstacle: Obstacle) -> bool:
    # TODO make it use proper methods
    if obstacle.type == "GEOM_BOX":
        xc, yc, _ = tuple(obstacle.pos)
        w, h, _ = tuple(obstacle.dim)
        x1, y1 = xc - w / 2, yc - h / 2
        x2, y2 = xc + w / 2, yc + h / 2
        return x1 <= x <= x2 and y1 <= y <= y2
    elif obstacle.type == "GEOM_CYLINDER":
        raise Exception("Not supported")


def obstacle_to_convex(obstacle: Obstacle, margin=None) -> np.ndarray:
    """
        This function represents each obstacle by a set of half planes.
        The half planes are given as: ax + by + c.
        When this is smaller than 0 for every half plane, then the point is inside the obstacle.

        :return np.ndarray[[a, b, c], ...]
    """
    if margin is None:
        margin = [0, 0]

    # Check if the rotation of the obstacle is zero
    assert obstacle.pos[2] == 0
    c_x, c_y = obstacle.pos[:2]

    # Implement a margin
    m_x, m_y = margin

    if obstacle.type == "GEOM_BOX":
        w_x, w_y = obstacle.dim[:2]
        return np.array([[1, 0, -(c_x + w_x/2 + m_x)],
                         [-1, 0, c_x - w_x/2 - m_x],
                         [0, 1, -(c_y + w_y/2 + m_y)],
                         [0, -1, c_y - w_y/2 - m_y]])
    elif obstacle.type == "GEOM_CYLINDER":
        r = obstacle.dim[0]
        off = r * np.tan(np.pi/8)
        return np.array([[1, 0, -(c_x + r + m_x)],
                         [-1, 0, c_x - r - m_x],
                         [0, 1, -(c_y + r + m_y)],
                         [0, -1, c_y - r - m_y],
                         [-1, 1, c_x - c_y - r * np.sqrt(2)],
                         [1, -1, -c_x + c_y - r * np.sqrt(2)],
                         [-1, -1, c_x + c_y - r * np.sqrt(2)],
                         [1, 1, -c_x - c_y - r * np.sqrt(2)]])
