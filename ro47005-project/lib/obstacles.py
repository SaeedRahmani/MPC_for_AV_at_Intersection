from dataclasses import dataclass
from typing import List, Tuple, Optional

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


def draw_point_arrow(point: Tuple[float, float, float], ax, color=None):
    x, y, theta = point
    u = np.cos(theta)
    v = np.sin(theta)
    ax.quiver(x, y, u, v, color=color)


def check_collision(obstacle_halfplanes: np.ndarray, points: np.ndarray) -> bool:
    """
    Checks whether a single obstacle (defined by a union of half-planes) collides with ANY (i.e. at least one) point
    from the set of points
    :param obstacle_halfplanes: Matrix of shape (M, 3) where each row are the [a, b, c] coordinates of a halfplane
    :param points: Matrix of shape (2, N) where each column is a point in 2D with [x, y] coordinates
    :return:
    """

    n_halfplanes, n_hp_coords = obstacle_halfplanes.shape
    assert n_hp_coords == 3

    n_p_coords, n_points = points.shape
    assert n_p_coords == 2

    points = np.vstack([points, np.ones((n_points,))])
    result_all = (obstacle_halfplanes @ points) <= 0
    result_per_point = np.all(result_all, axis=0)

    return bool(np.any(result_per_point))


def obstacle_to_convex(obstacle: Obstacle, margin: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """
        This function represents each obstacle by a set of half planes.
        The half planes are given as: ax + by + c.
        When this is smaller than 0 for every half plane, then the point is inside the obstacle.

        :return np.ndarray[[a, b, c], ...]
    """
    if margin is None:
        margin = (0, 0)

    # Check if the rotation of the obstacle is zero
    assert obstacle.pos[2] == 0
    c_x, c_y = obstacle.pos[:2]

    # Implement a margin
    m_x, m_y = margin

    if obstacle.type == "GEOM_BOX":
        w_x, w_y = obstacle.dim[:2]
        return np.array([[1, 0, -(c_x + w_x / 2 + m_x)],  # right
                         [-1, 0, c_x - w_x / 2 - m_x],  # left
                         [0, 1, -(c_y + w_y / 2 + m_y)],  # top
                         [0, -1, c_y - w_y / 2 - m_y]])  # bottom
    elif obstacle.type == "GEOM_CYLINDER":
        r = obstacle.dim[0]
        off = r * np.tan(np.pi / 8)
        return np.array([[1, 0, -(c_x + r + m_x)],  # right
                         [-1, 0, c_x - r - m_x],  # left
                         [0, 1, -(c_y + r + m_y)],  # top
                         [0, -1, c_y - r - m_y],  # bottom
                         [-1, 1, c_x - c_y - r * np.sqrt(2) - m_x - m_y],  # top left
                         [1, -1, -c_x + c_y - r * np.sqrt(2) - m_x - m_y],  # bottom right
                         [-1, -1, c_x + c_y - r * np.sqrt(2) - m_x - m_y],  # bottom left
                         [1, 1, -c_x - c_y - r * np.sqrt(2) - m_x - m_y]])  # top right
