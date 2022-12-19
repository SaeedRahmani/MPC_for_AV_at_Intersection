from abc import ABC, abstractmethod
from typing import Tuple, Optional

import numpy as np


class Obstacle(ABC):

    @abstractmethod
    def add_to_bullet_env(self, env):
        """
        This method adds the obstacle into your PyBullet simulation environment
        :param env: The PyBullet simulation environment
        """
        pass

    @abstractmethod
    def draw(self, ax, color=None):
        """
        This method draws the obstacle into matplotlib pyplot axes
        :param ax: matplotlib.pyplot axes
        :param color: Color argument for pyplot
        """
        pass

    @abstractmethod
    def to_convex(self, margin: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
            This function represents each obstacle by a set of half planes.
            The half planes are given as: ax + by + c.
            When this is smaller than 0 for every half plane, then the point is inside the obstacle.

            :return: np.ndarray[[a, b, c], ...]
        """
        pass


class BoxObstacle(Obstacle):
    def __init__(self, dim: Tuple[float, float, float], xy: Tuple[float, float]):
        self._type = "GEOM_BOX"
        self._dim = list(dim)
        self._pos = [*xy, 0]

    def add_to_bullet_env(self, env):
        env.add_shapes(shape_type=self._type, dim=self._dim, poses_2d=[self._pos])

    def draw(self, ax, color=None):
        from matplotlib.patches import Rectangle

        x, y = self._pos[0], self._pos[1]
        width, height, orientation = self._dim[0], self._dim[1], self._dim[2]
        x, y = x - width / 2, y - height / 2
        # assert orientation == 0.
        ax.add_patch(Rectangle((x, y), width, height, edgecolor=color, facecolor='none'))

    def to_convex(self, margin: Optional[Tuple[float, float]] = None) -> np.ndarray:
        if margin is None:
            margin = (0, 0)

        # Check if the rotation of the obstacle is zero
        assert self._pos[2] == 0
        c_x, c_y = self._pos[:2]

        # Implement a margin
        m_x, m_y = margin

        w_x, w_y = self._dim[:2]
        return np.array([[1, 0, -(c_x + w_x / 2 + m_x)],  # right
                         [-1, 0, c_x - w_x / 2 - m_x],  # left
                         [0, 1, -(c_y + w_y / 2 + m_y)],  # top
                         [0, -1, c_y - w_y / 2 - m_y]])  # bottom


class CircleObstacle(Obstacle):

    def __init__(self, radius: float, height: float, xy: Tuple[float, float]):
        self._type = "GEOM_CYLINDER"
        self._dim = [radius, height]
        self._pos = [*xy, 0]

    def add_to_bullet_env(self, env):
        env.add_shapes(shape_type=self._type, dim=self._dim, poses_2d=[self._pos])

    def draw(self, ax, color=None):
        from matplotlib.patches import Circle
        radius = self._dim[0]
        x, y = self._pos[0], self._pos[1]
        ax.add_patch(Circle((x, y), radius, edgecolor=color, facecolor='none'))

    def to_convex(self, margin: Optional[Tuple[float, float]] = None) -> np.ndarray:
        if margin is None:
            margin = (0, 0)

        # Check if the rotation of the obstacle is zero
        assert self._pos[2] == 0
        c_x, c_y = self._pos[:2]

        # Implement a margin
        m_x, m_y = margin

        r = self._dim[0]
        off = r * np.tan(np.pi / 8)
        return np.array([[1, 0, -(c_x + r + m_x)],  # right
                         [-1, 0, c_x - r - m_x],  # left
                         [0, 1, -(c_y + r + m_y)],  # top
                         [0, -1, c_y - r - m_y],  # bottom
                         [-1, 1, c_x - c_y - r * np.sqrt(2) - m_x - m_y],  # top left
                         [1, -1, -c_x + c_y - r * np.sqrt(2) - m_x - m_y],  # bottom right
                         [-1, -1, c_x + c_y - r * np.sqrt(2) - m_x - m_y],  # bottom left
                         [1, 1, -c_x - c_y - r * np.sqrt(2) - m_x - m_y]])  # top right


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
