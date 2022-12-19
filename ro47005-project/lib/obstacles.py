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
    def __init__(self, xy_width: Tuple[float, float], height: float, xy_center: Tuple[float, float]):
        self.xy_width = xy_width
        self.height = height
        self.xy_center = xy_center

        c_x, c_y = xy_center
        w_x, h_x = xy_width
        self.xy1 = c_x - w_x / 2, c_y - h_x / 2
        self.xy2 = c_x + w_x / 2, c_y + h_x / 2

        def _setattr(self, key, value):
            raise Exception("BoxObstacle objects are read-only")

        self.__setattr__ = _setattr

    def add_to_bullet_env(self, env):
        env.add_shapes(shape_type="GEOM_BOX", dim=[*self.xy_width, self.height], poses_2d=[[*self.xy_center, 0]])

    def draw(self, ax, color=None):
        # TODO: this assumes that the rectangle is not rotated
        from matplotlib.patches import Rectangle

        w_x, w_h = self.xy_width
        ax.add_patch(Rectangle(self.xy1, w_x, w_h, edgecolor=color, facecolor='none'))

    def to_convex(self, margin: Optional[Tuple[float, float]] = None) -> np.ndarray:
        # TODO: this assumes that the rectangle is not rotated
        if margin is None:
            margin = (0, 0)

        # Check if the rotation of the obstacle is zero
        # assert self._pos[2] == 0

        # Implement a margin
        m_x, m_y = margin

        x1, y1 = self.xy1
        x2, y2 = self.xy2
        return np.array([[1, 0, -(x2 + m_x)],  # right
                         [-1, 0, x1 - m_x],  # left
                         [0, 1, -(y2 + m_y)],  # top
                         [0, -1, y1 - m_y]])  # bottom


class CircleObstacle(Obstacle):

    def __init__(self, radius: float, height: float, xy_center: Tuple[float, float]):
        self.radius = radius
        self.height = height
        self.xy_center = xy_center

        def _setattr(self, key, value):
            raise Exception("CircleObstacle objects are read-only")

        self.__setattr__ = _setattr

    def add_to_bullet_env(self, env):
        env.add_shapes(shape_type="GEOM_CYLINDER", dim=[self.radius, self.height], poses_2d=[[*self.xy_center, 0]])

    def draw(self, ax, color=None):
        from matplotlib.patches import Circle
        ax.add_patch(Circle(self.xy_center, self.radius, edgecolor=color, facecolor='none'))

    def to_convex(self, margin: Optional[Tuple[float, float]] = None) -> np.ndarray:
        if margin is None:
            margin = (0, 0)

        # Check if the rotation of the obstacle is zero
        # assert self._pos[2] == 0

        # Implement a margin
        m_x, m_y = margin

        c_x, c_y = self.xy_center
        r = self.radius
        # off = r * np.tan(np.pi / 8)
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
