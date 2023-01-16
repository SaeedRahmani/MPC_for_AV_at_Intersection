from abc import ABC, abstractmethod
from typing import Tuple

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
    def to_convex(self, margin: float = 0.) -> np.ndarray:
        """
            This function represents each obstacle by a set of half planes.
            The half planes are given as: ax + by + c.
            When this is smaller than 0 for every half plane, then the point is inside the obstacle.

            :return: np.ndarray[[a, b, c], ...]
        """
        pass

    @abstractmethod
    def distance_to_point(self, point: Tuple[float, float]) -> float:
        pass

    @property
    @abstractmethod
    def hidden(self):
        pass


class BoxObstacle(Obstacle):
    def __init__(self, xy_width: Tuple[float, float], height: float, xy_center: Tuple[float, float],
                 hidden: bool = False):
        self.xy_width = xy_width
        self.height = height
        self.xy_center = xy_center

        c_x, c_y = xy_center
        w_x, h_x = xy_width
        self.xy1 = c_x - w_x / 2, c_y - h_x / 2
        self.xy2 = c_x + w_x / 2, c_y + h_x / 2
        self._hidden = hidden

        def _setattr(self, key, value):
            raise Exception("BoxObstacle objects are read-only")

        self.__setattr__ = _setattr

    @property
    def hidden(self):
        return self._hidden

    def add_to_bullet_env(self, env):
        env.add_shapes(shape_type="GEOM_BOX", dim=[*self.xy_width, self.height], poses_2d=[[*self.xy_center, 0]])

    def draw(self, ax, color=None, hidden_color='None'):
        # TODO: this assumes that the rectangle is not rotated
        from matplotlib.patches import Rectangle

        if self.hidden:
            color = hidden_color

        w_x, w_h = self.xy_width
        ax.add_patch(Rectangle(self.xy1, w_x, w_h, edgecolor=color, facecolor=color))

    def to_convex(self, margin: float = 0.) -> np.ndarray:
        # TODO: this assumes that the rectangle is not rotated

        # Check if the rotation of the obstacle is zero
        # assert self._pos[2] == 0

        x1, y1 = self.xy1
        x2, y2 = self.xy2
        return np.array([[1, 0, -(x2 + margin)],  # right
                         [-1, 0, x1 - margin],  # left
                         [0, 1, -(y2 + margin)],  # top
                         [0, -1, y1 - margin]])  # bottom

    def distance_to_point(self, point: Tuple[float, float]) -> float:
        # TODO: this assumes that the rectangle is not rotated
        x1, y1 = self.xy1
        x2, y2 = self.xy2
        x, y = point

        dx = max(x1 - x, 0, x - x2)
        dy = max(y1 - y, 0, y - y2)
        return np.sqrt(dx * dx + dy * dy)


class CircleObstacle(Obstacle):

    def __init__(self, radius: float, height: float, xy_center: Tuple[float, float], hidden: bool = False):
        self.radius = radius
        self.height = height
        self.xy_center = xy_center
        self._hidden = hidden

        def _setattr(self, key, value):
            raise Exception("CircleObstacle objects are read-only")

        self.__setattr__ = _setattr

    @property
    def hidden(self):
        return self._hidden

    def add_to_bullet_env(self, env):
        env.add_shapes(shape_type="GEOM_CYLINDER", dim=[self.radius, self.height], poses_2d=[[*self.xy_center, 0]])

    def draw(self, ax, color=None, hidden_color='None'):
        from matplotlib.patches import Circle

        if self.hidden:
            color = hidden_color

        ax.add_patch(Circle(self.xy_center, self.radius, edgecolor=color, facecolor=color))

    def to_convex(self, margin: float = 0.) -> np.ndarray:
        # Check if the rotation of the obstacle is zero
        # assert self._pos[2] == 0

        c_x, c_y = self.xy_center
        r = self.radius
        # off = r * np.tan(np.pi / 8)
        return np.array([[1, 0, -(c_x + r + margin)],  # right
                         [-1, 0, c_x - r - margin],  # left
                         [0, 1, -(c_y + r + margin)],  # top
                         [0, -1, c_y - r - margin],  # bottom
                         [-1, 1, c_x - c_y - r * np.sqrt(2) - 2 * margin],  # top left
                         [1, -1, -c_x + c_y - r * np.sqrt(2) - 2 * margin],  # bottom right
                         [-1, -1, c_x + c_y - r * np.sqrt(2) - 2 * margin],  # bottom left
                         [1, 1, -c_x - c_y - r * np.sqrt(2) - 2 * margin]])  # top right

    def distance_to_point(self, point: Tuple[float, float]) -> float:
        p_x, p_y = point
        r_x, r_y = self.xy_center

        return max(0, np.sqrt((r_x - p_x) ** 2 + (r_y - p_y) ** 2) - self.radius)


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
