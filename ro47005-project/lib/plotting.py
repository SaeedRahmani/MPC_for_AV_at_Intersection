import math
from typing import Tuple

import numpy as np
from matplotlib.collections import EllipseCollection
from matplotlib.patches import Rectangle, Circle

from lib.car_dimensions import CarDimensions
from lib.linalg import create_2d_transform_mtx, transform_2d_pts


def draw_point_arrow(point: Tuple[float, float, float], ax, color=None):
    x, y, theta = point
    u = np.cos(theta)
    v = np.sin(theta)
    ax.quiver(x, y, u, v, color=color)


def draw_circles(points: np.ndarray, radius: float, ax, color=None):
    ax.add_collection(
        EllipseCollection(widths=radius * 2, heights=radius * 2, angles=0, units='xy', edgecolors=color, facecolors='none', offsets=points[:, :2],
                          offset_transform=ax.transData))
    ax.scatter(points[:, 0], points[:, 1], color=color, marker='+')


def draw_car(start: Tuple[float, float, float], car_dimensions: CarDimensions, ax, color='b'):
    width, length = car_dimensions.bounding_box_size

    c_x, c_y, theta = start
    point_rect_bl = c_x - length / 2, c_y - width / 2
    ax.add_patch(Rectangle(point_rect_bl, length, width, rotation_point='center', edgecolor=color, facecolor='none',
                           angle=theta * 180. / math.pi))

    # transform all circle centers to world space
    circle_center_mtx = create_2d_transform_mtx(*start)
    circle_centers = transform_2d_pts(start[2], circle_center_mtx, car_dimensions.circle_centers)

    ax.scatter(circle_centers[:, 0], circle_centers[:, 1], color=color)