from typing import Tuple

import numpy as np


def draw_point_arrow(point: Tuple[float, float, float], ax, color=None):
    x, y, theta = point
    u = np.cos(theta)
    v = np.sin(theta)
    ax.quiver(x, y, u, v, color=color)
