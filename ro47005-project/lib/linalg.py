import numpy as np


def create_2d_transform_mtx(x: float, y: float, theta: float) -> np.ndarray:
    """
    Creates a transform matrix that can translate points by (x, y) and rotate by theta.
    :param x: X axis offset to translate by
    :param y: Y axis offset to translate by
    :param theta: Angle in radians to rotate by
    :return:
    """
    return np.array([[np.cos(theta), -np.sin(theta), x],
                     [np.sin(theta), np.cos(theta), y],
                     [0, 0, 1]])


def transform_2d_pts(theta: float, transform_mtx: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Transform (i.e. rotate and translate) a set of points of type (x, y, theta)
    :param theta: The theta value (i.e. angle to rotate by)
    :param transform_mtx: Transformation matrix (i.e. result of create_2d_transform_mtx(x, y, theta) )
    :param points: A matrix of points of shape (N, 3), where each row is (x, y, theta).
    :return: Transformed matrix of points of shape (N, 3)
    """
    points2 = points.copy()
    points2[:, 2] = 1
    points2 = points2 @ transform_mtx.T
    points2[:, 2] = points[:, 2] + theta
    return points2
