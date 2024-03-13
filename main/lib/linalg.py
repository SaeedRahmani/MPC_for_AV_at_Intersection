import numpy as np


def create_2d_transform_mtx(x: float, y: float, theta: float) -> np.ndarray:
    """
    Creates a transform matrix that can translate points by (x, y) and rotate by theta.
    :param x: X axis offset to translate by
    :param y: Y axis offset to translate by
    :param theta: Angle in radians to rotate by
    :return:
    """

    if x == 0 and y == 0:
        # rotate only (2, 2) matrix
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]
                 ])

    # rotate and translate (3, 3) matrix
    return np.array([[np.cos(theta), -np.sin(theta), x],
                     [np.sin(theta), np.cos(theta), y],
                     [0, 0, 1]])


def transform_2d_pts(theta: float, transform_mtx: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Transform (i.e. rotate and translate) a set of points of type (x, y, theta)
    :param theta: The theta value (i.e. angle to rotate by)
    :param transform_mtx: Transformation matrix (i.e. result of create_2d_transform_mtx(x, y, theta) )
    :param points: A matrix of points of shape (N, 2) or (N, 3), where each row is (x, y, [theta]).
    :return: Transformed matrix of points of the same shape as points
    """

    points_no_theta = points[:, :2]

    # perform transformation
    if transform_mtx.shape == (2, 2):
        # rotate only (2, 2) matrix
        points_no_theta = points_no_theta @ transform_mtx.T
    elif transform_mtx.shape == (3, 3):
        # rotate and translate (3, 3) matrix
        points_no_theta = np.append(points_no_theta, np.ones((points.shape[0], 1)), axis=1)
        points_no_theta = (points_no_theta @ transform_mtx.T)[:, :2]
    else:
        raise RuntimeError()

    if points.shape[1] == 3:
        # has theta
        return np.append(points_no_theta, points[:, 2:] + theta, axis=1)
    elif points.shape[1] == 2:
        # does not have theta
        return points_no_theta
    else:
        raise RuntimeError()



