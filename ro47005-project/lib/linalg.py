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


def filter_trajectory_min_distance_points(points: np.ndarray, min_adjacent_distance: float,
                                          keep_last_point: bool = True) -> np.ndarray:
    """
    For a trajectory represented as a set of points, filter the points such that in the filtered representation,
    all adjacent points have at least `min_adjacent_distance` between them. Always keeps the first point.
    :param points: (N, 2) or (N, 3) matrix of points
    :param min_adjacent_distance: The minimum distance from the previous point to keep it.
    :param keep_last_point: If true, always keeps the last point. (default: True)
    :return: Filtered points matrix
    """

    assert 2 <= points.shape[1] <= 3

    # compute distances between adjacent points
    step_dists = np.linalg.norm(points[1:, :2] - points[:-1, :2], axis=1)
    # add missing first zero back
    step_dists = np.append(0., step_dists)

    step_dists = np.floor(step_dists.cumsum() / min_adjacent_distance).astype(int)
    mask = (step_dists[1:] - step_dists[:-1]) >= 1.  # where there is the step in value, we say True

    # add missing first boolean back (and make it always True)
    mask = np.append(True, mask)

    if keep_last_point:
        # make last boolean always True
        mask[-1] = True

    return points[mask].copy()



