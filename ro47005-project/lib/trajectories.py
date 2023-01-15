import heapq
from enum import Enum
from typing import List, Union

import numpy as np

from lib.car_dimensions import CarDimensions
from lib.simulation import State


def shift_car_trajectory_by_objspace_offset(trajectory: np.ndarray,
                                            x_offset: float, y_offset: float) -> np.ndarray:
    """
    Given a car trajectory (world-space points [x, y, theta]) and an offset point (object-space for the car),
    offsets each point within the trajectory by the offset in the direction of the theta.

    This is useful for example to convert a trajectory that tracks the center of the vehicle to a trajectory
    that tracks the back position of the vehicle.

    :param trajectory: NumPy array of shape (N, 3), where N is the no. of points and each row is [x, y, theta].
    :param x_offset: offset for the x axis in object space (for each point on the trajectory)
    :param y_offset: offset for the y axis in object space (for each point on the trajectory)
    :return: NumPy array of shape (N, 3)
    """
    # get point thetas
    thetas = trajectory[:, 2]
    # rotate by each individual theta
    offset_trajectory = np.vstack([
        np.cos(thetas) * x_offset - np.sin(thetas) * y_offset,
        np.sin(thetas) * x_offset + np.cos(thetas) * y_offset
    ]).T
    # offset by the original points' positions
    offset_trajectory += trajectory[:, :2]

    # add back the theta column
    offset_trajectory = np.append(offset_trajectory, np.atleast_2d(thetas).T, axis=1)
    return offset_trajectory


def car_trajectory_to_collision_point_trajectories(trajectory: np.ndarray, car_dimensions: CarDimensions) -> List[np.ndarray]:
    """
    Given a car trajectory and car dimensions, converts the trajectory to trajectories for each of the collision circles
    (for collision checking of the car)
    :param trajectory: NumPy array of shape (N, 3), where N is the no. of points and each row is [x, y, theta].
    :param car_dimensions: An object of class CarDimensions, providing information about the dimensions of the car
    :return: A list of world-space trajectories -- one for each of the circle centers of the car
    """
    offset_trajectories = []

    # for each circle center:
    for cc in car_dimensions.circle_centers:
        offset_trajectory = shift_car_trajectory_by_objspace_offset(trajectory, cc[0], cc[1])
        offset_trajectories.append(offset_trajectory)

    return offset_trajectories


def resample_curve(points: np.ndarray, dl: Union[float, np.ndarray],
                   keep_last_point: bool = True) -> np.ndarray:
    """
    For a trajectory represented as a set of points, filter the points such that in the filtered representation,
    all adjacent points have at least `min_adjacent_distance` between them. Always keeps the first point.
    :param points: (N, 2) or (N, 3) or (N, 4) matrix of points
    :param dl: The minimum distance from the previous point to keep it.
    :param keep_last_point: If true, always keeps the last point. (default: True)
    :return: Filtered points matrix
    """

    assert 2 <= points.shape[1]

    # compute distances between adjacent points
    step_dists = np.linalg.norm(points[1:, :2] - points[:-1, :2], axis=1)
    # add missing first zero back
    step_dists = np.append(0., step_dists)

    step_dists = np.floor(step_dists.cumsum() / dl).astype(int)
    mask = (step_dists[1:] - step_dists[:-1]) >= 1.  # where there is the step in value, we say True

    # add missing first boolean back (and make it always True)
    mask = np.append(True, mask)

    if keep_last_point:
        # make last boolean always True
        mask[-1] = True

    return points[mask].copy()


def calc_nearest_index(state: State, cx: np.ndarray, cy: np.ndarray, start_index: int = 0) -> int:
    dx = cx[start_index:] - state.x
    dy = cy[start_index:] - state.y

    d = dx ** 2 + dy ** 2
    if len(d) > 0:
        return np.argmin(d) + start_index
    else:
        return start_index


def calc_nearest_index_in_direction(state: State, cx: np.ndarray, cy: np.ndarray, start_index: int = 0, forward: bool = True) -> int:
    dist = np.linalg.norm([cx[start_index:] - state.x, cy[start_index:] - state.y], axis=0)

    if len(dist) >= 3:
        if len(dist) > 3:
            ind = np.argpartition(dist, 3)[:3]
            order = np.argsort(dist[ind])
            ind = ind[order]
        else:
            ind = np.argsort(dist)

        if np.abs(ind[1] - ind[2]) == 2:
            return ind[0] + start_index

        if np.abs(ind[0] - ind[1]) == 1:
            if forward:
                return max(ind[0], ind[1]) + start_index
            else:
                return min(ind[0], ind[1]) + start_index

        raise Exception("something wrong")

    if len(dist) == 2:
        return 1 + start_index if forward else start_index

    if len(dist) <= 1:
        return start_index
