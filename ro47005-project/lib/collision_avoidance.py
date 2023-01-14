from typing import List, Tuple, Union, Optional

import numpy as np

from lib.car_dimensions import CarDimensions
from lib.helpers import measure_time
from lib.trajectories import car_trajectory_to_collision_point_trajectories


def _combine_rowwise_repeat(arrays_2d: List[np.ndarray], repeats=1) -> Tuple[int, np.ndarray]:
    n = len(arrays_2d) * repeats
    arr = np.concatenate([*arrays_2d] * repeats, axis=1)
    a, b = arr.shape
    arr = arr.reshape((a * n, b // n))
    return n, arr


def _pad_trajectory(traj: np.ndarray, n_iterations: int) -> np.ndarray:
    if len(traj) < n_iterations:
        return np.vstack([traj, np.repeat(traj[-1:, :], n_iterations - len(traj), axis=0)])
    else:
        return traj[:n_iterations]


def _pad_trajectories(traj_agent: np.ndarray, trajs_o: List[np.ndarray]):
    n_iterations = max(len(traj_agent), max((len(tr) for tr in trajs_o)))
    traj_agent = _pad_trajectory(traj_agent, n_iterations)
    trajs_o = [_pad_trajectory(tr, n_iterations) for tr in trajs_o]
    return traj_agent, trajs_o


def _get_rowwise_diffs(car_dimensions, traj_agent: np.ndarray, traj_obstacles: List[np.ndarray]):
    n_circle_centers = len(car_dimensions.circle_centers)

    traj_agent, traj_obstacles = _pad_trajectories(traj_agent, traj_obstacles)

    rows_per_frame_ag, cc_pts_ag = _combine_rowwise_repeat(
        [tr[:, :2] for tr in car_trajectory_to_collision_point_trajectories(traj_agent, car_dimensions)], repeats=1)
    rows_per_frame_ag *= n_circle_centers * len(traj_obstacles)
    cc_pts_ag = np.repeat(cc_pts_ag, len(traj_obstacles) * n_circle_centers, axis=0)
    rows_per_frame_obs, cc_pts_obs = _combine_rowwise_repeat(
        [cc_tr[:, :2] for tr in traj_obstacles for cc_tr in
         car_trajectory_to_collision_point_trajectories(tr, car_dimensions)], repeats=n_circle_centers)
    assert rows_per_frame_ag == rows_per_frame_obs
    diff_pts = cc_pts_obs - cc_pts_ag
    print("Number of point pairs:", len(diff_pts))
    return rows_per_frame_ag, diff_pts


def _offset_trajectories_by_frames(trajs: List[np.ndarray], offsets: Union[List[int], np.ndarray]) -> List[np.ndarray]:
    out = []

    for traj in trajs:
        for idx_offset in offsets:
            if idx_offset < 0:
                obst2 = np.concatenate([traj[-idx_offset:], np.repeat(traj[-1:, :], repeats=-idx_offset, axis=0)],
                                       axis=0)
            elif idx_offset > 0:
                obst2 = np.concatenate([np.repeat(traj[0:1], repeats=idx_offset, axis=0), traj[:-idx_offset]], axis=0)
            else:
                obst2 = traj
            out.append(obst2)

    return out


def check_collision_moving_cars(car_dimensions: CarDimensions, traj_agent: np.ndarray,
                                traj_obstacles: List[np.ndarray], frame_window: int = 0) -> Optional[
    Tuple[float, float]]:
    offsets = np.array(range(-frame_window, frame_window + 1, 1))
    traj_obstacles = _offset_trajectories_by_frames(traj_obstacles, offsets=offsets)

    min_distance = 2 * car_dimensions.radius

    rows_per_frame, diff_pts = _get_rowwise_diffs(car_dimensions, traj_agent, traj_obstacles)

    diff_pts = np.linalg.norm(diff_pts, axis=1) <= min_distance
    # print("Total point pairs to collision-check:", len(diff_pts))
    first_row_idx = np.argmax(diff_pts)

    if not diff_pts[first_row_idx]:
        # no collision
        return None

    first_frame_idx = first_row_idx // rows_per_frame
    x, y = traj_agent[first_frame_idx, :2]
    return x, y


def cutoff_curve_by_position(points: np.ndarray, x: float, y: float, radius: float = 0.001) -> np.ndarray:
    points_diff = points[:, :2].copy()
    points_diff[:, 0] -= x
    points_diff[:, 1] -= y

    points_dist = np.linalg.norm(points_diff, axis=1) <= radius
    first_idx = np.argmax(points_dist)

    if not points_dist[first_idx]:
        # no cutoff
        return points

    return points[:first_idx]
