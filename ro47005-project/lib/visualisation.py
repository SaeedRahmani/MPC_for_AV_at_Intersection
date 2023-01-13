import numpy as np
from typing import Callable, List, Tuple
import matplotlib.pyplot as plt

width_car = 1  # m
length_car = 2  # m


def R_box(theta: float) -> np.ndarray:
    """ The input should be in radians. Counterclockwise from the positive x-axis. """
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


def rad_to_deg(rad: float) -> float:
    return rad * 180 / np.pi


def create_animation(ax, positions_car=None, positions_obstacles=None, draw_car=True, draw_car_center=True,
                     draw_obstacles=True, draw_obstacles_center=True) -> Callable[[int], Tuple]:
    """
    Function that returns the animation function that is needed for matplotlib.animation.FuncAnimation.
    :param ax: The matplotlib object which contains the plot
    :param positions_car: np.ndarray with length steps of the simulation and values [x, y, theta, ...]
    :param positions_obstacles: np.ndarray with length steps of the simulation and values [x, y, v, theta, ...]; x, y and theta are used

    :return: function for matplotlib.animation.FuncAnimation
    """

    if draw_car or draw_car_center:
        if positions_car is None:
            raise "You should give the array with the positions of the car!"
        car_0 = positions_car[0, :2]

    car_point = None
    if draw_car_center:
        # init the car center point
        car_point, = ax.plot(*car_0, 'o')
        car_point.set_zorder(10)

    car_patch = None
    if draw_car:
        # init the car rectangle
        theta_car_0 = positions_car[0, 2]
        car_box_0 = car_0 + R_box(theta_car_0) @ np.array([-length_car / 2, -width_car / 2])
        car_patch = plt.Rectangle((0, 0), width=length_car, height=width_car, color='r')
        car_patch.xy = car_box_0
        car_patch.angle = rad_to_deg(theta_car_0)
        car_patch.set_zorder(9)
        ax.add_patch(car_patch)

    if draw_obstacles or draw_obstacles_center:
        if positions_obstacles is None:
            raise "You should give the array with the positions of the obstacles!"

    obs_points = None
    if draw_obstacles_center:
        # init the obstacle center points
        x_obs_o = positions_obstacles[:, 0, 0]
        y_obs_o = positions_obstacles[:, 0, 1]
        obs_points = ax.scatter(x_obs_o, y_obs_o)
        obs_points.set_zorder(8)

    obs_patches = None
    if draw_obstacles:
        obs_patches = []
        # init the obstacles rectangles
        for i in range(len(positions_obstacles)):
            theta_obs = positions_obstacles[i, 0, 3]
            obs_patch = plt.Rectangle((0, 0), width=length_car, height=width_car, color='c')
            obs_patch.xy, obs_patch.angle = patch_value(positions_obstacles[i, 0, :2], theta_obs)
            obs_patch.set_zorder(7)
            ax.add_patch(obs_patch)
            obs_patches.append(obs_patch)

    animate = animate_general(positions_car, positions_obstacles, car_point, car_patch, obs_points, obs_patches)

    return animate


def patch_value(center: np.ndarray, theta: float) -> Tuple[np.ndarray, float]:
    xy = center + R_box(theta) @ np.array([-length_car / 2, -width_car / 2])
    angle = rad_to_deg(theta)

    return xy, angle


def animate_general(positions_car, positions_obstacles, car_point, car_patch, obs_points,
                    obs_patches: List or None) -> Callable[[int], Tuple]:
    def animate(i) -> Tuple:
        return_values = []
        if car_patch is not None or car_point is not None:
            car_i = positions_car[i][:2]

        if car_point is not None:
            car_point.set_data(*car_i)
            car_point.set_color((0, 0, 0))
            return_values.append(car_point)

        if car_patch is not None:
            theta_car_i = positions_car[i, 2]
            car_patch.xy, car_patch.angle = patch_value(car_i, theta_car_i)
            return_values.append(car_patch)

        if obs_points is not None:
            obs_points.set_offsets(positions_obstacles[:, i, :2])
            obs_points.set_facecolors((0, 0, 0))
            return_values.append(obs_points)

        if obs_patches is not None:
            for idx, obs_patch in enumerate(obs_patches):
                theta_obs_i = positions_obstacles[idx, i, 3]
                obs_i = positions_obstacles[idx, i, :2]
                obs_patch.xy, obs_patch.angle = patch_value(obs_i, theta_obs_i)
                return_values.append(obs_patch)

        # Return the objects to be able to set blit=True
        return tuple(return_values)

    return animate
