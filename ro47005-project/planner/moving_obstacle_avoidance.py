from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from envs.t_intersection import t_intersection
from lib.car_dimensions import BicycleModelDimensions, CarDimensions
from lib.collision_avoidance import check_collision_moving_cars
from lib.plotting import draw_circles, draw_car


def plot_trajectory(tr, radius: float, i: Optional[int] = None, marker='+'):
    if not isinstance(tr, list):
        tr = [tr]

    COLORS = ['b', 'c']

    for j, trr in enumerate(tr):
        if i is not None:
            i = min(i, len(trr))
            trr = trr[i:i + 1]

        if radius is not None:
            draw_circles(trr, radius, marker=marker, color=COLORS[j % len(COLORS)], ax=plt.gca())
        else:
            plt.plot(trr[:, 0], trr[:, 1], marker, color=COLORS[j % len(COLORS)])


def plot_trajectories(traj_agent, trajs_o, radius: Optional[float] = None, i: Optional[int] = None, marker='+'):
    if not isinstance(traj_agent, list):
        traj_agent = [traj_agent]

    for tr in [*trajs_o, *traj_agent]:
        plot_trajectory(tr, radius, i=i, marker=marker)
    plt.axis('equal')


if __name__ == '__main__':

    # -----
    # SETUP
    # -----

    traj_agent = np.loadtxt('../data/example_trajectory_agent_motion_primitives.csv', delimiter=',')
    trajs_o = [
        np.loadtxt('../data/example_trajectory_obstacle.csv', delimiter=','),
        np.loadtxt('../data/example_trajectory_obstacle2.csv', delimiter=','),
    ]

    trajs_o[0][:, 1] += 0.3
    trajs_o[1][:, 1] -= 0.3

    for tr in trajs_o:
        tr[:, :2] /= 0.3

    # trajs_o[0] = trajs_o[0][::2]
    # trajs_o[1] = trajs_o[1][::2]
    #
    FRAME_WINDOW = 3
    MAX_LENGTH_TRAJECTORY = 10.  # seconds
    DT = 0.2

    car_dimensions: CarDimensions = BicycleModelDimensions()
    scenario = t_intersection()

    # trajs_o = [tr[::2] for tr in trajs_o]
    # trajs_o[0] = trajs_o[0][::2]

    traj_agent = traj_agent[:int(MAX_LENGTH_TRAJECTORY / DT)]
    trajs_o = [tr[:int(MAX_LENGTH_TRAJECTORY / DT)] for tr in trajs_o]

    # -----
    # EXECUTE:
    # -----

    collision_position = check_collision_moving_cars(traj_agent, trajs_o, frame_window=FRAME_WINDOW)

    # -----
    # PLOT:
    # -----

    for i in range(len(traj_agent)):
        plt.cla()
        for o in scenario.obstacles:
            o.draw(plt.gca(), color='b')
        draw_car(traj_agent[i], car_dimensions, ax=plt.gca(), color='k')
        if collision_position is not None:
            plt.scatter([collision_position[0]], [collision_position[1]], color='r')

        # c = cutoff_curve_by_position(traj_agent, *collision_position) if collision_position is not None else traj_agent
        # plt.plot(c[:, 0], c[:, 1], 'r')
        plt.title(f"Frame {i}")
        # plt.pause(0.01)
        plt.show()
