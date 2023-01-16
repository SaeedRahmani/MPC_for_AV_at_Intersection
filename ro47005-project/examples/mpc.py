import itertools
import math
from typing import List

import numpy as np
from matplotlib import pyplot as plt

from envs.t_intersection import t_intersection
from lib.car_dimensions import CarDimensions, BicycleModelDimensions
from lib.collision_avoidance import check_collision_moving_cars, get_cutoff_curve_by_position_idx
from lib.motion_primitive import load_motion_primitives
from lib.motion_primitive_search import MotionPrimitiveSearch
from lib.moving_obstacles import MovingObstacleTIntersection
from lib.moving_obstacles_prediction import MovingObstaclesPrediction
from lib.mpc import MPC, MAX_ACCEL
from lib.plotting import draw_car
from lib.simulation import State, Simulation, History, HistorySimulation
from lib.trajectories import resample_curve, calc_nearest_index_in_direction


def visualize_mpc(mpc: MPC, state: State, history: History, car_dimensions: CarDimensions):
    if mpc.ox is not None:
        plt.plot(mpc.ox, mpc.oy, "+r", label="MPC")
    # plt.plot(mpc.cx, mpc.cy, "-r", label="course")
    # plt.plot(history.x, history.y, "ob", label="trajectory")
    plt.plot(mpc.xref[0, :], mpc.xref[1, :], "+k", label="xref")
    # plt.plot(mpc.cx[mpc.target_ind], mpc.cy[mpc.target_ind], "xg", label="target")
    # for i in range(T + 1):
    #     plt.plot([mpc.xref[0, i], mpc.ox[i]], [mpc.xref[1, i], mpc.oy[i]], c='k')
    draw_car((state.x, state.y, state.yaw), steer=mpc.di, car_dimensions=car_dimensions, ax=plt.gca(), color='k')


def visualize_frame(dt, frame_window, car_dimensions, collision_xy, i, moving_obstacles, mpc, scenario, simulation,
                    state, tmp_trajectory, trajectory_res, trajs_moving_obstacles):
    if i >= 0:
        plt.cla()
        plt.plot(tmp_trajectory[:, 0], tmp_trajectory[:, 1], color='k')

        if collision_xy is not None:
            plt.scatter([collision_xy[0]], [collision_xy[1]], color='r')

        plt.scatter([state.x], [state.y], color='r')
        plt.scatter([trajectory_res[0, 0]], [trajectory_res[0, 1]], color='b')

        for tr in [*trajs_moving_obstacles]:
            plt.plot(tr[:, 0], tr[:, 1], color='b')

        for obstacle in scenario.obstacles:
            obstacle.draw(plt.gca(), color='b')

        for mo in moving_obstacles:
            x, y, _, theta, _, _ = mo.get()
            draw_car((x, y, theta), car_dimensions, ax=plt.gca())

        visualize_mpc(mpc, state, simulation.history, car_dimensions)
        plt.title("Time: %.2f" % (i * dt))
        plt.axis("equal")
        plt.grid(True)

        plt.xlim((-45, 45))
        plt.ylim((-50, 20))

        plt.pause(0.001)
        # plt.show()


def main():
    #########
    # INIT ENVIRONMENT
    #########
    mps = load_motion_primitives(version='bicycle_model')
    scenario = t_intersection()
    car_dimensions: CarDimensions = BicycleModelDimensions(skip_back_circle_collision_checking=False)

    DT = 0.2

    moving_obstacles: List[MovingObstacleTIntersection] = [
        MovingObstacleTIntersection(car_dimensions, direction=1, offset=1., turning=True, speed=10 / 3.6, dt=DT),
        MovingObstacleTIntersection(car_dimensions, direction=-1, turning=True, speed=25 / 3.6, dt=DT),
    ]

    #########
    # MOTION PRIMITIVE SEARCH
    #########
    search = MotionPrimitiveSearch(scenario, car_dimensions, mps, margin=car_dimensions.radius)

    _, _, trajectory = search.run(debug=False)

    #########
    # INIT MPC
    #########
    dl = np.linalg.norm(trajectory[0, :2] - trajectory[1, :2])

    mpc = MPC(cx=trajectory[:, 0], cy=trajectory[:, 1], cyaw=trajectory[:, 2], dl=dl, dt=DT,
              car_dimensions=car_dimensions)
    state = State(x=trajectory[0, 0], y=trajectory[0, 1], yaw=trajectory[0, 2], v=0.0)

    simulation = HistorySimulation(car_dimensions=car_dimensions, sample_time=DT, initial_state=state)
    history = simulation.history  # gets updated automatically as simulation runs

    #########
    # SIMULATE
    #########
    TIME_HORIZON = 7.
    FRAME_WINDOW = 20
    EXTRA_CUTOFF_MARGIN = 4 * int(
        math.ceil(car_dimensions.radius / dl))  # no. of frames - corresponds approximately to car length

    traj_agent_idx = 0
    tmp_trajectory = None

    for i in itertools.count():
        if mpc.is_goal(state):
            break

        # cutoff the trajectory by the closest future index
        # but don't do it if the trajectory is exactly one point already,
        # so that the car doesn't move slowly forwards
        if tmp_trajectory is None or np.any(tmp_trajectory[traj_agent_idx, :] != tmp_trajectory[-1, :]):
            traj_agent_idx = calc_nearest_index_in_direction(state, trajectory[:, 0], trajectory[:, 1],
                                                             start_index=traj_agent_idx, forward=True)
        trajectory_res = trajectory[traj_agent_idx:]

        # compute trajectory to correspond to a car that starts from its current speed and accelerates
        # as much as it can -> this is a prediction for our own agent
        if state.v < Simulation.MAX_SPEED:
            resample_dl = np.zeros((trajectory_res.shape[0],)) + MAX_ACCEL
            resample_dl = np.cumsum(resample_dl) + state.v
            resample_dl = DT * np.minimum(resample_dl, Simulation.MAX_SPEED)
            trajectory_res = resample_curve(trajectory_res, dl=resample_dl)
        else:
            trajectory_res = resample_curve(trajectory_res, dl=DT * Simulation.MAX_SPEED)

        # predict the movement of each moving obstacle, and retrieve the predicted trajectories
        trajs_moving_obstacles = [
            np.vstack(MovingObstaclesPrediction(*o.get(), sample_time=DT, car_dimensions=car_dimensions)
                      .state_prediction(TIME_HORIZON)).T
            for o in moving_obstacles]

        # find the collision location
        collision_xy = check_collision_moving_cars(car_dimensions, trajectory_res, trajs_moving_obstacles, frame_window=FRAME_WINDOW)

        # cutoff the curve such that it ends right before the collision (and some margin)
        if collision_xy is not None:
            cutoff_idx = get_cutoff_curve_by_position_idx(trajectory, collision_xy[0],
                                                          collision_xy[1]) - EXTRA_CUTOFF_MARGIN
            cutoff_idx = max(traj_agent_idx + 1, cutoff_idx)
            # cutoff_idx = max(traj_agent_idx + 1, cutoff_idx)
            tmp_trajectory = trajectory[:cutoff_idx]
        else:
            tmp_trajectory = trajectory

        # pass the cut trajectory to the MPC
        mpc.set_trajectory_fromarray(tmp_trajectory)

        # compute the MPC
        delta, acceleration = mpc.step(state)

        # show the computation results
        visualize_frame(DT, FRAME_WINDOW, car_dimensions, collision_xy, i, moving_obstacles, mpc, scenario, simulation,
                        state, tmp_trajectory, trajectory_res, trajs_moving_obstacles)

        # move all obstacles forward
        for o in moving_obstacles:
            o.step()

        # step the simulation (i.e. move our agent forward)
        state = simulation.step(a=acceleration, delta=delta)


if __name__ == '__main__':
    main()
