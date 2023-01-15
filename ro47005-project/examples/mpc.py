import math
from typing import List

import numpy as np
from matplotlib import pyplot as plt

from envs.t_intersection import t_intersection
from examples.moving_obstacle_avoidance import plot_trajectories
from lib.car_dimensions import CarDimensions, BicycleModelDimensions
from lib.collision_avoidance import check_collision_moving_cars, cutoff_curve_by_position, \
    _offset_trajectories_by_frames
from lib.motion_primitive import load_motion_primitives
from lib.motion_primitive_search import MotionPrimitiveSearch
from lib.moving_obstacles import MovingObstacleTIntersection
from lib.moving_obstacles_prediction import MovingObstaclesPrediction
from lib.mpc import MPC, MAX_ACCEL
from lib.plotting import draw_car
from lib.simulation import State, Simulation, History, HistorySimulation
from lib.trajectories import resample_curve, calc_nearest_index, car_trajectory_to_collision_point_trajectories


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
    plt.axis("equal")
    plt.grid(True)
    plt.title("Time[s]:" + str(round(history.get_current_time(), 2))
              + ", speed[km/h]:" + str(round(state.v * 3.6, 2)))


def main():
    #########
    # INIT ENVIRONMENT
    #########
    mps = load_motion_primitives(version='bicycle_model')
    scenario = t_intersection()
    car_dimensions: CarDimensions = BicycleModelDimensions(skip_back_circle_collision_checking=False)

    DT = 0.2

    moving_obstacles: List[MovingObstacleTIntersection] = [
        MovingObstacleTIntersection(car_dimensions, direction=1, turning=True, speed=15 / 3.6, dt=DT),
        MovingObstacleTIntersection(car_dimensions, direction=-1, turning=False, speed=15 / 3.6, dt=DT),
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

    mpc = MPC(cx=trajectory[:, 0], cy=trajectory[:, 1], cyaw=trajectory[:, 2], dl=dl, dt=DT, car_dimensions=car_dimensions)
    state = State(x=trajectory[0, 0], y=trajectory[0, 1], yaw=trajectory[0, 2], v=0.0)

    simulation = HistorySimulation(car_dimensions=car_dimensions, sample_time=DT, initial_state=state)

    #########
    # SIMULATE
    #########
    TIME_HORIZON = 5.
    FRAME_WINDOW = 6
    EXTRA_CUTOFF_MARGIN = int(car_dimensions.radius / dl)  # no. of frames

    traj_agent_prediction_full = resample_curve(trajectory, dl=DT * 15 / 3.6)
    traj_agent_idx = 0

    i = 0
    while not mpc.is_goal(state):
        traj_agent_idx = calc_nearest_index(state, traj_agent_prediction_full[:, 0], traj_agent_prediction_full[:, 1],
                                            start_index=traj_agent_idx)
        traj_agent_prediction = traj_agent_prediction_full[traj_agent_idx:]
        trajs_moving_obstacles = [
            np.vstack(MovingObstaclesPrediction(*o.step(), sample_time=DT, car_dimensions=car_dimensions)
                      .state_prediction(TIME_HORIZON)).T
            for o in moving_obstacles]

        collision_xy = check_collision_moving_cars(car_dimensions, traj_agent_prediction, trajs_moving_obstacles,
                                                   frame_window=FRAME_WINDOW)
        if collision_xy is not None:
            tmp_trajectory = cutoff_curve_by_position(trajectory, collision_xy[0], collision_xy[1])[
                             :-EXTRA_CUTOFF_MARGIN]
        else:
            tmp_trajectory = trajectory
        mpc.set_trajectory_fromarray(tmp_trajectory)

        delta, acceleration = mpc.step(state)
        state = simulation.step(a=acceleration, delta=delta)

        trajs_o = _offset_trajectories_by_frames(trajs_moving_obstacles,
                                                 list(range(-FRAME_WINDOW, FRAME_WINDOW + 1, 1)))
        trajs_agent = [traj_agent_prediction]

        cc_trajs_agent = [car_trajectory_to_collision_point_trajectories(tr, car_dimensions) for tr in trajs_agent]
        cc_trajs_o = [car_trajectory_to_collision_point_trajectories(tr, car_dimensions) for tr in trajs_o]

        for j in range(len(trajs_moving_obstacles[0])):
            plt.cla()
            plot_trajectories(cc_trajs_agent, cc_trajs_o, radius=car_dimensions.radius, i=j)

            plt.plot(tmp_trajectory[:, 0], tmp_trajectory[:, 1], color='k')

            for tr in [*trajs_moving_obstacles]:
                plt.plot(tr[:, 0], tr[:, 1], color='b')

            for obstacle in scenario.obstacles:
                obstacle.draw(plt.gca(), color='b')

            for mo in moving_obstacles:
                draw_car(mo.get(), car_dimensions, ax=plt.gca())

            visualize_mpc(mpc, state, simulation.history, car_dimensions=car_dimensions)
            plt.pause(0.001)
            plt.title(f"Time: {i * DT}; frame: {j}")
            # plt.show()
            break
        i += 1


if __name__ == '__main__':
    main()
