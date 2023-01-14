import math
from typing import List

import numpy as np
from matplotlib import pyplot as plt

from envs.t_intersection import t_intersection
from examples.moving_obstacle_avoidance import plot_trajectories
from lib.car_dimensions import CarDimensions, PriusDimensions
from lib.collision_avoidance import check_collision_moving_cars, cutoff_curve_by_position, \
    _offset_trajectories_by_frames
from lib.motion_primitive import load_motion_primitives
from lib.motion_primitive_search import MotionPrimitiveSearch
from lib.moving_obstacles import MovingObstacleTIntersection
from lib.moving_obstacles_prediction import MovingObstaclesPrediction
from lib.mpc import MPC, History, T, BACKTOWHEEL, LENGTH, WIDTH, WHEEL_LEN, WHEEL_WIDTH, TREAD, WB, smooth_yaw
from lib.plotting import draw_car
from lib.simulation import State, update_state, DT
from lib.trajectories import resample_curve, calc_nearest_index, car_trajectory_to_collision_point_trajectories


def visualize_mpc(mpc: MPC, state: State, history: History):
    if mpc.ox is not None:
        plt.plot(mpc.ox, mpc.oy, "+r", label="MPC")
    # plt.plot(mpc.cx, mpc.cy, "-r", label="course")
    # plt.plot(history.x, history.y, "ob", label="trajectory")
    plt.plot(mpc.xref[0, :], mpc.xref[1, :], "+k", label="xref")
    # plt.plot(mpc.cx[mpc.target_ind], mpc.cy[mpc.target_ind], "xg", label="target")
    # for i in range(T + 1):
    #     plt.plot([mpc.xref[0, i], mpc.ox[i]], [mpc.xref[1, i], mpc.oy[i]], c='k')
    plot_car(state.x, state.y, state.yaw, steer=mpc.di)
    plt.axis("equal")
    plt.grid(True)
    plt.title("Time[s]:" + str(round(history.get_current_time(), 2))
              + ", speed[km/h]:" + str(round(state.v * 3.6, 2)))


def plot_car(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover

    outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD,
                          -WHEEL_WIDTH - TREAD]])

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB

    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fl_wheel[0, :]).flatten(),
             np.array(fl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(x, y, "*")


def main():
    #########
    # INIT ENVIRONMENT
    #########
    mps = load_motion_primitives(version='bicycle_model_real_size')
    scenario = t_intersection()
    car_dimensions: CarDimensions = PriusDimensions(skip_back_circle_collision_checking=False)

    moving_obstacles: List[MovingObstacleTIntersection] = [
        MovingObstacleTIntersection(direction=1, turning=False, speed=10 / 3.6, dt=DT)
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

    mpc = MPC(cx=trajectory[:, 0], cy=trajectory[:, 1], cyaw=trajectory[:, 2], dl=dl)
    history = History()
    state = State(x=trajectory[0, 0], y=trajectory[0, 1], yaw=trajectory[0, 2], v=0.0)
    history.store_state(state)

    #########
    # SIMULATE
    #########
    TIME_HORIZON = 5.
    FRAME_WINDOW = 6
    EXTRA_CUTOFF_MARGIN = 0 #int((2 * car_dimensions.radius) / dl)  # no. of frames

    traj_agent_prediction_full = resample_curve(trajectory, dl=DT * 15 / 3.6)
    traj_agent_idx = 0

    i = 0
    while not mpc.is_goal(state):
        traj_agent_idx = calc_nearest_index(state, traj_agent_prediction_full[:, 0], traj_agent_prediction_full[:, 1],
                                            start_index=traj_agent_idx)
        traj_agent_prediction = traj_agent_prediction_full[traj_agent_idx:]
        trajs_moving_obstacles = [
            np.vstack(MovingObstaclesPrediction(*o.step(), sample_time=DT).state_prediction(TIME_HORIZON)).T for o in
            moving_obstacles]

        collision_xy = check_collision_moving_cars(car_dimensions, traj_agent_prediction, trajs_moving_obstacles,
                                                   frame_window=FRAME_WINDOW)
        if collision_xy is not None:
            tmp_trajectory = cutoff_curve_by_position(trajectory, collision_xy[0], collision_xy[1])[
                             :EXTRA_CUTOFF_MARGIN]
        else:
            tmp_trajectory = trajectory
        mpc.set_trajectory_fromarray(tmp_trajectory)

        delta, acceleration = mpc.step(state)
        state = update_state(state, a=acceleration, delta=delta)
        history.store(state, a=acceleration, d=delta)

        trajs_o = _offset_trajectories_by_frames(trajs_moving_obstacles,
                                                 list(range(-FRAME_WINDOW, FRAME_WINDOW + 1, 1)))
        trajs_agent = [traj_agent_prediction]
        # trajs_agent = _offset_trajectories_by_frames(trajs_agent, set(range(-FRAME_WINDOW, FRAME_WINDOW + 1, 1)))

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

            visualize_mpc(mpc, state, history)
            # plt.pause(0.001)
            plt.title(f"Time: {i * DT}; frame: {j}")
            plt.show()
            break
        i += 1


if __name__ == '__main__':
    main()
