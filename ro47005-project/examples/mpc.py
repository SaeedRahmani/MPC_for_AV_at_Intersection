import math

import numpy as np
from matplotlib import pyplot as plt

from envs.t_intersection import t_intersection
from lib.car_dimensions import CarDimensions, PriusDimensions
from lib.motion_primitive import load_motion_primitives
from lib.motion_primitive_search import MotionPrimitiveSearch
from lib.mpc import MPC, History, T, BACKTOWHEEL, LENGTH, WIDTH, WHEEL_LEN, WHEEL_WIDTH, TREAD, WB, smooth_yaw
from lib.simulation import State, update_state
from lib.trajectories import resample_curve


def visualize_mpc(mpc: MPC, state: State, history: History):
    if mpc.ox is not None:
        plt.plot(mpc.ox, mpc.oy, "+r", label="MPC")
    plt.plot(mpc.cx, mpc.cy, "-r", label="course")
    plt.plot(history.x, history.y, "ob", label="trajectory")
    plt.plot(mpc.xref[0, :], mpc.xref[1, :], "+k", label="xref")
    plt.plot(mpc.cx[mpc.target_ind], mpc.cy[mpc.target_ind], "xg", label="target")
    for i in range(T + 1):
        plt.plot([mpc.xref[0, i], mpc.ox[i]], [mpc.xref[1, i], mpc.oy[i]], c='k')
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
    # INIT
    #########
    mps = load_motion_primitives(version='bicycle_model_real_size')
    scenario = t_intersection()
    car_dimensions: CarDimensions = PriusDimensions(skip_back_circle_collision_checking=False)

    search = MotionPrimitiveSearch(scenario, car_dimensions, mps, margin=car_dimensions.radius)

    _, _, trajectory = search.run(debug=False)

    plt.plot(trajectory[:, 0], trajectory[:, 1], color='b')
    plt.scatter(trajectory[480:481, 0], trajectory[480:481, 1], color='r')

    dl = np.linalg.norm(trajectory[0, :2] - trajectory[1, :2])

    mpc = MPC(cx=trajectory[:, 0], cy=trajectory[:, 1], cyaw=trajectory[:, 2], dl=dl)
    history = History()
    state = State(x=trajectory[0, 0], y=trajectory[0, 1], yaw=trajectory[0, 2], v=0.0)
    history.store_state(state)

    while not mpc.is_goal(state):
        delta, acceleration = mpc.step(state)
        state = update_state(state, a=acceleration, delta=delta)
        history.store(state, a=acceleration, d=delta)

        plt.cla()
        for obstacle in scenario.obstacles:
            obstacle.draw(plt.gca(), color='b')

        visualize_mpc(mpc, state, history)
        plt.pause(0.001)
        # plt.show()


if __name__ == '__main__':
    main()
