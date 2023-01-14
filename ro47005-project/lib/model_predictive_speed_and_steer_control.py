"""
Path tracking simulation with iterative linear model predictive control for speed and steer control
author: Atsushi Sakai (@Atsushi_twi)
"""
import matplotlib.pyplot as plt
import cvxpy
import math
import numpy as np

NX = 4  # x = x, y, v, yaw
NU = 2  # a = [accel, steer]
T = 14  # horizon length

# mpc parameters
R = np.diag([0.01, 0.01])  # input cost matrix
Rd = np.diag([0.01, 1.0])  # input difference cost matrix
Q_v_yaw = np.diag([0., 0.5])  # state cost matrix [v, yaw]
Qf = np.diag([1.0, 1.0, 0., 0.5]) * 10.  # state final matrix [x, y, v, yaw]
GOAL_DIS = 1.5  # goal distance
STOP_SPEED = 0.5 / 3.6  # stop speed
MAX_TIME = 13.0  # max simulation time

# iterative paramter
MAX_ITER = 1  # Max iteration
DU_TH = 0.1  # iteration finish param

DT = 0.2  # [s] time tick

# Vehicle parameters
LENGTH = 4.5  # [m]
WIDTH = 2.0  # [m]
BACKTOWHEEL = 1.0  # [m]
WHEEL_LEN = 0.3  # [m]
WHEEL_WIDTH = 0.2  # [m]
TREAD = 0.7  # [m]
WB = 2.5  # [m]

MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(30.0)  # maximum steering speed [rad/s]
MAX_SPEED = 30.0 / 3.6  # maximum speed [m/s]
MIN_SPEED = 0.  # minimum speed [m/s]
MAX_ACCEL = 2.0  # maximum accel [m/ss]
MIN_ACCEL = -3.5  # maximum accel [m/ss]

show_animation = True


class State:
    """
    vehicle state class
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.predelta = None


def pi_2_pi(angle):
    while(angle > math.pi):
        angle = angle - 2.0 * math.pi

    while(angle < -math.pi):
        angle = angle + 2.0 * math.pi

    return angle


def get_linear_model_matrix(v, phi, delta):

    A = np.zeros((NX, NX))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 1.0
    A[3, 3] = 1.0
    A[0, 2] = DT * math.cos(phi)
    A[0, 3] = - DT * v * math.sin(phi)
    A[1, 2] = DT * math.sin(phi)
    A[1, 3] = DT * v * math.cos(phi)
    A[3, 2] = DT * math.tan(delta) / WB

    B = np.zeros((NX, NU))
    B[2, 0] = DT
    B[3, 1] = DT * v / (WB * math.cos(delta) ** 2)

    C = np.zeros(NX)
    C[0] = DT * v * math.sin(phi) * phi
    C[1] = - DT * v * math.cos(phi) * phi
    C[3] = - DT * v * delta / (WB * math.cos(delta) ** 2)

    return A, B, C


def plot_car(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover

    outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

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


def update_state(state, a, delta):

    # input check
    if delta >= MAX_STEER:
        delta = MAX_STEER
    elif delta <= -MAX_STEER:
        delta = -MAX_STEER

    state.x = state.x + state.v * math.cos(state.yaw) * DT
    state.y = state.y + state.v * math.sin(state.yaw) * DT
    state.yaw = state.yaw + state.v / WB * math.tan(delta) * DT
    state.v = state.v + a * DT

    if state.v > MAX_SPEED:
        state.v = MAX_SPEED
    elif state.v < MIN_SPEED:
        state.v = MIN_SPEED

    return state


def get_nparray_from_matrix(x):
    return np.array(x).flatten()


def calc_nearest_index(state, cx, cy, pind):
    dx = cx[pind:] - state.x
    dy = cy[pind:] - state.y

    d = dx ** 2 + dy ** 2
    return np.argmin(d) + pind


def predict_motion(x0, oa, od, xref):
    xbar = xref * 0.0
    for i, _ in enumerate(x0):
        xbar[i, 0] = x0[i]

    state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
    for (ai, di, i) in zip(oa, od, range(1, T + 1)):
        state = update_state(state, ai, di)
        xbar[0, i] = state.x
        xbar[1, i] = state.y
        xbar[2, i] = state.v
        xbar[3, i] = state.yaw

    return xbar


def iterative_linear_mpc_control(x0, oa, od, state, cx, cy, cyaw, dl, target_ind):
    """
    MPC contorl with updating operational point iteraitvely
    :param state:
    :param cx:
    :param cy:
    :param cyaw:
    :param dl:
    :param target_ind:
    """

    if oa is None or od is None:
        oa = [0.0] * T
        od = [0.0] * T

    ov = None

    for _ in range(MAX_ITER):
        xref, target_ind, dref, reaches_end = calc_ref_trajectory(state, cx, cy, cyaw, dl, target_ind, ov)
        xbar = predict_motion(x0, oa, od, xref)
        # poa, pod = oa, od
        oa, od, ox, oy, oyaw, ov = linear_mpc_control(xref, xbar, x0, dref, reaches_end)
        # du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value
        # if du <= DU_TH:
        #     break
    # else:
    #     print("Iterative is max iter")

    return oa, od, ox, oy, oyaw, ov, xref, target_ind


def get_xy_cost_mtx_for_orientation(angle: float):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [c ** 2, c * s],
        [c * s, s ** 2]
    ])


def linear_mpc_control(xref, xbar, x0, dref, reaches_end):
    """
    linear mpc control
    xref: reference point
    xbar: operational point
    x0: initial state
    dref: reference steer angle
    :param reaches_end:
    """

    x = cvxpy.Variable((NX, T + 1))
    u = cvxpy.Variable((NU, T))

    cost = 0.0
    constraints = []

    for t in range(T + 1):
        if t > 0:
            if not reaches_end[t]:
                # penalize difference from reference perpendicular to yaw strongly
                ref_yaw_perp = xref[3, t] + 0.5 * np.pi
                cost += cvxpy.quad_form(xref[:2, t] - x[:2, t], get_xy_cost_mtx_for_orientation(ref_yaw_perp) * 10.)

                # penalize difference from reference parallel to yaw weakly
                ref_yaw = xref[3, t]
                cost += cvxpy.quad_form(xref[:2, t] - x[:2, t], get_xy_cost_mtx_for_orientation(ref_yaw) * 1.0)

                # penalize velocity and yaw itself
                cost += cvxpy.quad_form(xref[2:, t] - x[2:, t], Q_v_yaw)
            else:
                cost += cvxpy.quad_form(xref[:, t] - x[:, t], Qf)


        if t < T:
            cost += cvxpy.quad_form(u[:, t], R)

            A, B, C = get_linear_model_matrix(
                xbar[2, t], xbar[3, t], dref[0, t])
            constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

        if t < (T - 1):
            cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)
            constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= MAX_DSTEER * DT]

    constraints += [x[:, 0] == x0]
    constraints += [x[2, :] <= MAX_SPEED]
    constraints += [x[2, :] >= MIN_SPEED]
    constraints += [u[0, :] <= MAX_ACCEL]
    constraints += [u[0, :] >= MIN_ACCEL]
    constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=cvxpy.ECOS, verbose=False)

    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        ox = get_nparray_from_matrix(x.value[0, :])
        oy = get_nparray_from_matrix(x.value[1, :])
        ov = get_nparray_from_matrix(x.value[2, :])
        oyaw = get_nparray_from_matrix(x.value[3, :])
        oa = get_nparray_from_matrix(u.value[0, :])
        odelta = get_nparray_from_matrix(u.value[1, :])

    else:
        raise Exception("Error: Cannot solve mpc...")
        # oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

    return oa, odelta, ox, oy, oyaw, ov


def calc_ref_trajectory(state, cx, cy, cyaw, dl, start_idx, ov):
    xref = np.zeros((NX, T + 1))
    dref = np.zeros((1, T + 1))
    ncourse = len(cx)

    start_idx = max(calc_nearest_index(state, cx, cy, start_idx), start_idx)

    dref[:, :T + 1] = 0.0  # steer operational point should be 0.0 for indices 0 to T

    if ov is None:
        ov = np.ones((T + 1,)) * max(state.v, 5 / 3.6)

    travel = np.cumsum(np.abs(ov) * DT)
    idx = np.rint(travel / dl).astype(int)
    idx = np.minimum(idx + start_idx, ncourse - 1)

    xref[0, :T + 1] = cx[idx]
    xref[1, :T + 1] = cy[idx]
    # we skip the velocity - we don't need it (Q must have a 0 for it)
    xref[3, :T + 1] = cyaw[idx]

    reaches_end = idx == ncourse - 1

    return xref, start_idx, dref, reaches_end


def check_goal(state, goal, tind, nind):

    # check goal
    dx = state.x - goal[0]
    dy = state.y - goal[1]
    d = math.hypot(dx, dy)

    isgoal = (d <= GOAL_DIS)

    if abs(tind - nind) >= 5:
        isgoal = False

    isstop = (abs(state.v) <= STOP_SPEED)

    if isgoal and isstop:
        return True

    return False


def do_simulation(cx, cy, cyaw, ck, dl, initial_state):
    """
    Simulation
    cx: course x position list
    cy: course y position list
    cy: course yaw position list
    ck: course curvature list
    dl: course tick [m]
    """

    goal = [cx[-1], cy[-1]]

    state = initial_state

    # initial yaw compensation
    if state.yaw - cyaw[0] >= math.pi:
        state.yaw -= math.pi * 2.0
    elif state.yaw - cyaw[0] <= -math.pi:
        state.yaw += math.pi * 2.0

    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    d = [0.0]
    a = [0.0]
    target_ind = calc_nearest_index(state, cx, cy, 0)

    odelta, oa = None, None

    cyaw = smooth_yaw(cyaw)

    should_terminate = False

    def terminate_early():
        nonlocal should_terminate
        should_terminate = True

    while not should_terminate and MAX_TIME >= time:
        x0 = [state.x, state.y, state.v, state.yaw]  # current state

        oa, odelta, ox, oy, oyaw, ov, xref, target_ind = iterative_linear_mpc_control(x0, oa, odelta, state, cx, cy,
                                                                                      cyaw, dl, target_ind)

        if odelta is not None:
            di, ai = odelta[0], oa[0]

        state = update_state(state, ai, di)
        time = time + DT

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)
        d.append(di)
        a.append(ai)

        if check_goal(state, goal, target_ind, len(cx)):
            print("Goal")
            break

        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [terminate_early() if event.key == 'escape' else None])
        plt.gcf().canvas.mpl_connect('close_event', lambda event: terminate_early())
        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            if ox is not None:
                plt.plot(ox, oy, "+r", label="MPC")
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(x, y, "ob", label="trajectory")
            plt.plot(xref[0, :], xref[1, :], "+k", label="xref")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            for i in range(T):
                plt.plot([xref[0, i], ox[i]], [xref[1, i], oy[i]], c='k')
            plot_car(state.x, state.y, state.yaw, steer=di)
            plt.axis("equal")
            plt.grid(True)
            plt.title("Time[s]:" + str(round(time, 2))
                      + ", speed[km/h]:" + str(round(state.v * 3.6, 2)))
            plt.pause(0.0001)

    return t, x, y, yaw, v, d, a


def smooth_yaw(yaw):

    for i in range(len(yaw) - 1):
        dyaw = yaw[i + 1] - yaw[i]

        while dyaw >= math.pi / 2.0:
            yaw[i + 1] -= math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

        while dyaw <= -math.pi / 2.0:
            yaw[i + 1] += math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

    return yaw

def call_path():
    import pandas as pd
    df = pd.read_csv('example_traj_bicycle.csv', sep=',')
    print(df)
    cx = df.iloc[:,0].values * (10/3)
    cy = df.iloc[:, 1].values * (10/3)
    cyaw = df.iloc[:, 2].values
    ck = 0

    return cx, cy, cyaw, ck


def main(cx, cy, cyaw, ck=0., dl=0.0451 / 0.3):
    print(__file__ + " start!!")

    initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)

    t, x, y, yaw, v, d, a = do_simulation(
        cx, cy, cyaw, ck, dl, initial_state)

    if show_animation:  # pragma: no cover
        plt.close("all")
        plt.subplots()
        plt.plot(cx, cy, "-r", label="spline")
        plt.plot(x, y, "-g", label="tracking")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        fig, ax = plt.subplots()
        plt.plot(t, np.array(v) * 3.6, "-r", label="speed")
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Speed [kmh]")

        plt.show()

    return x, y, yaw, v

def main2():
    print(__file__ + " start!!")

    dl = 0.0451 / 0.3  # course tick
    cx, cy, cyaw, ck = get_straight_course3(dl)

    initial_state = State(x=cx[0], y=cy[0], yaw=0.0, v=0.0)

    t, x, y, yaw, v, d, a = do_simulation(
        cx, cy, cyaw, ck, dl, initial_state)

    if show_animation:  # pragma: no cover
        plt.close("all")
        plt.subplots()
        plt.plot(cx, cy, "-r", label="spline")
        plt.plot(x, y, "-g", label="tracking")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        plt.subplots()
        plt.plot(t, v, "-r", label="speed")
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Speed [kmh]")

        plt.show()


if __name__ == '__main__':
    # call_path()
    main()
    # main2()