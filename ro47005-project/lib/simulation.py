import math

import numpy as np

DT = 0.2  # [s] time tick
WB = 2.5  # [m]
MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]
MAX_SPEED = 30.0 / 3.6  # maximum speed [m/s]
MIN_SPEED = 0.  # minimum speed [m/s]


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


def update_state(state: State, a: float, delta: float) -> State:
    state2 = State()
    # input check
    if delta >= MAX_STEER:
        delta = MAX_STEER
    elif delta <= -MAX_STEER:
        delta = -MAX_STEER

    state2.x = state.x + state.v * math.cos(state.yaw) * DT
    state2.y = state.y + state.v * math.sin(state.yaw) * DT
    state2.yaw = state.yaw + state.v / WB * math.tan(delta) * DT
    state2.v = state.v + a * DT

    if state2.v > MAX_SPEED:
        state2.v = MAX_SPEED
    elif state2.v < MIN_SPEED:
        state2.v = MIN_SPEED

    return state2
