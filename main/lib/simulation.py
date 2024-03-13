import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from bicycle.main import Bicycle
from lib.car_dimensions import CarDimensions


@dataclass
class State:
    """
    vehicle state class
    """
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    v: float = 0.0


class Simulation:
    MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]
    MAX_SPEED = 30.0 / 3.6  # maximum speed [m/s]
    MIN_SPEED = -5.  # minimum speed [m/s]

    def __init__(self, car_dimensions: CarDimensions, sample_time: float, initial_state: State):
        self._bicycle = Bicycle(car_dimensions=car_dimensions, sample_time=sample_time)
        self._sample_time = sample_time
        self._bicycle.xc = initial_state.x
        self._bicycle.yc = initial_state.y
        self._bicycle.theta = initial_state.yaw
        self._v = initial_state.v

    def step(self, a: float, delta: float) -> State:
        # clamp input into allowed interval
        delta = max(min(delta, Simulation.MAX_STEER), -Simulation.MAX_STEER)

        self._bicycle.step(v=self._v, delta=delta)

        # update velocity by acceleration
        self._v += a * self._sample_time

        # clamp velocity into [min, max] interval
        self._v = max(min(self._v, Simulation.MAX_SPEED), Simulation.MIN_SPEED)

        return State(x=self._bicycle.xc, y=self._bicycle.yc, yaw=self._bicycle.theta, v=self._v)


class HistorySimulation(Simulation):
    """Simulation that also automatically keeps track of history"""

    def __init__(self, car_dimensions: CarDimensions, sample_time: float, initial_state: State):
        super().__init__(car_dimensions, sample_time, initial_state) # loads from the Simulation Class
        self.history = History(sample_time=sample_time) # initializing the History class to make the place holders for needed variables
        self.history.store(initial_state, a=0., delta=0., xref_deviation=0.) # Storing the variables for the initial state

    def step(self, a: float, delta: float, xref_deviation: Optional[float] = None) -> State:
        new_state = super().step(a, delta)
        self.history.store(new_state, a=a, delta=delta, xref_deviation=xref_deviation)
        return new_state


class History:

    def __init__(self, sample_time: float):
        self.x: List[float] = []
        self.y: List[float] = []
        self.yaw: List[float] = []
        self.v: List[float] = []
        self.t: List[float] = []
        self.delta: List[float] = []
        self.a: List[float] = []
        self.xref_deviation: List[float] = []
        self._sample_time = sample_time

    def store(self, state: State, a: float, delta: float, xref_deviation: Optional[float] = None):
        self.x.append(state.x)
        self.y.append(state.y)
        self.yaw.append(state.yaw)
        self.v.append(state.v)
        self.t.append(self.get_current_time() + self._sample_time)
        self.delta.append(delta)
        self.a.append(a)
        self.xref_deviation.append(xref_deviation if xref_deviation is not None else np.nan)

    def get_current_time(self) -> float:
        return self.t[-1] if len(self.t) > 0 else 0.
