import itertools
import json
from typing import Optional, NamedTuple, List

import gym
from urdfenvs.robots.prius import Prius
import numpy as np
from tqdm.auto import tqdm


class Configuration(NamedTuple):
    forward_speed: float
    steering_angle: float
    n_seconds: float


def run_prius(forward_speed: float, steering_angle: float, n_seconds=1., render=False):
    DT = 0.01

    robots = [
        Prius(mode="vel"),  # vel is only possibility
    ]
    env = gym.make(
        "urdf-env-v0",
        dt=DT, robots=robots, render=render
    )
    # action = 2x1 with [forward_speed, steering_angle_dot]
    # forward_speed is used to set the angular speed of the wheel
    action = np.array([0., steering_angle])
    # pos0 = 3x1 with (x-pos, y-pos, orientation]
    pos0 = np.array([0, 0, 0])
    ob = env.reset(pos=pos0)
    # print(f"Initial observation : {ob}")

    # Build the T-intersection
    # t_intersection(env)

    points = []

    end_time: Optional[float] = None

    for step in itertools.count(1):
        ob, _, _, _ = env.step(action)
        if end_time is None and ob['robot_0']['joint_state']['steering'] >= steering_angle:
            end_time = step * DT + n_seconds
            print(step)

        if end_time is not None:
            # Stop steering
            action[1] = 0.
            # Increase the forward velocity
            action[0] = forward_speed
            points.append(ob['robot_0']['joint_state']['position'])

            if step * DT > end_time:
                break
    env.close()
    return points


# history is a dictionary with the following keys:
# 'position' = 3x1 with [x, y, orientation]
# 'forward_velocity' = 2x1 with [velocity in the car frame, orientationdot]
# 'velocity' = 3x1 with [xdot, ydot, orientationdot]
# 'steering' = 1x1 with [steering_angle]
if __name__ == "__main__":
    CONFIGURATIONS: List[Configuration] = [
        Configuration(n_seconds=1., forward_speed=1., steering_angle=0.),
        Configuration(n_seconds=1., forward_speed=1., steering_angle=0.2),
        Configuration(n_seconds=1., forward_speed=1., steering_angle=0.4),
    ]

    for conf in tqdm(CONFIGURATIONS):
        points = run_prius(
            n_seconds=conf.n_seconds,
            forward_speed=conf.forward_speed,
            steering_angle=conf.steering_angle,
            render=True
        )

        points = np.array(points)

        file_name = f'./data/motion_primitives/{conf.n_seconds}_{conf.forward_speed}_{conf.steering_angle}.json'

        # compute total length
        total_length = np.linalg.norm(points[:-1, :2] - points[1:, :2], axis=1).sum()

        with open(file_name, 'w') as file:
            json.dump({
                'n_seconds': conf.n_seconds,
                'forward_speed_wheel_angular': conf.forward_speed,
                'steering_angle': conf.steering_angle,
                'total_length': total_length,
                'points': points.tolist()
            }, file, indent=4)
