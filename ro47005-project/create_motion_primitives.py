import itertools
import json
from typing import Optional, NamedTuple, List

import gym
from urdfenvs.robots.prius import Prius
import numpy as np
from tqdm.auto import tqdm


class Configuration(NamedTuple):
    name: str
    forward_speed: float
    steering_angle: float
    n_seconds: float


def run_prius(forward_speed: float, steering_angle: float, n_seconds=1., render=False):
    DT = 0.005

    robots = [
        Prius(mode="vel"),  # vel is only possibility
    ]
    env = gym.make(
        "urdf-env-v0",
        dt=DT, robots=robots, render=render
    )
    # action = 2x1 with [forward_speed, steering_angle_dot]
    # forward_speed is used to set the angular speed of the wheel
    action = np.array([1., 0])
    # pos0 = 3x1 with (x-pos, y-pos, orientation]
    pos0 = np.array([0, 0, 0])
    ob = env.reset(pos=pos0)
    # print(f"Initial observation : {ob}")

    # Build the T-intersection
    # t_intersection(env)

    points = []

    end_time: Optional[float] = None
    state: int = 1

    for step in itertools.count(1):
        ob, _, _, _ = env.step(action)

        if state == 1:
            if ob['robot_0']['joint_state']['forward_velocity'][0] >= forward_speed:
                state = 2
            else:
                action[0] += 0.05

        if state == 2:
            action[0] = forward_speed
            action[1] = 0.2

            if ob['robot_0']['joint_state']['steering'] >= steering_angle:
                state = 3
                end_time = step * DT + n_seconds
                start_pos = ob['robot_0']['joint_state']['position']

        if state == 3:
            action[0] = forward_speed
            # Stop steering
            action[1] = 0.
            points.append(ob['robot_0']['joint_state']['position'])

            if step * DT > end_time:
                break
    env.close()
    points = np.array(points)
    # Define the homogeneous transformation matrix
    theta = start_pos[2]
    v = start_pos[:2]
    transformation = np.array([[np.cos(theta), -np.sin(theta), v[0]],
                               [np.sin(theta), np.cos(theta), v[1]],
                               [0, 0, 1]])
    # Create the array with vectors for the homogeneous transformation: [x, y, 1]
    points_to_transform = points.copy()
    points_to_transform[:, 2] = 1
    # Apply the homogeneous transformation
    points_to_transform = np.tensordot(np.linalg.inv(transformation), points_to_transform, axes=[[1], [1]]).T
    # Construct the original points array: [x, y, theta], where theta = theta - start_theta
    points_to_transform[:, 2] = points[:, 2] - theta

    return points_to_transform


# history is a dictionary with the following keys:
# 'position' = 3x1 with [x, y, orientation]
# 'forward_velocity' = 2x1 with [velocity in the car frame, orientationdot]
# 'velocity' = 3x1 with [xdot, ydot, orientationdot]
# 'steering' = 1x1 with [steering_angle]
if __name__ == "__main__":
    CONFIGURATIONS: List[Configuration] = [
        Configuration(n_seconds=0.3, forward_speed=8.3, steering_angle=0., name='straight'),
        Configuration(n_seconds=0.3, forward_speed=8.3, steering_angle=0.1, name='left1'),
        Configuration(n_seconds=0.3, forward_speed=8.3, steering_angle=0.2, name='left2'),
    ]

    for conf in tqdm(CONFIGURATIONS):
        points = run_prius(
            n_seconds=conf.n_seconds,
            forward_speed=conf.forward_speed,
            steering_angle=conf.steering_angle,
            render=True
        )

        points = np.array(points)

        file_name = f'./data/motion_primitives/{conf.name}.json'

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
