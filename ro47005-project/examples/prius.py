import gym
import numpy as np
from urdfenvs.robots.prius import Prius

from envs.t_intersection import t_intersection
from lib.linalg import create_2d_transform_mtx, transform_2d_pts


def run_prius(n_steps=200, render=False, goal=True, obstacles=True):
    prius = Prius(mode="vel")  # vel is only possibility

    # Build the T-intersection
    scenario = t_intersection()

    start_rotate_mtx = create_2d_transform_mtx(0, 0, scenario.start[2])
    prius._spawn_offset[:2] = np.squeeze(
        transform_2d_pts(scenario.start[2], start_rotate_mtx, np.atleast_2d(prius._spawn_offset[:2])))

    robots = [
        prius,
    ]
    env = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    # action = 2x1 with [forward_speed, steering_angle_dot]
    # forward_speed is used to set the angular speed of the wheel
    action = np.array([0., 0.])
    # pos0 = 3x1 with (x-pos, y-pos, orientation]
    pos0 = np.array(scenario.start)
    ob = env.reset(pos=pos0)
    print(f"Initial observation : {ob}")

    for o in scenario.obstacles:
        o.add_to_bullet_env(env)

    history = []
    points = []
    while True:
        ob, _, _, _ = env.step(action)
        # if ob['robot_0']['joint_state']['steering'] > 0.2:
        # Stop steering
        action[1] = 0.
        # Increase the forward velocity
        # action[0] = 1.
        history.append(ob)
        points.append(ob['robot_0']['joint_state']['position'])
    env.close()
    return history, points


# history is a dictionary with the following keys:
# 'position' = 3x1 with [x, y, orientation]
# 'forward_velocity' = 2x1 with [velocity in the car frame, orientationdot]
# 'velocity' = 3x1 with [xdot, ydot, orientationdot]
# 'steering' = 1x1 with [steering_angle]
if __name__ == "__main__":
    history, points = run_prius(render=True)

print("Placeholder")
