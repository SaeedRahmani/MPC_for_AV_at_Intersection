import gym
from urdfenvs.robots.prius import Prius
import numpy as np
from envs.t_intersection import t_intersection


def run_prius(n_steps=1000, render=False, goal=True, obstacles=True):
    robots = [
        Prius(mode="vel"),  # vel is only possibility
    ]
    env = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    # action = 2x1 with [forward_speed, steering_angle_dot]
    # forward_speed is used to set the angular speed of the wheel
    action = np.array([1.1, 0.2])
    # pos0 = 3x1 with (x-pos, y-pos, orientation]
    pos0 = np.array([1.5, -5, 0.5 * np.pi])
    ob = env.reset(pos=pos0)
    print(f"Initial observation : {ob}")

    # Build the T-intersection
    t_intersection(env)

    history = []
    for i in range(n_steps):
        ob, _, _, _ = env.step(action)
        if ob['robot_0']['joint_state']['steering'] > 0.2:
            action[1] = 0
        history.append(ob)
    # env.close()
    return history

# history is a dictionary with the following keys:
# 'position' = 3x1 with [x, y, orientation]
# 'forward_velocity' = 2x1 with [velocity in the car frame, orientationdot]
# 'velocity' = 3x1 with [xdot, ydot, orientationdot]
# 'steering' = 1x1 with [steering_angle]
if __name__ == "__main__":
    history = run_prius(render=True)

print("Placeholder")
