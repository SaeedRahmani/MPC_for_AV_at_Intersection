import gym
from urdfenvs.robots.prius import Prius
import numpy as np


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

    # Add our own obstacles
    width_road = 2
    width_traffic_island = 0.5
    width_pavement = 2
    length = 5
    height = 0.2
    distance_center = 3

    # T-intersection
    # Leg of T
    env.add_shapes(shape_type="GEOM_BOX", dim=[width_traffic_island, length, height],
                   poses_2d=[[0, -(length / 2 + distance_center), 0]])
    env.add_shapes(shape_type="GEOM_BOX", dim=[width_pavement, length, height],
                   poses_2d=[
                       [(width_traffic_island / 2 + width_road + width_pavement / 2), -(length / 2 + distance_center),
                        0]])
    env.add_shapes(shape_type="GEOM_BOX", dim=[width_pavement, length, height],
                   poses_2d=[
                       [-(width_traffic_island / 2 + width_road + width_pavement / 2), -(length / 2 + distance_center),
                        0]])
    env.add_shapes(shape_type="GEOM_CYLINDER", dim=[(width_traffic_island/2), height],
                   poses_2d=[[0, -distance_center, 0]])

    # left part of T
    env.add_shapes(shape_type="GEOM_BOX", dim=[width_traffic_island, length, height],
                   poses_2d=[[-(length / 2 + distance_center), 0, 0.5 * np.pi]])
    env.add_shapes(shape_type="GEOM_BOX", dim=[width_pavement, length, height],
                   poses_2d=[
                       [-(length / 2 + distance_center), -(width_traffic_island / 2 + width_road + width_pavement / 2),
                        0.5 * np.pi]])
    env.add_shapes(shape_type="GEOM_CYLINDER", dim=[(distance_center - width_traffic_island / 2 - width_road), height],
                   poses_2d=[[-distance_center, -distance_center, 0]])
    env.add_shapes(shape_type="GEOM_CYLINDER", dim=[(width_traffic_island / 2), height],
                   poses_2d=[[-distance_center, 0, 0]])

    # right part of T
    env.add_shapes(shape_type="GEOM_BOX", dim=[width_traffic_island, length, height],
                   poses_2d=[[(length / 2 + distance_center), 0, 0.5 * np.pi]])
    env.add_shapes(shape_type="GEOM_BOX", dim=[width_pavement, length, height],
                   poses_2d=[
                       [(length / 2 + distance_center), -(width_traffic_island / 2 + width_road + width_pavement / 2),
                        0.5 * np.pi]])
    env.add_shapes(shape_type="GEOM_CYLINDER", dim=[(distance_center - width_traffic_island / 2 - width_road), height],
                   poses_2d=[[distance_center, -distance_center, 0]])
    env.add_shapes(shape_type="GEOM_CYLINDER", dim=[(width_traffic_island / 2), height],
                   poses_2d=[[distance_center, 0, 0]])

    # upper part of T
    env.add_shapes(shape_type="GEOM_BOX", dim=[width_pavement, (2 * length + 2 * distance_center), height],
                   poses_2d=[
                       [0, (width_traffic_island / 2 + width_road + width_pavement / 2),
                        0.5 * np.pi]])

    history = []
    for i in range(n_steps):
        ob, _, _, _ = env.step(action)
        if ob['robot_0']['joint_state']['steering'] > 0.2:
            action[1] = 0
        history.append(ob)
    env.close()
    return history

# history is a dictionary with the following keys:
# 'position' = 3x1 with [x, y, orientation]
# 'forward_velocity' = 2x1 with [velocity in the car frame, orientationdot]
# 'velocity' = 3x1 with [xdot, ydot, orientationdot]
# 'steering' = 1x1 with [steering_angle]
if __name__ == "__main__":
    history = run_prius(render=True)

print("Placeholder")
