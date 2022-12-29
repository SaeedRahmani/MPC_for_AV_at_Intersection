import gym
import numpy as np
from urdfenvs.robots.prius import Prius

from envs.t_intersection import t_intersection
from lib.linalg import create_2d_transform_mtx, transform_2d_pts
from lib.moving_obstacles import MovingObstacleTIntersection
import pybullet as p
import random


def run_prius(n_steps=1000, render=False, goal=True, obstacles=True):
    prius = Prius(mode="vel")  # vel is only possibility

    # Build the T-intersection
    scenario = t_intersection()

    start_rotate_mtx = create_2d_transform_mtx(0, 0, scenario.start[2])
    prius._spawn_offset[:2] = np.squeeze(
        transform_2d_pts(scenario.start[2], start_rotate_mtx, np.atleast_2d(prius._spawn_offset[:2])))

    robots = [
        prius,
    ]
    dt = 0.01
    env = gym.make(
        "urdf-env-v0",
        dt=dt, robots=robots, render=render
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

    # Moving obstacle
    # from MotionPlanningEnv.dynamicSphereObstacle import DynamicSphereObstacle
    #
    # dynamicObst2Dict = {
    #     "type": "analyticSphere",
    #     "geometry": {"trajectory": ['-10 + 5*t', '-1.25', '0.5'], "radius": 0.5},
    # }
    # dynamicSphereObst2 = DynamicSphereObstacle(name="simpleSphere", content_dict=dynamicObst2Dict)
    # env.add_obstacle(dynamicSphereObst2)

    # orientation = p.getQuaternionFromEuler([0, 0, 0])
    # obs_id = p.loadURDF(fileName="/home/christiaan/gym_env_urdf/gym_envs_urdf/urdfenvs/robots/prius/prius.urdf",
    #                     basePosition=[-10, -1.25, 0.],
    #                     baseOrientation=orientation,
    #                     globalScaling=0.3,
    #                     )
    # print(f"Pybullet engine{obs_id}")
    history = []
    points = []
    pos_increment = 0

    # Moving obstacle
    moving_obs = []
    trajectory_type_options = ['straight', 'turn']
    # The trajectory type is chosen randomly
    # The initialisation of the vehicles is also chosen randomly, when random_start = True
    random_start = False
    for i in range(3):
        if random.choice([True, False]) and random_start:
            moving_obs.append(MovingObstacleTIntersection(trajectory_type=random.choice(trajectory_type_options),
                                                          dt=dt,
                                                          direction=1,
                                                          speed=4,
                                                          offset=i * 2,
                                                          ))
        elif random_start is not True:
            moving_obs.append(MovingObstacleTIntersection(trajectory_type=random.choice(trajectory_type_options),
                                                          dt=dt,
                                                          direction=1,
                                                          speed=4,
                                                          offset=i * 2,
                                                          ))
        if random.choice([True, False]) and random_start:
            moving_obs.append(MovingObstacleTIntersection(trajectory_type=random.choice(trajectory_type_options),
                                                          dt=dt,
                                                          direction=-1,
                                                          speed=4,
                                                          offset=i * 2 + 0.25,
                                                          ))
        elif random_start is not True:
            moving_obs.append(MovingObstacleTIntersection(trajectory_type=random.choice(trajectory_type_options),
                                                          dt=dt,
                                                          direction=-1,
                                                          speed=4,
                                                          offset=i * 2 + 0.25,
                                                          ))

    while True:
        pos_increment += 0.01 * 5
        # p.resetBasePositionAndOrientation(obs_id, [-10 + pos_increment, -1.25, 0], orientation)
        ob, _, _, _ = env.step(action)
        # if ob['robot_0']['joint_state']['steering'] > 0.2:
        # Stop steering
        action[1] = 0.
        # Increase the forward velocity
        # action[0] = 1.
        history.append(ob)
        points.append(ob['robot_0']['joint_state']['position'])

        # Step moving obstacles
        for o in moving_obs:
            o.step()
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
