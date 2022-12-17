from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from envs.t_intersection import t_intersection
from lib.a_star import AStar
from lib.motion_primitive import load_motion_primitives, create_transform_mtx, transform_pts, \
    MotionPrimitiveSearchMetadata
from lib.obstacles import check_collision
from lib.obstacles import draw_obstacle
from lib.obstacles import obstacle_to_convex

if __name__ == '__main__':
    mps = load_motion_primitives()

    scenario = t_intersection()

    fig, ax = plt.subplots()

    # draw the start point
    ax.scatter([scenario.start[0]], [scenario.start[1]], color='b')

    for obstacle in scenario.obstacles:
        draw_obstacle(obstacle, ax, color='b')

    draw_obstacle(scenario.goal_area, ax, color='r')

    mtx = create_transform_mtx(*scenario.start)

    for mp in mps.values():
        points = transform_pts(scenario.start[2], mtx, mp.points)
        ax.plot(points[:, 0], points[:, 1], color='b')

    a_star_data = MotionPrimitiveSearchMetadata(scenario, mps)

    a_star = AStar(neighbor_function=a_star_data.mp_neighbor_function)

    goal_area = scenario.goal_area
    gx, gy, gtheta = scenario.goal_point


    def is_goal(node: Tuple[float, float, float]) -> bool:
        x, y, theta = node
        return check_collision(x, y, goal_area) and abs(gtheta - theta) <= np.pi / 8


    def distance_to_goal(node: Tuple[float, float, float]) -> float:
        x, y, theta = node
        return np.sqrt((gx - x) ** 2 + (gy - y) ** 2) + np.abs(np.sin((gtheta - theta) / 2))


    cost, path = a_star.run(start=scenario.start, is_goal_function=is_goal, heuristic_function=distance_to_goal)

    points = a_star_data.augment_path(path)
    ax.plot(points[:, 0], points[:, 1], color='b')

    ax.axis('equal')

    plt.show()
