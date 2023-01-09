import math
from typing import Tuple, Dict, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import EllipseCollection
from matplotlib.patches import Rectangle

from lib.a_star import AStar
from lib.car_dimensions import CarDimensions
from lib.linalg import create_2d_transform_mtx, transform_2d_pts
from lib.motion_primitive import MotionPrimitive
from lib.motion_primitive_search import MotionPrimitiveSearch
from lib.scenario import Scenario


def draw_scenario(scenario: Scenario, mps: Dict[str, MotionPrimitive], car_dimensions: CarDimensions,
                  search: MotionPrimitiveSearch, ax,
                  draw_obstacles=True, draw_goal=True, draw_car=True, draw_mps=True, draw_collision_checking=True,
                  draw_car2=True, draw_mps2=True, mp_name='right1'):
    if draw_obstacles:
        for obstacle in scenario.obstacles:
            obstacle.draw(ax, color='b')

    if draw_goal:
        scenario.goal_area.draw(ax, color='r')
        draw_point_arrow(scenario.goal_point, ax, color='r')

    # For Start Point:

    draw_car_func = globals()['draw_car']

    if draw_car:
        draw_car_func(start=scenario.start, car_dimensions=car_dimensions, ax=ax, color='g',
                      draw_collision_circles=not draw_collision_checking)

    # Draw Motion Primitives
    if draw_mps:
        for mp_name in mps.keys():
            mp_points = search.motion_primitive_at(mp_name, configuration=scenario.start)
            ax.plot(mp_points[:, 0], mp_points[:, 1], color='g')
            draw_point_arrow(mp_points[-1], ax=ax, color='g')

    # For One Random Neighbor Of The Start Point:
    if draw_car2 or draw_mps2:
        neighbor = tuple(search.motion_primitive_at(mp_name, configuration=scenario.start)[-1].tolist())

        # Draw Car:
        if draw_car2:
            draw_car_func(start=neighbor, car_dimensions=car_dimensions, ax=ax, color='b', draw_collision_circles=False)

        # Draw Motion Primitives
        if draw_mps2:
            for mp_name in mps.keys():
                mp_points = search.motion_primitive_at(mp_name, configuration=neighbor)
                ax.plot(mp_points[:, 0], mp_points[:, 1], color='b')
                draw_point_arrow(mp_points[-1], ax=ax, color='b')

    # Draw Collision Checking Points Of Random Motion Primitive As Circles With Radius
    if draw_collision_checking:
        cc_points = search.collision_checking_points_at(mp_name, configuration=scenario.start)
        colormap = plt.get_cmap('Set2')
        for i, cc_points_this in enumerate(np.array_split(cc_points, len(car_dimensions.circle_centers))):
            draw_circles(cc_points_this, radius=car_dimensions.radius, ax=ax, color=colormap(i))


def draw_point_arrow(point: Tuple[float, float, float], ax, color=None):
    x, y, theta = point
    u = np.cos(theta)
    v = np.sin(theta)
    ax.quiver(x, y, u, v, color=color)


def draw_circles(points: np.ndarray, radius: float, ax, color=None):
    ax.add_collection(
        EllipseCollection(widths=radius * 2, heights=radius * 2, angles=0, units='xy', edgecolors=color,
                          facecolors='none', offsets=points[:, :2],
                          offset_transform=ax.transData))
    ax.scatter(points[:, 0], points[:, 1], color=color)


def draw_car(start: Tuple[float, float, float], car_dimensions: CarDimensions, ax, color='b',
             draw_collision_circles=False):
    width, length = car_dimensions.bounding_box_size

    c_x, c_y, theta = start
    point_rect_bl = c_x - length / 2, c_y - width / 2
    ax.add_patch(Rectangle(point_rect_bl, length, width, rotation_point='center', edgecolor=color, facecolor='none',
                           angle=theta * 180. / math.pi))

    # transform all circle centers to world space
    circle_center_mtx = create_2d_transform_mtx(*start)
    circle_centers = transform_2d_pts(start[2], circle_center_mtx, car_dimensions.circle_centers)

    ax.scatter([start[0]], [start[1]], color=color)

    if draw_collision_circles:
        draw_circles(circle_centers, radius=car_dimensions.radius, ax=ax, color=color)
        ax.scatter(circle_centers[:, 0], circle_centers[:, 1], color=color)


def draw_astar_search_points(search: Union[MotionPrimitiveSearch, AStar], ax, visualize_heuristic: bool, visualize_cost_to_come: bool):
    debug_points = np.array([p.node for p in search.debug_data])

    if visualize_heuristic or visualize_cost_to_come:
        c = np.zeros((len(search.debug_data)))

        if visualize_heuristic:
            c += np.array([p.h for p in search.debug_data])

        if visualize_cost_to_come:
            c += np.array([p.g for p in search.debug_data])

    else:
        c = None

    ax.scatter(debug_points[:, 0], debug_points[:, 1], c=c, cmap='viridis_r')
