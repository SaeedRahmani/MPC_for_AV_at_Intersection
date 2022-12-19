import numpy as np

from lib.obstacles import BoxObstacle, CircleObstacle
from lib.scenario import Scenario


def t_intersection() -> Scenario:
    """ Function the build a T-intersection. Takes the gym environment as input. """
    width_road = 2
    width_traffic_island = 0.5
    width_pavement = 2
    length = 5
    height = 0.2
    distance_center = 3

    start = (1.25, -6, 0.5 * np.pi)
    goal = (-(distance_center + length * 0.8), (width_traffic_island + width_road) / 2, -np.pi)
    goal_area = BoxObstacle(xy_width=(width_road * 2, width_road), height=height,
                            xy_center=(goal[0], goal[1]))

    obstacles = [
        # T-intersection
        # Leg of T
        BoxObstacle(xy_width=(width_traffic_island, length), height=height,
                    xy_center=(0, -(length / 2 + distance_center))),
        BoxObstacle(xy_width=(width_pavement, length), height=height,
                    xy_center=(
                    (width_traffic_island / 2 + width_road + width_pavement / 2), -(length / 2 + distance_center))),
        BoxObstacle(xy_width=(width_pavement, length), height=height,
                    xy_center=(
                    -(width_traffic_island / 2 + width_road + width_pavement / 2), -(length / 2 + distance_center))),
        CircleObstacle(radius=(width_traffic_island / 2), height=height,
                       xy_center=(0, -distance_center)),

        # left part of T
        BoxObstacle(xy_width=(length, width_traffic_island), height=height,
                    xy_center=(-(length / 2 + distance_center), 0.)),
        BoxObstacle(xy_width=(length, width_pavement), height=height,
                    xy_center=(
                    -(length / 2 + distance_center), -(width_traffic_island / 2 + width_road + width_pavement / 2))),
        CircleObstacle(radius=(distance_center - width_traffic_island / 2 - width_road), height=height,
                       xy_center=(-distance_center, -distance_center)),
        CircleObstacle(radius=(width_traffic_island / 2), height=height,
                       xy_center=(-distance_center, 0)),

        # right part of T
        BoxObstacle(xy_width=(length, width_traffic_island), height=height,
                    xy_center=((length / 2 + distance_center), 0)),
        BoxObstacle(xy_width=(length, width_pavement), height=height,
                    xy_center=(
                    (length / 2 + distance_center), -(width_traffic_island / 2 + width_road + width_pavement / 2))),
        CircleObstacle(radius=(distance_center - width_traffic_island / 2 - width_road), height=height,
                       xy_center=(distance_center, -distance_center)),
        CircleObstacle(radius=(width_traffic_island / 2), height=height,
                       xy_center=(distance_center, 0)),

        # upper part of T
        BoxObstacle(xy_width=((2 * length + 2 * distance_center), width_pavement), height=height,
                    xy_center=(0, (width_traffic_island / 2 + width_road + width_pavement / 2))),
    ]

    return Scenario(
        start=start,
        goal_point=goal,
        goal_area=goal_area,
        obstacles=obstacles,
    )
