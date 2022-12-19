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
    goal_area = BoxObstacle(dim=(width_road * 2, width_road, 0),
                            xy=(goal[0], goal[1]))

    obstacles = [
        # T-intersection
        # Leg of T
        BoxObstacle(dim=(width_traffic_island, length, height),
                    xy=(0, -(length / 2 + distance_center))),
        BoxObstacle(dim=(width_pavement, length, height),
                    xy=(
                    (width_traffic_island / 2 + width_road + width_pavement / 2), -(length / 2 + distance_center))),
        BoxObstacle(dim=(width_pavement, length, height),
                    xy=(
                    -(width_traffic_island / 2 + width_road + width_pavement / 2), -(length / 2 + distance_center))),
        CircleObstacle(radius=(width_traffic_island / 2), height=height,
                       xy=(0, -distance_center)),

        # left part of T
        BoxObstacle(dim=(length, width_traffic_island, height),
                    xy=(-(length / 2 + distance_center), 0.)),
        BoxObstacle(dim=(length, width_pavement, height),
                    xy=(
                    -(length / 2 + distance_center), -(width_traffic_island / 2 + width_road + width_pavement / 2))),
        CircleObstacle(radius=(distance_center - width_traffic_island / 2 - width_road), height=height,
                       xy=(-distance_center, -distance_center)),
        CircleObstacle(radius=(width_traffic_island / 2), height=height,
                       xy=(-distance_center, 0)),

        # right part of T
        BoxObstacle(dim=(length, width_traffic_island, height),
                    xy=((length / 2 + distance_center), 0)),
        BoxObstacle(dim=(length, width_pavement, height),
                    xy=(
                    (length / 2 + distance_center), -(width_traffic_island / 2 + width_road + width_pavement / 2))),
        CircleObstacle(radius=(distance_center - width_traffic_island / 2 - width_road), height=height,
                       xy=(distance_center, -distance_center)),
        CircleObstacle(radius=(width_traffic_island / 2), height=height,
                       xy=(distance_center, 0)),

        # upper part of T
        BoxObstacle(dim=((2 * length + 2 * distance_center), width_pavement, height),
                    xy=(0, (width_traffic_island / 2 + width_road + width_pavement / 2))),
    ]

    return Scenario(
        start=start,
        goal_point=goal,
        goal_area=goal_area,
        obstacles=obstacles,
    )
