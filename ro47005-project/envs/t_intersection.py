import numpy as np

from lib.obstacles import Obstacle, Scenario


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
    goal_area = Obstacle(type="GEOM_BOX", dim=[width_road * 2, width_road, 0],
                    pos=[goal[0], goal[1], 0])

    obstacles = [
        # T-intersection
        # Leg of T
        Obstacle(type="GEOM_BOX", dim=[width_traffic_island, length, height],
                 pos=[0, -(length / 2 + distance_center), 0]),
        Obstacle(type="GEOM_BOX", dim=[width_pavement, length, height],
                 pos=[(width_traffic_island / 2 + width_road + width_pavement / 2), -(length / 2 + distance_center),
                      0]),
        Obstacle(type="GEOM_BOX", dim=[width_pavement, length, height],
                 pos=[-(width_traffic_island / 2 + width_road + width_pavement / 2), -(length / 2 + distance_center),
                      0]),
        Obstacle(type="GEOM_CYLINDER", dim=[(width_traffic_island / 2), height],
                 pos=[0, -distance_center, 0]),

        # left part of T
        Obstacle(type="GEOM_BOX", dim=[length, width_traffic_island, height],
                 pos=[-(length / 2 + distance_center), 0, 0.]),
        Obstacle(type="GEOM_BOX", dim=[length, width_pavement, height],
                 pos=[-(length / 2 + distance_center), -(width_traffic_island / 2 + width_road + width_pavement / 2),
                      0.]),
        Obstacle(type="GEOM_CYLINDER", dim=[(distance_center - width_traffic_island / 2 - width_road), height],
                 pos=[-distance_center, -distance_center, 0]),
        Obstacle(type="GEOM_CYLINDER", dim=[(width_traffic_island / 2), height],
                 pos=[-distance_center, 0, 0]),

        # right part of T
        Obstacle(type="GEOM_BOX", dim=[length, width_traffic_island, height],
                 pos=[(length / 2 + distance_center), 0, 0.]),
        Obstacle(type="GEOM_BOX", dim=[length, width_pavement, height],
                 pos=[(length / 2 + distance_center), -(width_traffic_island / 2 + width_road + width_pavement / 2),
                      0.]),
        Obstacle(type="GEOM_CYLINDER", dim=[(distance_center - width_traffic_island / 2 - width_road), height],
                 pos=[distance_center, -distance_center, 0]),
        Obstacle(type="GEOM_CYLINDER", dim=[(width_traffic_island / 2), height],
                 pos=[distance_center, 0, 0]),

        # upper part of T
        Obstacle(type="GEOM_BOX", dim=[(2 * length + 2 * distance_center), width_pavement, height],
                 pos=[0, (width_traffic_island / 2 + width_road + width_pavement / 2),
                      0.]),
    ]

    return Scenario(
        start=start,
        goal_point=goal,
        goal_area=goal_area,
        obstacles=obstacles,
    )
