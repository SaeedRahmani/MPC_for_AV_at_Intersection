import numpy as np

def t_intersection(env):
    """ Function the build a T-intersection. Takes the gym environment as input. """
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
    env.add_shapes(shape_type="GEOM_CYLINDER", dim=[(width_traffic_island / 2), height],
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
