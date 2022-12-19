import matplotlib.pyplot as plt
import numpy as np

from envs.t_intersection import t_intersection
from lib.helpers import measure_time
from lib.motion_primitive import load_motion_primitives, create_transform_mtx, transform_pts, \
    MotionPrimitiveSearch
from lib.obstacles import draw_obstacle, draw_point_arrow


@measure_time
def run():
    mps = load_motion_primitives()
    scenario = t_intersection()

    fig, ax = plt.subplots()

    # draw the start point
    ax.scatter([scenario.start[0]], [scenario.start[1]], color='b')

    for obstacle in scenario.obstacles:
        draw_obstacle(obstacle, ax, color='b')

    # draw goal area
    draw_obstacle(scenario.goal_area, ax, color='r')
    # draw goal area with direction
    draw_point_arrow(scenario.goal_point, ax, color='r')

    # draw the immediate motion primitives, right at the start position of the car
    mtx = create_transform_mtx(*scenario.start)
    for mp in mps.values():
        points = transform_pts(scenario.start[2], mtx, mp.points)
        ax.plot(points[:, 0], points[:, 1], color='b')

    search = MotionPrimitiveSearch(scenario, mps, margin=(0.5, 0.5))
    try:
        cost, trajectory = search.run(debug=True)

        # draw resulting trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], color='b')
        print("Total cost:", cost)
        print("Nodes searched:", len(search.debug_data))
    except KeyboardInterrupt:
        # break the search on keyboard interrupt
        pass

    # draw all search points
    debug_points = np.array([p[0] for p in search.debug_data])
    debug_point_dists = np.array([[p[1] for p in search.debug_data]])
    ax.scatter(debug_points[:, 0], debug_points[:, 1], c=debug_point_dists)

    ax.axis('equal')

    plt.show()


if __name__ == '__main__':
    run()
