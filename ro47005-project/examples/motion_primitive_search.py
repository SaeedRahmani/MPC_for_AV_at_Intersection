import matplotlib.pyplot as plt
import numpy as np

from envs.t_intersection import t_intersection
from lib.car_dimensions import PriusDimensions, CarDimensions
from lib.helpers import measure_time
from lib.motion_primitive import load_motion_primitives
from lib.motion_primitive_search import MotionPrimitiveSearch
from lib.plotting import draw_point_arrow, draw_car, draw_circles


def run():
    mps = load_motion_primitives()
    scenario = t_intersection()
    car_dimensions: CarDimensions = PriusDimensions(skip_back_circle_collision_checking=False)

    fig, ax = plt.subplots()

    # draw the start point
    ax.scatter([scenario.start[0]], [scenario.start[1]], color='b')

    for obstacle in scenario.obstacles:
        obstacle.draw(ax, color='b')

    # draw goal area
    scenario.goal_area.draw(ax, color='r')
    # draw goal area with direction
    draw_point_arrow(scenario.goal_point, ax, color='r')

    search = MotionPrimitiveSearch(scenario, car_dimensions, mps, margin=car_dimensions.radius)

    # For Start Point:

    # Draw Car:
    draw_car(start=scenario.start, car_dimensions=car_dimensions, ax=ax, color='g')

    # Draw Motion Primitives
    for mp_name in mps.keys():
        mp_points = search.motion_primitive_at(mp_name, configuration=scenario.start)
        ax.plot(mp_points[:, 0], mp_points[:, 1], color='g')
        draw_point_arrow(mp_points[-1], ax=ax, color='g')

    # Draw Collision Checking Points Of Random Motion Primitive As Circles With Radius
    cc_points = search.collision_checking_points_at('right1', configuration=scenario.start)
    colormap = plt.get_cmap('Set2')
    for i, cc_points_this in enumerate(np.array_split(cc_points, len(car_dimensions.circle_centers))):
        draw_circles(cc_points_this, radius=car_dimensions.radius, ax=ax, color=colormap(i))

    # For One Random Neighbor Of The Start Point:
    neighbor = tuple(search.motion_primitive_at('right1', configuration=scenario.start)[-1].tolist())

    # Draw Car:
    draw_car(start=neighbor, car_dimensions=car_dimensions, ax=ax, color='c')

    # Draw Motion Primitives
    for mp_name in mps.keys():
        mp_points = search.motion_primitive_at(mp_name, configuration=neighbor)
        ax.plot(mp_points[:, 0], mp_points[:, 1], color='c')
        draw_point_arrow(mp_points[-1], ax=ax, color='c')

    # Perform The Search:

    @measure_time
    def run_the_search():
        try:
            cost, trajectory = search.run(debug=True)

            # Draw Resulting Trajectory
            ax.plot(trajectory[:, 0], trajectory[:, 1], color='b')
            print("Total cost:", cost)
        except KeyboardInterrupt:
            # Break The Search On Keyboard Interrupt
            pass

    run_the_search()
    print("Nodes searched:", len(search.debug_data))

    # Draw All Search Points
    debug_points = np.array([p.node for p in search.debug_data])
    debug_point_dists = np.array([[p.h for p in search.debug_data]])
    ax.scatter(debug_points[:, 0], debug_points[:, 1], c=debug_point_dists, cmap='viridis_r')

    ax.axis('equal')

    plt.show()


if __name__ == '__main__':
    run()
