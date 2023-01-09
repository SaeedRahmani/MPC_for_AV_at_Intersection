from matplotlib import pyplot as plt

from envs.t_intersection import t_intersection
from lib.car_dimensions import PriusDimensions, CarDimensions
from lib.motion_primitive import load_motion_primitives
from lib.motion_primitive_search import MotionPrimitiveSearch
from lib.plotting import draw_scenario, draw_astar_search_points

if __name__ == '__main__':
    mps = load_motion_primitives()
    scenario = t_intersection()
    car_dimensions: CarDimensions = PriusDimensions(skip_back_circle_collision_checking=False)

    search = MotionPrimitiveSearch(scenario, car_dimensions, mps, margin=car_dimensions.radius)

    fig, ax = plt.subplots()
    draw_scenario(scenario, mps, car_dimensions, search, ax,
                  draw_obstacles=True, draw_goal=True, draw_car=True, draw_mps=False, draw_collision_checking=False,
                  draw_car2=False, draw_mps2=False, mp_name='right1')

    # Perform The Search:

    try:
        cost, path, trajectory = search.run(debug=True)

        cx = trajectory[:, 0]
        cy = trajectory[:, 1]
        # cyaw = trajectory[:, 2]
        # sp = np.ones_like(cyaw) * 8.3

        # Draw trajectory
        ax.plot(cx, cy, color='b')
    except KeyboardInterrupt:
        pass  # Break The Search On Keyboard Interrupt

    # Draw All Search Points
    draw_astar_search_points(search, ax, visualize_heuristic=True, visualize_cost_to_come=False)

    ax.axis('equal')
    plt.show()
