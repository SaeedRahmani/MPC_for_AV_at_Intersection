from matplotlib import pyplot as plt
import sys
import time
from itertools import product
sys.path.append('..')

from envs.intersection_multi_lanes import intersection
from lib.car_dimensions import BicycleModelDimensions, CarDimensions
from lib.motion_primitive import load_motion_primitives
from lib.motion_primitive_search_multi_lane import MotionPrimitiveSearch
from lib.plotting import draw_scenario, draw_astar_search_points

def measure_time(func):
    """Decorator to measure runtime of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return end_time - start_time, result
    return wrapper

@measure_time
def run_search(search):
    return search.run(debug=True)

if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(12, 8))

    start_pos = 1
    turn_indicator = 1
    scenario = intersection(turn_indicator=turn_indicator, start_pos=start_pos, 
                            start_lane=1, goal_lane=1, number_of_lanes=2)
    mps = load_motion_primitives(version='bicycle_model')
    car_dimensions: CarDimensions = BicycleModelDimensions(skip_back_circle_collision_checking=False)

    # Parameter values
    wh_steering_values = [10, 15, 20]
    wh_dist_values = [1, 1.5, 2]
    wc_obstacle_values = [0.1, 0.5, 1]

    # Colors for each parameter
    colors = ['red', 'green', 'blue']
    # Line styles for different values
    line_styles = ['-', '--', ':']

    parameter_combinations = list(product(wh_steering_values, wh_dist_values, wc_obstacle_values))

    for index, (wh_steering, wh_dist, wc_obstacle) in enumerate(parameter_combinations):
        search = MotionPrimitiveSearch(scenario, car_dimensions, mps, margin=car_dimensions.radius,
                                       wh_dist=wh_dist, wh_theta=2.7, wh_steering=wh_steering, 
                                       wh_obstacle=0, wh_center=0, wc_dist=1, wc_steering=5, 
                                       wc_obstacle=wc_obstacle, wc_center=0)

        runtime, (cost, path, trajectory) = run_search(search)
        cx = trajectory[:, 0]
        cy = trajectory[:, 1]

        # Determine color and style
        color_index = index // 9  # 9 combinations per parameter (3^2)
        style_index = index % 3  # 3 values per parameter
        color = colors[color_index]
        line_style = line_styles[style_index]
        
        label_text = f'Steering: {wh_steering}, Dist: {wh_dist}, Obstacle: {wc_obstacle}, Time: {runtime:.2f}s'
        ax.plot(cx, cy, color=color, linestyle=line_style, label=label_text)

    draw_scenario(scenario, mps, car_dimensions, search, ax, draw_obstacles=True, draw_goal=True, 
                  draw_car=True, draw_mps=False, draw_collision_checking=False, draw_car2=False, draw_mps2=False, 
                  mp_name='right1')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax.axis('equal')
    plt.tight_layout()
    plt.show()


