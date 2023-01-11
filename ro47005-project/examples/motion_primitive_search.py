from matplotlib import pyplot as plt

from envs.t_intersection import t_intersection
from lib.car_dimensions import PriusDimensions, CarDimensions
from lib.model_predictive_speed_and_steer_control import main
from lib.motion_primitive import load_motion_primitives
from lib.motion_primitive_search import MotionPrimitiveSearch
from lib.plotting import draw_scenario, draw_astar_search_points

if __name__ == '__main__':
    # fig, ax = plt.subplots()
    for version in ['prius', 'bicycle_model']:
        mps = load_motion_primitives(version=version)
        scenario = t_intersection()
        car_dimensions: CarDimensions = PriusDimensions(skip_back_circle_collision_checking=False)

        search = MotionPrimitiveSearch(scenario, car_dimensions, mps, margin=car_dimensions.radius)

        try:
            cost, path, trajectory = search.run(debug=True)

            cx = trajectory[:, 0]
            cy = trajectory[:, 1]
            cyaw = trajectory[:, 2]
            # sp = np.ones_like(cyaw) * 8.3

            main(cx / 0.3, cy / 0.3, cyaw)
        except KeyboardInterrupt:
            pass  # Break The Search On Keyboard Interrupt
