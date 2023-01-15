import pickle
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from bicycle.main import Bicycle
from create_motion_primitives_prius import CONFIGURATIONS
from lib.car_dimensions import CarDimensions, BicycleModelDimensions


def run_bicycle_model(car_dimensions: CarDimensions, forward_speed: float, steering_angle: float, n_seconds=1.):
    # Defined in the Bicycle class
    DT = 0.005

    model = Bicycle(car_dimensions=car_dimensions)
    # model.theta = 0
    time = np.linspace(0, n_seconds, int(n_seconds / DT) + 1)

    points = []
    for i in time:
        points.append(np.array([model.xc, model.yc, model.theta]))
        model.step(forward_speed, steering_angle)

    points = np.array(points)

    return points


if __name__ == "__main__":
    car_dimensions: CarDimensions = BicycleModelDimensions()

    for mp in tqdm(CONFIGURATIONS):
        points = run_bicycle_model(
            car_dimensions=car_dimensions,
            n_seconds=mp.n_seconds,
            forward_speed=mp.forward_speed,
            steering_angle=mp.steering_angle,
        )

        project_root_dir = Path(__file__).parent
        file_name = project_root_dir.joinpath(f'data/motion_primitives_bicycle_model/{mp.name}.pkl')

        # compute total length
        mp.total_length = np.linalg.norm(points[:-1, :2] - points[1:, :2], axis=1).sum()

        # no need to offset points - they're already at the back wheel
        mp.points = points

        with open(file_name, 'wb') as file:
            pickle.dump(mp, file)
