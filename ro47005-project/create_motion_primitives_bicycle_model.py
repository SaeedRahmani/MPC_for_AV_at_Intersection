import numpy as np
from tqdm.auto import tqdm
import pickle

from bicycle.main import Bicycle
from lib.motion_primitive import MotionPrimitive
from create_motion_primitives_prius import CONFIGURATIONS


def run_bicycle_model(forward_speed: float, steering_angle: float, n_seconds=1.):
    # Defined in the Bicycle class
    DT = 0.005

    model = Bicycle()
    # model.theta = 0
    time = np.linspace(0, n_seconds, int(n_seconds / DT) + 1)

    points = []
    for i in time:
        points.append(np.array([model.xc, model.yc, model.theta]))
        model.step(forward_speed, steering_angle)

    points = np.array(points)

    return points


if __name__ == "__main__":
    for mp in tqdm(CONFIGURATIONS):
        points = run_bicycle_model(
            n_seconds=mp.n_seconds,
            forward_speed=mp.forward_speed,
            steering_angle=mp.steering_angle,
        )

        file_name = f'./data/motion_primitives_bicycle_model_real_size/{mp.name}.pkl'

        # compute total length
        mp.total_length = np.linalg.norm(points[:-1, :2] - points[1:, :2], axis=1).sum()

        mp.points = points

        with open(file_name, 'wb') as file:
            pickle.dump(mp, file)
