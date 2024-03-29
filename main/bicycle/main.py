import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('..')

from lib.car_dimensions import CarDimensions, BicycleModelDimensions


# Reference paper:
# https://uwspace.uwaterloo.ca/bitstream/handle/10012/16847/Ahmadi_Behnaz.pdf?sequence=1
# see page 29/80 to see the bicycle model used

class Bicycle:
    def __init__(self, car_dimensions: CarDimensions, sample_time: float = 0.2): # old: 10e-3
        self.xc = 0
        self.yc = 0
        self.theta = 0
        self.sample_time = sample_time

        self.L = car_dimensions.distance_back_to_front_wheel

    def reset(self):
        self.xc = 0 # vehicle x position
        self.yc = 0 # vehicle y position
        self.theta = 0 # angle between vehicle frame and x-axis

    def step(self, v, delta):
        # ==================================
        #  Implement kinematic model here
        # ==================================

        # implementing the differential equations
        xc_dot = v * np.cos(self.theta)
        yc_dot = v * np.sin(self.theta)
        theta_dot = (v / self.L) * np.tan(delta)

        # update equations using the sampling time
        self.xc += xc_dot * self.sample_time
        self.yc += yc_dot * self.sample_time
        self.theta += theta_dot * self.sample_time

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    car_dimensions: CarDimensions = BicycleModelDimensions()
    sample_time = 0.01
    time_end = 1
    model = Bicycle(car_dimensions=car_dimensions)
    model1 = Bicycle(car_dimensions=car_dimensions)
    model2 = Bicycle(car_dimensions=car_dimensions)

    t_data = np.arange(0, time_end, sample_time)
    x_data = np.zeros_like(t_data)
    y_data = np.zeros_like(t_data)
    x_data1 = np.zeros_like(t_data)
    y_data1 = np.zeros_like(t_data)
    x_data2 = np.zeros_like(t_data)
    y_data2 = np.zeros_like(t_data)

    for i in range(t_data.shape[0]):
        x_data[i] = model.xc
        y_data[i] = model.yc
        x_data1[i] = model1.xc
        y_data1[i] = model1.yc
        x_data2[i] = model2.xc
        y_data2[i] = model2.yc
        model.step(1.5, 2.35)
        model1.step(1.5, 0)
        model2.step(1.5, 0.78)

    plt.axis('equal')
    plt.scatter(x_data, y_data, label='Delta = 135 degree')
    plt.scatter(x_data1, y_data1, label='Delta = 0 degree')
    plt.scatter(x_data2, y_data2, label='Delta = 45 degree')
    plt.legend()
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
