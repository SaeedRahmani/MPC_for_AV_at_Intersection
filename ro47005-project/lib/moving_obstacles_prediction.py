import matplotlib.pyplot as plt
import numpy as np
import math
class MovingObstaclesPrediction:
    def __init__(self, x, y, v, yaw, a, steering_angle):
        self.x = x
        self.y = y
        self.v = v
        self.yaw = yaw
        self.a = a
        self.steering_angle = steering_angle
        self.scale_factor = 0.3
        self.L = (1.45 + 1.41) * self.scale_factor

    def step(self, sample_time):
        ''' Function to predict 1 sample time step ahead'''

        self.x += self.v * math.cos(self.yaw) * sample_time
        self.y += self.v * math.sin(self.yaw) * sample_time
        self.v += self.a * sample_time
        self.yaw += (self.v / self.L) * math.tan(self.steering_angle) * sample_time

        return self.x, self.y, self.v, self.yaw

    def state_prediction(self, time_horizon):
        '''
        Function to predict state prediction for a specified time horizon
        '''
        sample_time = 5e-3

        t_data = np.arange(0, time_horizon, sample_time)  # array size (0, (time_end - 0) / sample_time)
        x_data = np.zeros_like(t_data)
        y_data = np.zeros_like(t_data)
        yaw_data = np.zeros_like(t_data)

        for time_index in range(0, len(t_data)):
            x, y, v, yaw = self.step(sample_time)
            x_data[time_index] = x
            y_data[time_index] = y
            yaw_data[time_index] = yaw
            t_data[time_index] = time_index * sample_time

        return x_data, y_data, yaw_data, t_data


if __name__ == '__main__':
    '''
    Change x, y, yaw, v, steering_angle, and a based on output of Design MO()!
    '''
    x = 0
    y = 0
    yaw = 0
    v = 3
    steering_angle = 0
    a = 0

    model = MovingObstaclesPrediction(x, y, v, yaw, a, steering_angle)
    x_data, y_data, yaw_data, t_data = model.state_prediction(time_horizon=1)
    plt.plot(t_data, x_data)
    plt.show()