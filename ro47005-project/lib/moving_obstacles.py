import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.animation as animation
import numpy as np
from typing import Tuple

from bicycle.main import Bicycle
import warnings


class MovingObstacleTIntersection:
    def __init__(self, direction: int, turning: bool, speed: float, offset=None, dt=10e-3):
        """
        Function that creates moving obstacles
        :param direction: positive for moving to the right, negative for moving to the left
        :param turning: True for turning, False for going straight
        :param speed: sets the forward speed and is not bounded
        :param offset: sets the time in seconds it should start moving after the start of the simulation. None or 0 for no offset
        :param dt: the dt used in the simulator. !WARNING! is 10e-3 in the Bicycle model
        """
        self.direction = 1 if direction >= 0 else -1
        self.turning = turning
        self.forward_velocity = speed
        self.model = Bicycle()
        self.offset = None if offset is None else offset if offset > 0 else None  # None except if offset > 0
        self.dt = dt
        if abs(dt - 10e-3) > 1e-6:
            warnings.warn("The dt is most likely not compatible with the bicycle model!")
        self.counter = 0
        if self.direction == 1:
            self.model.xc = -10
            self.model.yc = -1.25
            self.model.theta = 0
            self.x_turn = -3
        else:
            self.model.xc = 10
            self.model.yc = 1.25
            self.model.theta = np.pi
            self.x_turn = 3

    def steering_angle(self) -> float:
        steering_angle = 0.  # rad
        if self.direction == 1:
            if self.model.xc >= self.x_turn and self.model.theta > (-np.pi / 2):
                steering_angle = -0.45  # steering angle right (short turn)
        else:
            if self.model.xc <= self.x_turn and self.model.theta < (3 * np.pi / 2):
                steering_angle = 0.2  # steering angle left (long turn)

        return steering_angle

    def step(self) -> Tuple[float, float, float, float, float]:
        steering_angle = 0 if self.turning is not True else self.steering_angle()
        if self.offset is None or self.counter > (self.offset / self.dt):
            forward_velocity = self.forward_velocity
        else:
            forward_velocity = 0
        self.model.step(forward_velocity, steering_angle)

        self.counter += 1
        return self.model.xc, self.model.yc, forward_velocity, self.model.theta, steering_angle


if __name__ == "__main__":
    obstacles = [MovingObstacleTIntersection(1, True, 5),
                 MovingObstacleTIntersection(1, False, 5),
                 MovingObstacleTIntersection(-1, False, 5),
                 MovingObstacleTIntersection(-1, True, 5)]

    length = 600  # steps in the simulation
    colors = np.array([np.zeros(length), np.linspace(0, 1, length), np.ones(length)]).T.astype(float)
    positions = np.zeros((len(obstacles), length, 5))

    # Run the simulation
    for t in range(length):  # 5 seconds, because dt of bicycle model is 10e-3
        for i_obs, obstacle in enumerate(obstacles):
            positions[i_obs, t] = obstacle.step()

    print(positions.shape)

    ###################### Create scatter animation with changing colors
    fig, ax = plt.subplots()

    # Create the initial plot
    x0 = positions[:, 0, 0]
    y0 = positions[:, 0, 1]
    scat = ax.scatter(x0, y0)
    mat, = ax.plot(x0, y0, 'o')

    # different color arrays
    colors_array = []
    combinations = [(0, 0), (0, 1), (1, 1)]
    for i in range(6):
        colors_array.append(
            np.array([np.linspace(*combinations[i % 3], length), np.linspace(*combinations[(i + 1) % 3], length),
                      np.linspace(*combinations[(i + 2) % 3], length)]).T.astype(float))

    # Setting to True gives every obstacle a different color ranges, but it becomes really messy
    # and somehow there are only 3 unique combinations
    different_colors = False


    def animate(i):
        """ Function that is called for every animation frame """
        # Set x and y data...
        interval = 5
        if different_colors:
            x = positions[:, :i + 1:interval, 0].flatten('C')
            y = positions[:, :i + 1:interval, 1].flatten('C')
        else:
            x = positions[:, :i + 1:interval, 0].flatten('F')
            y = positions[:, :i + 1:interval, 1].flatten('F')
            # x_car = ((i+1)%5) + 1
            # y_car = 0
            # mat.set_data(x_car, y_car)
            # mat.set_color((1, 0, 0))

        scat.set_offsets(np.array([x, y]).T)
        if different_colors:
            scat.set_facecolors(np.vstack((colors_array[0][:i + 1:interval], colors_array[1][:i + 1:interval],
                                           colors_array[2][:i + 1:interval], colors_array[5][:i + 1:interval])))
        else:
            scat.set_facecolors(np.repeat(colors[:i + 1:interval], positions.shape[0], axis=0))


    # Set the axis values
    ax.axis([-20, 20, -25, 5])
    ani = animation.FuncAnimation(fig, animate, frames=600, interval=10e-3)

    start = mlines.Line2D([], [], color=colors[0], marker='o', ls='', label='Start point')
    end = mlines.Line2D([], [], color=colors[-1], marker='o', ls='', label='End point')
    plt.legend(handles=[start, end])

    # Code to save the animation
    # FFwriter = animation.FFMpegWriter(fps=100)
    # ani.save('moving_obstacles_trajectory.mp4', writer=FFwriter)

    plt.show()

    ################# Create animation: Not possible with changing colors
    # fig, ax = plt.subplots()
    #
    # # Create the initial plot
    # x0 = positions[:, 0, 0]
    # y0 = positions[:, 0, 1]
    # mat, = ax.plot(x0, y0, 'o')
    #
    # def animate(i):
    #     """ Function that is called for every animation frame """
    #     interval = 5
    #     x = positions[:, :i+1:interval, 0]
    #     y = positions[:, :i+1:interval, 1]
    #     mat.set_data(x, y)
    #     mat.set_color((0, 0, 1))
    #     return mat,
    #
    # # Set the axis values
    # ax.axis([-20, 20, -25, 5])
    # ani = animation.FuncAnimation(fig, animate, frames=600, interval=10e-3, repeat=True)

    # Code to save the animation
    # FFwriter = animation.FFMpegWriter(fps=100)
    # ani.save('moving_obstacles_trajectory.mp4', writer=FFwriter)

    # plt.show()

    #################3 Create plot
    plt.figure()
    for position_obstacle in positions:
        plt.scatter(position_obstacle[:, 0], position_obstacle[:, 1], c=colors)
    start = mlines.Line2D([], [], color=colors[0], marker='o', ls='', label='Start point')
    end = mlines.Line2D([], [], color=colors[-1], marker='o', ls='', label='End point')
    plt.legend(handles=[start, end])

    plt.axis('equal')

    # Code to save the plot
    # plt.savefig('moving_obstacles_trajectory.png')

    plt.show()
