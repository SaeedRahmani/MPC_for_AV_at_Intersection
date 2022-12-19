import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from envs.t_intersection import t_intersection
from lib.obstacles import obstacle_to_convex


def plot_obstacle_region(obstacle, margin=None):
    if margin is None:
        color = '#000000'
    else:
        color = '#bfbfbf'
    c_x, c_y = obstacle.pos[:2]
    w_x, w_y = obstacle.dim[:2]
    extra = 1
    step = 0.05
    x = np.linspace(c_x - w_x - extra, c_x + w_x + extra, int((w_x + 2 * extra)/step))
    y = np.linspace(c_y - w_y - extra, c_y + w_y + extra, int((w_y + 2 * extra)/step))

    half_spaces = obstacle_to_convex(obstacle,  margin)

    # if (x[-1] - x[0]) > (y[-1] - y[0]):
    #     plt.xlim(x[0], x[-1])
    #     plt.ylim(x[0], x[-1])
    # else:
    #     plt.xlim(y[0], y[-1])
    #     plt.ylim(y[0], y[-1])
    plt.axis('equal')
    plt.xlim(x[0], x[-1])
    plt.ylim(y[0], y[-1])

    for i in tqdm(x):
        for j in y:
            free = np.any((half_spaces @ np.array([i, j, 1]) >= 0))
            if free:
                continue
            else:
                plt.scatter(i, j, color=color)


scenario = t_intersection()
obstacle = scenario.obstacles[3] # Example of a circle
# obstacle = scenario.obstacles[0] # Example of a box

plt.figure()
plot_obstacle_region(obstacle, margin=(0.1, 0.2))
plot_obstacle_region(obstacle)
plt.show()


