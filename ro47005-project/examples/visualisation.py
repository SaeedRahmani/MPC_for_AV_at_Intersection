import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from envs.t_intersection import t_intersection
from lib.car_dimensions import PriusDimensions, CarDimensions
from lib.motion_primitive import load_motion_primitives
from lib.motion_primitive_search import MotionPrimitiveSearch
from lib.plotting import draw_scenario, draw_astar_search_points
from lib.moving_obstacles import MovingObstacleTIntersection
from lib.visualisation import create_animation

if __name__ == "__main__":
    fig, ax = plt.subplots()
    mps = load_motion_primitives(version='bicycle_model')
    scenario = t_intersection()
    car_dimensions: CarDimensions = PriusDimensions(skip_back_circle_collision_checking=False)

    search = MotionPrimitiveSearch(scenario, car_dimensions, mps, margin=car_dimensions.radius)

    draw_scenario(scenario, mps, car_dimensions, search, ax,
                  draw_obstacles=True, draw_goal=False, draw_car=False, draw_mps=False, draw_collision_checking=False,
                  draw_car2=False, draw_mps2=False)

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

    # simulation time:
    t = 5  # s
    dt = 10e-3  # has to be the same as in the bicycle model
    length = int(5 / dt)
    print(length)

    # Create the moving obstacles
    obstacles = [MovingObstacleTIntersection(1, False, 8.3, ),
                 MovingObstacleTIntersection(1, True, 8.3, offset=1),
                 MovingObstacleTIntersection(-1, False, 8.3),
                 MovingObstacleTIntersection(-1, True, 8.3, offset=2)]
    positions_obstacles = np.zeros((len(obstacles), length, 5))  # To log the positions of the moving obstacles
    positions_car = np.zeros((length, 3))  # x, y, theta

    # ================= The simulation loop =================
    for t in range(length):  # 5 seconds, because dt of bicycle model is 10e-3
        # Update the moving obstacles
        for i_obs, obstacle in enumerate(obstacles):
            positions_obstacles[i_obs, t] = obstacle.step()

        # Check for collision

        # Step the MPC
        positions_car[t] = trajectory[t % 280, :3]  # Just an arbitrary value for plotting

    #  ================= Start animation =================
    animate = create_animation(ax,
                               positions_car,
                               positions_obstacles,
                               draw_car=True,
                               draw_car_center=True,
                               draw_obstacles=True,
                               draw_obstacles_center=True)

    # The interval is not working well: It seems to be the delay between frames after the calculation,
    # so it depends on the computation time
    ani = animation.FuncAnimation(fig, animate, frames=length, interval=5, blit=True, repeat=True)

    # Code to save the animation
    # FFwriter = animation.FFMpegWriter(fps=int(1/dt))
    # ani.save('visualisation.mp4', writer=FFwriter)
    #  ================= End animation =================

    # ax.axis([-12, 12, -12, 6])
    ax.axis([-8, 8, -8, 4])
    ax.set_aspect(1)
    # ax.axis('equal')
    plt.show()
