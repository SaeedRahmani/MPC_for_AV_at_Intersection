import sys
sys.path.append('..')

import numpy as np
from matplotlib import pyplot as plt

from lib.obstacles import BoxObstacle, CircleObstacle
from lib.scenario import Scenario

def free_area(angle: float = 0.0, start_pos: float = 0.0, goal_distance = 20) -> Scenario:
    allowed_goal_theta_difference = np.pi / 16

    # 1: south, 2: west, 3: north, 4, east
    start_positions = (start_pos, start_pos, 0.)
    angle = angle
    
    
    # 1: turn left, 2: go straight, 3: turn right
    goal_distance = goal_distance
    start = start_positions
    goal = (start_pos + goal_distance*np.cos(angle), start_pos + goal_distance*np.sin(angle), angle)
    print(goal, type(goal), goal[0], goal[1], angle)
    goal_area = BoxObstacle(xy_width=(4 * 1.8, 4), height=0.5, xy_center=(goal[0], goal[1]))
    obstacles = []

    return Scenario(
        start=start,
        goal_point=goal,
        goal_area=goal_area,
        allowed_goal_theta_difference=allowed_goal_theta_difference,
        obstacles=obstacles,
    )
    
# def plot_free_area():
    # Initialize the figure and axis
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle

    fig, ax = plt.subplots(figsize=(10, 10))      
    # Function to add a box obstacle with label
    def add_box(center, width, height, label=None, color='gray'):
        lower_left_corner = (center[0] - width / 2, center[1] - height / 2)
        ax.add_patch(Rectangle(lower_left_corner, width, height, color=color))
        if label:
            ax.text(center[0], center[1], label, ha='center', va='center')

    # Function to add a circle obstacle with label
    def add_circle(center, radius, label=None, color='gray'):
        ax.add_patch(Circle(center, radius, color=color))
        if label:
            ax.text(center[0], center[1], label, ha='center', va='center')

    # Add the medians
    add_box((0, -(length / 2 + distance_center)), width_traffic_island, length, 'South Median')  
    add_circle((0, -distance_center), width_traffic_island / 2)
    add_box((0, (length / 2 + distance_center)), width_traffic_island, length, 'North Median')  
    add_circle((0, distance_center), width_traffic_island / 2)
    add_box((-(length / 2 + distance_center), 0), length, width_traffic_island, 'West Median')  
    add_circle((-distance_center, 0), width_traffic_island / 2)
    add_box(((length / 2 + distance_center), 0), length, width_traffic_island, 'East Median')  
    add_circle((distance_center, 0), width_traffic_island / 2)

    # Add the corners
    add_circle((-distance_center, -distance_center), distance_center - width_traffic_island / 2 - width_road, 'SW Corner')  
    add_circle((-distance_center, distance_center), distance_center - width_traffic_island / 2 - width_road, 'NW Corner')  
    add_circle((distance_center, distance_center), distance_center - width_traffic_island / 2 - width_road, 'NE Corner')  
    add_circle((distance_center, -distance_center), distance_center - width_traffic_island / 2 - width_road, 'SE Corner')  

    # Add the pavements
    add_box((-(width_traffic_island / 2 + width_road + width_pavement / 2), -(length / 2 + distance_center)), width_pavement, length)  # south
    add_box(((width_traffic_island / 2 + width_road + width_pavement / 2), -(length / 2 + distance_center)), width_pavement, length)  # south
    add_box((-(length / 2 + distance_center), -(width_traffic_island / 2 + width_road + width_pavement / 2)), length, width_pavement)  # west
    add_box((-(length / 2 + distance_center), (width_traffic_island / 2 + width_road + width_pavement / 2)), length, width_pavement)  # west
    add_box((-(width_traffic_island / 2 + width_road + width_pavement / 2), (length / 2 + distance_center)), width_pavement, length)  # north
    add_box(((width_traffic_island / 2 + width_road + width_pavement / 2), (length / 2 + distance_center)), width_pavement, length)  # north
    add_box(((length / 2 + distance_center), -(width_traffic_island / 2 + width_road + width_pavement / 2)), length, width_pavement)  # east
    add_box(((length / 2 + distance_center), (width_traffic_island / 2 + width_road + width_pavement / 2)), length, width_pavement)  # east

    # Set the aspect of the plot to be equal
    ax.set_aspect('equal', 'box')

    # Set the plot limits to focus on the intersection
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)

    # Show the plot
    plt.show()

# If you want to see how it looks like, uncomment the following and execute the py file
if __name__ == '__main__':
    scenario = free_area(start_pos=0, angle=0)
    
    for obstacle in scenario.obstacles:
        obstacle.draw(plt.gca(), color='b')
        
    free_area()