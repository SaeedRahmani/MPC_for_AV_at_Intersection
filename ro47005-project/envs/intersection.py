import sys

import numpy as np

from lib.obstacles import BoxObstacle, CircleObstacle
from lib.scenario import Scenario

def intersection(turn_indicator: int, start_pos: int) -> Scenario:
    width_road = 4
    width_traffic_island = 2
    width_pavement = 5
    length = 30
    height = 0.5
    corner_radius = 6
    distance_center = corner_radius + width_road + width_traffic_island
    allowed_goal_theta_difference = np.pi / 16

    # 1: south, 2: west, 3: north, 4, east
    start_positions = {
        1: (width_traffic_island / 2 + width_road / 2, -30, 0.5 * np.pi),
        2: (-30, -(width_traffic_island / 2 + width_road / 2), 0),
        3: (-(width_traffic_island / 2 + width_road / 2), 30, -0.5 * np.pi),
        4: (30, width_traffic_island / 2 + width_road / 2, np.pi),
    }
    
    # 1: turn left, 2: go straight, 3: turn right
    goal_distance = 30
    goal_positions = {
        1: {
            1: (-goal_distance, (width_traffic_island + width_road) / 2, -np.pi),
            2: ((width_traffic_island + width_road) / 2, goal_distance, 0.5 * np.pi),
            3: (goal_distance, -(width_traffic_island + width_road) / 2, 0)
        },
        2: {
            1: ((width_traffic_island + width_road) / 2, goal_distance, 0.5 * np.pi),
            2: (goal_distance, -(width_traffic_island + width_road) / 2, 0),
            3: (-(width_traffic_island + width_road) / 2, -goal_distance, -0.5 * np.pi)
        },
        3: {
            1: (goal_distance, -(width_traffic_island + width_road) / 2, 0),
            2: (-(width_traffic_island + width_road) / 2, -goal_distance, -0.5 * np.pi),
            3: (-goal_distance, (width_traffic_island + width_road) / 2, -np.pi)
        },
        4: {
            1: (-(width_traffic_island + width_road) / 2, -goal_distance, -0.5 * np.pi),
            2: (-goal_distance, (width_traffic_island + width_road) / 2, -np.pi),
            3: ((width_traffic_island + width_road) / 2, goal_distance, 0.5 * np.pi)
        }
    }

    start = start_positions[start_pos]
    goal = goal_positions[start_pos][turn_indicator]
    
    # Because the goal area is a rectangle and depends on the end position
    if ((start_pos == 1 or start_pos == 3) and (turn_indicator == 1 or turn_indicator == 3)) or ((start_pos == 2 or start_pos == 4) and (turn_indicator == 2 or turn_indicator == 4)):
        goal_area = BoxObstacle(xy_width=(width_road * 1.8, width_road), height=height, xy_center=(goal[0], goal[1]))
    else:
        goal_area = BoxObstacle(xy_width=(width_road, width_road * 1.8), height=height, xy_center=(goal[0], goal[1]))
    
    # Obstacles for defining the intersection
    obstacles = [
        # Full intersection
        
        # Medians (Islands)
        # south median
        BoxObstacle(xy_width=(width_traffic_island, length), height=height,
                        xy_center=(0, -(length / 2 + distance_center))),
        CircleObstacle(radius=(width_traffic_island / 2), height=height,
                           xy_center=(0, -distance_center)),
        # north median
        BoxObstacle(xy_width=(width_traffic_island, length), height=height,
                        xy_center=(0, (length / 2 + distance_center))),
        CircleObstacle(radius=(width_traffic_island / 2), height=height,
                           xy_center=(0, distance_center)),
        # west median
        BoxObstacle(xy_width=(length, width_traffic_island), height=height,
                        xy_center=(-(length / 2 + distance_center), 0)),
        CircleObstacle(radius=(width_traffic_island / 2), height=height,
                           xy_center=(-distance_center, 0)),
        # east median
        BoxObstacle(xy_width=(length, width_traffic_island), height=height,
                        xy_center=((length / 2 + distance_center), 0)),
        CircleObstacle(radius=(width_traffic_island / 2), height=height,
                           xy_center=(distance_center, 0)),
        
        # Corners
        # south west corner
        CircleObstacle(radius=(distance_center - width_traffic_island / 2 - width_road), height=height,
                           xy_center=(-distance_center, -distance_center)),
        # north west corner
        CircleObstacle(radius=(distance_center - width_traffic_island / 2 - width_road), height=height,
                           xy_center=(-distance_center, distance_center)),
        # north east corner
        CircleObstacle(radius=(distance_center - width_traffic_island / 2 - width_road), height=height,
                           xy_center=(distance_center, distance_center)),
        # north west corner
        CircleObstacle(radius=(distance_center - width_traffic_island / 2 - width_road), height=height,
                           xy_center=(distance_center, -distance_center)),
        
        # Pavements
        # south
        BoxObstacle(xy_width=(width_pavement, length), height=height,
                        xy_center=(
                            -(width_traffic_island / 2 + width_road + width_pavement / 2),
                            -(length / 2 + distance_center))),
        BoxObstacle(xy_width=(width_pavement, length), height=height,
                        xy_center=(
                            (width_traffic_island / 2 + width_road + width_pavement / 2),
                            -(length / 2 + distance_center))),
        # west
        BoxObstacle(xy_width=(length, width_pavement), height=height,
                        xy_center=(
                            -(length / 2 + distance_center),
                            -(width_traffic_island / 2 + width_road + width_pavement / 2)
                            )),
        BoxObstacle(xy_width=(length, width_pavement), height=height,
                        xy_center=(
                            -(length / 2 + distance_center),
                            (width_traffic_island / 2 + width_road + width_pavement / 2)
                            )),
        # north
        BoxObstacle(xy_width=(width_pavement, length), height=height,
                        xy_center=(
                            -(width_traffic_island / 2 + width_road + width_pavement / 2),
                            (length / 2 + distance_center))),
        BoxObstacle(xy_width=(width_pavement, length), height=height,
                        xy_center=(
                            (width_traffic_island / 2 + width_road + width_pavement / 2),
                            (length / 2 + distance_center))),
        # east
        BoxObstacle(xy_width=(length, width_pavement), height=height,
                        xy_center=(
                            (length / 2 + distance_center),
                            -(width_traffic_island / 2 + width_road + width_pavement / 2)
                            )),
        BoxObstacle(xy_width=(length, width_pavement), height=height,
                        xy_center=(
                            (length / 2 + distance_center),
                            (width_traffic_island / 2 + width_road + width_pavement / 2)
                            ))
    ]
    
    # We should also add some hidden obstacles in for prohibitting the traffic rules. But because this is a general scenario,
    # we should make it conditional to the vehicle's start and goal positions. Therefore, I will add it to Motion Primitives.
        
    if start_pos == 1:
        obstacles.extend([
            BoxObstacle(xy_width=(length, width_road), height=height,
                        xy_center=(-(length / 2 + distance_center), - (width_road + width_traffic_island) / 2),
                        hidden=True),
            BoxObstacle(xy_width=(length, width_road), height=height,
                        xy_center=((length / 2 + distance_center), (width_road + width_traffic_island) / 2),
                        hidden=True),
            BoxObstacle(xy_width=(width_road, length), height=height,
                        xy_center=(-(width_road + width_traffic_island) / 2, -(length / 2 + distance_center)),
                        hidden=True),
            BoxObstacle(xy_width=(width_road, length), height=height,
                        xy_center=(-(width_road + width_traffic_island) / 2, (length / 2 + distance_center)),
                        hidden=True)
            ])
    elif start_pos == 2:
        obstacles.extend([
            BoxObstacle(xy_width=(length, width_road), height=height,
                        xy_center=(-(length / 2 + distance_center), (width_road + width_traffic_island) / 2),
                        hidden=True),
            BoxObstacle(xy_width=(length, width_road), height=height,
                        xy_center=((length / 2 + distance_center), (width_road + width_traffic_island) / 2),
                        hidden=True),
            BoxObstacle(xy_width=(width_road, length), height=height,
                        xy_center=((width_road + width_traffic_island) / 2, -(length / 2 + distance_center)),
                        hidden=True),
            BoxObstacle(xy_width=(width_road, length), height=height,
                        xy_center=(-(width_road + width_traffic_island) / 2, (length / 2 + distance_center)),
                        hidden=True)
            ])
    elif start_pos == 3:
        obstacles.extend([
            BoxObstacle(xy_width=(length, width_road), height=height,
                        xy_center=(-(length / 2 + distance_center), - (width_road + width_traffic_island) / 2),
                        hidden=True),
            BoxObstacle(xy_width=(length, width_road), height=height,
                        xy_center=((length / 2 + distance_center), (width_road + width_traffic_island) / 2),
                        hidden=True),
            BoxObstacle(xy_width=(width_road, length), height=height,
                        xy_center=((width_road + width_traffic_island) / 2, -(length / 2 + distance_center)),
                        hidden=True),
            BoxObstacle(xy_width=(width_road, length), height=height,
                        xy_center=((width_road + width_traffic_island) / 2, (length / 2 + distance_center)),
                        hidden=True)
            ])
    else:
        obstacles.extend([
            BoxObstacle(xy_width=(length, width_road), height=height,
                        xy_center=(-(length / 2 + distance_center), - (width_road + width_traffic_island) / 2),
                        hidden=True),
            BoxObstacle(xy_width=(length, width_road), height=height,
                        xy_center=((length / 2 + distance_center), -(width_road + width_traffic_island) / 2),
                        hidden=True),
            BoxObstacle(xy_width=(width_road, length), height=height,
                        xy_center=((width_road + width_traffic_island) / 2, -(length / 2 + distance_center)),
                        hidden=True),
            BoxObstacle(xy_width=(width_road, length), height=height,
                        xy_center=(-(width_road + width_traffic_island) / 2, (length / 2 + distance_center)),
                        hidden=True)
            ])

    return Scenario(
        start=start,
        goal_point=goal,
        goal_area=goal_area,
        allowed_goal_theta_difference=allowed_goal_theta_difference,
        obstacles=obstacles,
    )
    
def plot_intersection():
    # Initialize the figure and axis
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle

    # Parameters from your script
    width_road = 4
    width_traffic_island = 2
    width_pavement = 5
    length = 30
    corner_radius = 6
    distance_center = corner_radius + width_road + width_traffic_island

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
    plot_intersection()