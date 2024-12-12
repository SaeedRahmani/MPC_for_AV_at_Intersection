import matplotlib.pyplot as plt

def plot_intersection(scenario):
    # Initialize the figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    # Initialize min and max coordinates
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    # Plot each obstacle using its draw method and update min/max coordinates
    for obstacle in scenario.obstacles:
        obstacle.draw(ax, color='gray')
        x, y = obstacle.xy_center
        width, height = obstacle.xy_width if hasattr(obstacle, 'xy_width') else (obstacle.radius * 2, obstacle.radius * 2)
        min_x = min(min_x, x - width / 2)
        max_x = max(max_x, x + width / 2)
        min_y = min(min_y, y - height / 2)
        max_y = max(max_y, y + height / 2)

    # Plot the goal area and update min/max coordinates
    scenario.goal_area.draw(ax, color='green')
    x, y = scenario.goal_area.xy_center
    width, height = scenario.goal_area.xy_width if hasattr(scenario.goal_area, 'xy_width') else (scenario.goal_area.radius * 2, scenario.goal_area.radius * 2)
    min_x = min(min_x, x - width / 2)
    max_x = max(max_x, x + width / 2)
    min_y = min(min_y, y - height / 2)
    max_y = max(max_y, y + height / 2)

    # Set the aspect of the plot to be equal
    ax.set_aspect('equal', 'box')

    # Set the plot limits based on the obstacles and goal area
    ax.set_xlim(min_x - 10, max_x + 10)
    ax.set_ylim(min_y - 10, max_y + 10)

    # Show the plot
    plt.show()