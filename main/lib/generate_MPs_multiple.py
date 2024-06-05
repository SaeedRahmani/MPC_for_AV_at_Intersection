import numpy as np
import matplotlib.pyplot as plt

# Define parameters here
num_mps = 5           # Number of motion primitives to generate
angle_range = 45      # Range of steering angles (from -x to +x degrees)
mp_length = 1       # Length of each motion primitive (total time) in seconds
steps = 1            # Number of steps to repeat generating the motion primitives

# Bicycle parameters
L = 1.0  # Length of the bicycle (adjust as needed)
v = 1.0  # Constant velocity (adjust as needed)
delta_t = 0.1  # Time step (adjust as needed)

# Generate steering angles
delta_angles = np.linspace(-angle_range, angle_range, num_mps)
delta_angles_rad = np.deg2rad(delta_angles)  # Convert to radians

# Initial state
x0, y0, theta0 = 0.0, 0.0, 0.0

def generate_trajectory(delta, x0, y0, theta0, L, v, delta_t, total_time):
    num_steps = int(total_time / delta_t)
    x, y, theta = x0, y0, theta0

    trajectory = [(x, y, theta)]
    
    for _ in range(num_steps):
        x += v * np.cos(theta) * delta_t
        y += v * np.sin(theta) * delta_t
        theta += (v / L) * np.tan(delta) * delta_t
        trajectory.append((x, y, theta))
    
    return np.array(trajectory)

# Function to recursively generate motion primitives
def recursive_generate_primitives(x0, y0, theta0, L, v, delta_t, total_time, steps, delta_angles_rad):
    all_trajectories = []
    initial_positions = [(x0, y0, theta0)]
    
    for step in range(steps):
        new_positions = []
        for x0, y0, theta0 in initial_positions:
            for delta in delta_angles_rad:
                trajectory = generate_trajectory(delta, x0, y0, theta0, L, v, delta_t, total_time)
                all_trajectories.append(trajectory)
                new_positions.append(trajectory[-1])  # Last position as starting point for next step
        initial_positions = new_positions
    
    return all_trajectories

# Generate motion primitives recursively
trajectories = recursive_generate_primitives(x0, y0, theta0, L, v, delta_t, mp_length, steps, delta_angles_rad)

# Plot the trajectories
plt.figure(figsize=(10, 8))
for trajectory in trajectories:
    plt.plot(trajectory[:, 0], trajectory[:, 1], linewidth=0.5)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Bicycle Model Trajectories Over Multiple Steps')
plt.grid(True)
plt.axis('equal')
plt.show()
