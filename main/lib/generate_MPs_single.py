import numpy as np
import matplotlib.pyplot as plt

# Bicycle parameters
L = 1.0  # Length of the bicycle (adjust as needed)
v = 1.0  # Constant velocity (adjust as needed)
delta_angles = np.linspace(-30, 30, 5)  # Steering angles in degrees
delta_angles_rad = np.deg2rad(delta_angles)  # Convert to radians
delta_t = 0.1  # Time step (adjust as needed)
total_time = 1.0  # Total time for simulation

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

# Generate trajectories for each steering angle
trajectories = {}
for delta in delta_angles_rad:
    trajectory = generate_trajectory(delta, x0, y0, theta0, L, v, delta_t, total_time)
    trajectories[np.rad2deg(delta)] = trajectory

# Plot the trajectories
plt.figure(figsize=(10, 8))
for delta, trajectory in trajectories.items():
    plt.plot(trajectory[:, 0], trajectory[:, 1], label=f'Steering Angle {delta:.1f}°')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Bicycle Model Trajectories')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
