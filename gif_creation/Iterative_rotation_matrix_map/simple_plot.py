import numpy as np
import matplotlib.pyplot as plt
import math

# --- Configuration ---
# Use the same rotation angle as the previous script for consistency
ANGLE_DEG_PER_STEP = 1.5
NUM_POINTS_ON_CIRCLE = 200 # Number of sample input angles to test

# --- Setup ---
# Fixed rotation angle (alpha) in radians
alpha_rad = np.radians(ANGLE_DEG_PER_STEP)

# Create the 2D rotation matrix for angle alpha
rotation_matrix = np.array([
    [np.cos(alpha_rad), -np.sin(alpha_rad)],
    [np.sin(alpha_rad),  np.cos(alpha_rad)]
])

# --- Generate Data ---
# Create a range of input angles (theta_n) from 0 to 2*pi
# Using linspace ensures we cover the full circle
theta_n_rad = np.linspace(0, 2 * np.pi, NUM_POINTS_ON_CIRCLE, endpoint=False) # endpoint=False avoids duplicating 0 and 2pi

theta_n_plus_1_rad = np.zeros_like(theta_n_rad) # Array to store output angles

print(f"Calculating input vs output angles for rotation by {ANGLE_DEG_PER_STEP} degrees...")
for i, tn in enumerate(theta_n_rad):
    # 1. Create a unit vector at angle theta_n
    vn = np.array([np.cos(tn), np.sin(tn)])

    # 2. Apply the rotation matrix
    vn1 = rotation_matrix @ vn

    # 3. Calculate the angle of the resulting vector
    #    atan2(y, x) gives the angle in radians from (-pi, pi]
    tn1 = np.arctan2(vn1[1], vn1[0])
    theta_n_plus_1_rad[i] = tn1

# --- Handle Angle Wrapping ---
# arctan2 wraps angles to (-pi, pi]. To see the continuous relationship theta_n + alpha,
# we need to unwrap the calculated theta_n+1 values.
# np.unwrap detects jumps > pi (by default) and adds multiples of 2*pi to make it continuous.
theta_n_plus_1_unwrapped_rad = np.unwrap(theta_n_plus_1_rad)

# --- Convert angles to degrees for plotting ---
theta_n_deg = np.degrees(theta_n_rad)
theta_n_plus_1_unwrapped_deg = np.degrees(theta_n_plus_1_unwrapped_rad)
alpha_deg = ANGLE_DEG_PER_STEP # Keep alpha in degrees for theoretical line

# --- Plotting ---
fig, ax = plt.subplots(figsize=(8, 8))

# Plot the calculated relationship
ax.plot(theta_n_deg, theta_n_plus_1_unwrapped_deg, 'o-', markersize=4, linewidth=1.5, color='darkcyan', label='Calculated (Matrix Rotation)')

# Plot the theoretical relationship: theta_{n+1} = theta_n + alpha
theoretical_theta_n_plus_1 = theta_n_deg + alpha_deg
ax.plot(theta_n_deg, theoretical_theta_n_plus_1, '--', linewidth=2, color='red', label=f'Theoretical (y = x + {alpha_deg:.1f}°)')

# Configure the plot
ax.set_title(f'Input Angle (θ_n) vs. Output Angle (θ_n+1)\nAfter Rotation by {alpha_deg:.1f}°')
ax.set_xlabel('Input Angle θ_n (degrees)')
ax.set_ylabel('Output Angle θ_n+1 (degrees)')
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend()
ax.set_aspect('equal', adjustable='box') # Keep aspect ratio 1:1

# Set appropriate limits, showing the 0-360 range and the offset
ax.set_xlim(0, 360)
ax.set_ylim(0, 360 + alpha_deg * 1.2) # Adjust ylim to show the full output range

plt.show()

print("Script finished.")