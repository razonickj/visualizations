import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

# --- Simulation Parameters ---
g = 9.81  # Acceleration due to gravity (m/s^2)

# Pendulum 1 properties
L1 = 1.0  # Length of pendulum 1 (m)
m1 = 1.0  # Mass of pendulum 1 bob (kg)
theta1_0 = np.radians(60)  # Initial angle of pendulum 1 (radians)
omega1_0 = 0.0  # Initial angular velocity of pendulum 1 (rad/s)
pivot1 = np.array([-0.7, 0.0]) # Pivot point for pendulum 1 (x, y)

# Pendulum 2 properties
L2 = 1.0  # Length of pendulum 2 (m) - Let's make them same length for potential resonance
m2 = 1.0  # Mass of pendulum 2 bob (kg)
theta2_0 = np.radians(-10) # Initial angle of pendulum 2 (radians) - Smaller initial angle
omega2_0 = 0.0  # Initial angular velocity of pendulum 2 (rad/s)
pivot2 = np.array([0.7, 0.0])  # Pivot point for pendulum 2 (x, y)

# Spring properties
k = 5.0   # Spring constant (N/m) - Adjust for different coupling strengths
# Calculate initial distance to set a potentially reasonable L0
x1_0 = pivot1[0] + L1 * np.sin(theta1_0)
y1_0 = pivot1[1] - L1 * np.cos(theta1_0)
x2_0 = pivot2[0] + L2 * np.sin(theta2_0)
y2_0 = pivot2[1] - L2 * np.cos(theta2_0)
initial_dist = np.sqrt((x2_0 - x1_0)**2 + (y2_0 - y1_0)**2)
L0 = initial_dist * 0.9 # Rest length of the spring (m) - slightly less than initial dist

print(f"Initial distance between bobs: {initial_dist:.2f} m")
print(f"Spring rest length (L0): {L0:.2f} m")

# Time settings
t_max = 5.0  # Maximum simulation time (s) - Longer to see coupling effects
dt = 0.03   # Time step for output
times = np.arange(0, t_max, dt)

# --- Coupled Pendulum Differential Equations ---
def coupled_pendulum_derivs(t, state, L1, m1, L2, m2, k, L0, g, pivot1, pivot2):
    theta1, omega1, theta2, omega2 = state

    # Calculate current bob positions
    x1 = pivot1[0] + L1 * np.sin(theta1)
    y1 = pivot1[1] - L1 * np.cos(theta1)
    x2 = pivot2[0] + L2 * np.sin(theta2)
    y2 = pivot2[1] - L2 * np.cos(theta2)

    # Vector from bob 1 to bob 2 and distance
    delta_x = x2 - x1
    delta_y = y2 - y1
    distance = np.sqrt(delta_x**2 + delta_y**2)

    # Spring force magnitude (Hooke's Law)
    # Positive if stretched, negative if compressed
    spring_force_mag = k * (distance - L0)

    # Avoid division by zero if distance is very small (unlikely here)
    if distance < 1e-9:
        unit_vec_x, unit_vec_y = 0.0, 0.0
    else:
        unit_vec_x = delta_x / distance
        unit_vec_y = delta_y / distance

    # Spring force vector components (Force ON bob 1 FROM spring/bob 2)
    Fs_x = spring_force_mag * unit_vec_x
    Fs_y = spring_force_mag * unit_vec_y

    # --- Calculate Torques ---
    # Torque = r x F = (rx * Fy - ry * Fx)
    # Lever arm vectors from pivot to bob
    r1_x = x1 - pivot1[0] # = L1*sin(theta1)
    r1_y = y1 - pivot1[1] # = -L1*cos(theta1)
    r2_x = x2 - pivot2[0] # = L2*sin(theta2)
    r2_y = y2 - pivot2[1] # = -L2*cos(theta2)

    # Torque on Pendulum 1
    # Gravity torque
    tau_g1 = -m1 * g * L1 * np.sin(theta1)
    # Spring torque (Force Fs acts ON bob 1)
    tau_s1 = r1_x * Fs_y - r1_y * Fs_x
    # Total torque on Pendulum 1
    tau1 = tau_g1 + tau_s1

    # Torque on Pendulum 2
    # Gravity torque
    tau_g2 = -m2 * g * L2 * np.sin(theta2)
    # Spring torque (Force -Fs acts ON bob 2)
    tau_s2 = r2_x * (-Fs_y) - r2_y * (-Fs_x)
    # Total torque on Pendulum 2
    tau2 = tau_g2 + tau_s2

    # --- Calculate Angular Accelerations ---
    # alpha = tau / I, where I = m * L^2 for a simple pendulum bob
    domega1_dt = tau1 / (m1 * L1**2)
    domega2_dt = tau2 / (m2 * L2**2)

    # Derivatives vector
    dtheta1_dt = omega1
    dtheta2_dt = omega2

    return [dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt]

# --- Solve Coupled Differential Equations ---
initial_state = [theta1_0, omega1_0, theta2_0, omega2_0]
solver_args = (L1, m1, L2, m2, k, L0, g, pivot1, pivot2)

sol = solve_ivp(
    coupled_pendulum_derivs,
    [0, t_max],
    initial_state,
    args=solver_args,
    dense_output=True,
    t_eval=times,
    method='RK45' # Or 'Radau' or 'BDF' for potentially stiffer systems
)

if not sol.success:
    print(f"ODE solver failed: {sol.message}")

# Extract results
theta1 = sol.y[0]
omega1 = sol.y[1]
theta2 = sol.y[2]
omega2 = sol.y[3]

# Convert angles to Cartesian coordinates relative to their pivots
x1 = pivot1[0] + L1 * np.sin(theta1)
y1 = pivot1[1] - L1 * np.cos(theta1)
x2 = pivot2[0] + L2 * np.sin(theta2)
y2 = pivot2[1] - L2 * np.cos(theta2)

# --- Animation Setup ---
fig, ax = plt.subplots(figsize=(9, 7)) # Adjusted figure size

# Determine plot limits dynamically
max_reach_x = max(abs(pivot1[0]) + L1, abs(pivot2[0]) + L2)
min_y = min(pivot1[1] - L1, pivot2[1] - L2)
max_y = max(pivot1[1], pivot2[1])
ax.set_xlim(-(max_reach_x + 0.3), max_reach_x + 0.3)
ax.set_ylim(min_y - 0.3, max_y + 0.3)
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_title(f'Coupled Pendulums Simulation (k={k} N/m, L0={L0:.2f} m)')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')

# Plot pivot points
ax.plot(pivot1[0], pivot1[1], 'ks', markersize=6, label='Pivot 1')
ax.plot(pivot2[0], pivot2[1], 'kD', markersize=6, label='Pivot 2')

# Pendulum 1 elements
line1, = ax.plot([], [], 'o-', lw=2, markersize=8, color='blue', label=f'P1 (L={L1}m)')
# Pendulum 2 elements
line2, = ax.plot([], [], 'o-', lw=2, markersize=8, color='green', label=f'P2 (L={L2}m)')
# Spring element
spring_line, = ax.plot([], [], '--', lw=1.5, color='grey', label=f'Spring (k={k})') # Dashed line for spring

# Gravity arrows (optional, can make plot busy)
# arrow1 = ax.arrow(0, 0, 0, 0, head_width=0.08, head_length=0.15, fc='red', ec='red', length_includes_head=True)
# arrow2 = ax.arrow(0, 0, 0, 0, head_width=0.08, head_length=0.15, fc='red', ec='red', length_includes_head=True)
# arrow_length = 0.3 * g / 9.81

# Time display
time_template = 'Time = %.1fs'
time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)

# --- Animation Function ---
def animate(i):
    # Ensure index is within bounds
    if i >= len(times):
        return line1, line2, spring_line, time_text # Add other elements if used

    # Update pendulum 1 line (pivot to bob)
    line1.set_data([pivot1[0], x1[i]], [pivot1[1], y1[i]])

    # Update pendulum 2 line (pivot to bob)
    line2.set_data([pivot2[0], x2[i]], [pivot2[1], y2[i]])

    # Update spring line (bob 1 to bob 2)
    spring_line.set_data([x1[i], x2[i]], [y1[i], y2[i]])

    # Update gravity arrows (if enabled)
    # arrow1.set_data(x=x1[i], y=y1[i], dx=0, dy=-arrow_length)
    # arrow2.set_data(x=x2[i], y=y2[i], dx=0, dy=-arrow_length)

    # Update time text
    time_text.set_text(time_template % times[i])

    # Return all animated elements
    return line1, line2, spring_line, time_text # Add arrows if used

# --- Create and Save Animation ---
frame_interval = 40 # milliseconds per frame (adjust for desired speed)

ani = animation.FuncAnimation(fig, animate, frames=len(times),
                              interval=frame_interval, blit=True, repeat=False)

# Place legend neatly
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3)
plt.tight_layout(rect=[0, 0.08, 1, 1]) # Adjust layout

# Save as GIF
gif_filename = 'coupled_pendulums_simulation.gif'
print(f"Saving animation to {gif_filename}...")
print("This might take a moment, especially for longer simulations...")

try:
    ani.save(gif_filename, writer='pillow', fps=int(1000 / frame_interval))
    print(f"Successfully saved {gif_filename} using Pillow.")
except ImportError:
    print("Pillow writer not found. Install with: pip install Pillow")
    print("Attempting fallback to ImageMagick (if installed and configured)...")
    try:
        ani.save(gif_filename, writer='imagemagick', fps=int(1000 / frame_interval))
        print(f"Successfully saved {gif_filename} using ImageMagick.")
    except Exception as e:
        print(f"Failed to save GIF with ImageMagick: {e}")
except Exception as e:
    print(f"An unexpected error occurred during saving: {e}")

plt.close(fig) # Close the plot window