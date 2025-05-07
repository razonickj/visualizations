import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp # Use a better solver

# --- Simulation Parameters ---
g = 9.81  # Acceleration due to gravity (m/s^2)

# Pendulum 1 properties
L1 = 1.0  # Length of pendulum 1 (m)
theta1_0 = np.radians(60)  # Initial angle of pendulum 1 (radians)
omega1_0 = 0.0  # Initial angular velocity of pendulum 1 (rad/s)
pivot1 = (-0.7, 0.0) # Pivot point for pendulum 1 (x, y)

# Pendulum 2 properties
L2 = 0.7  # Length of pendulum 2 (m)
theta2_0 = np.radians(-45) # Initial angle of pendulum 2 (radians)
omega2_0 = 0.0  # Initial angular velocity of pendulum 2 (rad/s)
pivot2 = (0.7, 0.0)  # Pivot point for pendulum 2 (x, y)

# Time settings
t_max = 10.0  # Maximum simulation time (s)
dt = 0.02   # Time step for output (solver might use smaller internal steps)
times = np.arange(0, t_max, dt)

# --- Pendulum Differential Equation Function (remains the same) ---
# d(theta)/dt = omega
# d(omega)/dt = -(g/L) * sin(theta)
def pendulum_derivs(t, state, L): # Note: solve_ivp expects t as the first argument
    theta, omega = state
    dtheta_dt = omega
    domega_dt = -(g / L) * np.sin(theta)
    return [dtheta_dt, domega_dt]

# --- Solve Differential Equations using solve_ivp ---
# Pendulum 1
sol1 = solve_ivp(
    pendulum_derivs,
    [0, t_max],             # Time span
    [theta1_0, omega1_0],   # Initial state [theta, omega]
    args=(L1,),             # Arguments to pass to pendulum_derivs (L)
    dense_output=True,      # Allows evaluating solution at arbitrary times
    t_eval=times,           # Times at which to store the computed solution
    method='RK45'           # Integration method (robust)
)
theta1 = sol1.y[0]
omega1 = sol1.y[1]

# Pendulum 2
sol2 = solve_ivp(
    pendulum_derivs,
    [0, t_max],
    [theta2_0, omega2_0],
    args=(L2,),
    dense_output=True,
    t_eval=times,
    method='RK45'
)
theta2 = sol2.y[0]
omega2 = sol2.y[1]

# Check if solvers were successful
if not sol1.success:
    print(f"Solver for Pendulum 1 failed: {sol1.message}")
if not sol2.success:
    print(f"Solver for Pendulum 2 failed: {sol2.message}")

# Convert angles to Cartesian coordinates relative to their pivots
x1 = pivot1[0] + L1 * np.sin(theta1)
y1 = pivot1[1] - L1 * np.cos(theta1) # y is negative downwards from pivot
x2 = pivot2[0] + L2 * np.sin(theta2)
y2 = pivot2[1] - L2 * np.cos(theta2)

# --- Animation Setup ---
fig, ax = plt.subplots(figsize=(8, 6)) # Adjusted figure size slightly

# Determine plot limits dynamically
max_reach_x = max(abs(pivot1[0]) + L1, abs(pivot2[0]) + L2)
min_y = min(pivot1[1] - L1, pivot2[1] - L2)
max_y = max(pivot1[1], pivot2[1])
ax.set_xlim(-(max_reach_x + 0.2), max_reach_x + 0.2)
ax.set_ylim(min_y - 0.2, max_y + 0.2)
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_title('Two Independent Pendulums Simulation (Separate Pivots)')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')

# Plot pivot points
ax.plot(pivot1[0], pivot1[1], 'ks', markersize=6, label='Pivot 1') # Black square
ax.plot(pivot2[0], pivot2[1], 'kD', markersize=6, label='Pivot 2') # Black diamond

# Pendulum 1 elements
line1, = ax.plot([], [], 'o-', lw=1.5, markersize=7, color='blue', label=f'Pendulum 1 (L={L1}m)')
arrow1 = ax.arrow(0, 0, 0, 0, head_width=0.08, head_length=0.15, fc='red', ec='red', length_includes_head=True)

# Pendulum 2 elements
line2, = ax.plot([], [], 'o-', lw=1.5, markersize=7, color='green', label=f'Pendulum 2 (L={L2}m)')
arrow2 = ax.arrow(0, 0, 0, 0, head_width=0.08, head_length=0.15, fc='red', ec='red', length_includes_head=True)

# Time display
time_template = 'Time = %.1fs'
time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)

# Gravity arrow properties
arrow_length = 0.3 * g / 9.81 # Scale arrow length slightly with g, normalized

# --- Animation Function ---
def animate(i):
    # Ensure index is within bounds (can happen with some save methods/lengths)
    if i >= len(times):
        return line1, line2, arrow1, arrow2, time_text

    # Update pendulum 1
    # Line goes from pivot to current bob position
    line1.set_data([pivot1[0], x1[i]], [pivot1[1], y1[i]])

    # Update gravity arrow 1 (starts at bob position, points down)
    arrow1.set_data(x=x1[i], y=y1[i], dx=0, dy=-arrow_length)

    # Update pendulum 2
    # Line goes from pivot to current bob position
    line2.set_data([pivot2[0], x2[i]], [pivot2[1], y2[i]])

    # Update gravity arrow 2 (starts at bob position, points down)
    arrow2.set_data(x=x2[i], y=y2[i], dx=0, dy=-arrow_length)

    # Update time text
    time_text.set_text(time_template % times[i])

    return line1, line2, arrow1, arrow2, time_text

# --- Create and Save Animation ---
# frame_interval = dt * 1000 # milliseconds per frame, matches simulation time
frame_interval = 30 # milliseconds per frame, fixed animation speed (adjust as needed)

ani = animation.FuncAnimation(fig, animate, frames=len(times),
                              interval=frame_interval, blit=True, repeat=False)

# Place legend neatly
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=4)
plt.tight_layout(rect=[0, 0.08, 1, 1]) # Adjust layout to make space for legend


# Save as GIF
gif_filename = 'two_pendulums_separate_pivots.gif'
print(f"Saving animation to {gif_filename}...")
print("This might take a moment...")

# Use pillow writer for better GIF quality if available
try:
    ani.save(gif_filename, writer='pillow', fps=int(1000 / frame_interval))
    print(f"Successfully saved {gif_filename} using Pillow.")
except ImportError:
    print("Pillow writer not found. Install with: pip install Pillow")
    print("Attempting fallback to ImageMagick (if installed and configured)...")
    try:
        # You might need to install ImageMagick: https://imagemagick.org/
        # And potentially configure matplotlib:
        # plt.rcParams['animation.convert_path'] = r'C:\path\to\magick.exe' # Example
        ani.save(gif_filename, writer='imagemagick', fps=int(1000 / frame_interval))
        print(f"Successfully saved {gif_filename} using ImageMagick.")
        print("Note: Ensure ImageMagick is installed and configured in matplotlib's rcParams if you encounter issues.")
    except Exception as e:
        print(f"Failed to save GIF with ImageMagick: {e}")
        print("Animation could not be saved. Consider installing Pillow or configuring ImageMagick.")
        # Optionally show plot if saving fails and Pillow is not installed
        # plt.show()
except Exception as e:
    print(f"An unexpected error occurred during saving: {e}")
    # Optionally show plot
    # plt.show()


# Close the plot display if it was opened implicitly
plt.close(fig)