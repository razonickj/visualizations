import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from scipy.interpolate import splprep, splev

# --- Parameters ---
N_POINTS_SPLINE1 = 5  # Number of points (n) on spline 1
M_POINTS_SPLINE2 = 7  # Number of points (m) on spline 2
SPLINE_RESOLUTION = 100 # Number of points for drawing smooth spline curves
INTERVAL = 150        # Milliseconds between frames

# --- 1. Generate Spline Data ---

# Define control points for two non-intersecting splines
# Ensure they are visually separated
control_points1 = np.array([
    [0.5, 1.0], [1.0, 2.5], [2.0, 3.0], [3.5, 2.8], [4.5, 1.5], [5.0, 0.5]
])
control_points2 = np.array([
    [1.0, -0.5], [2.0, -1.5], [3.5, -1.8], [5.0, -1.0], [6.0, 0.0]
])

# Function to generate spline curve and specific points
def generate_spline(control_points, num_points, resolution):
    """Generates smooth spline curve and evaluates specific points."""
    if len(control_points) < 2:
        raise ValueError("Need at least 2 control points")
    # Ensure control points are floats for splprep
    control_points = np.array(control_points, dtype=float).T # Transpose for splprep (dims, num_pts)

    # Parameterize the spline (k=3 for cubic, s=0 forces interpolation through points)
    tck, u = splprep(control_points, s=0, k=min(3, len(control_points[0])-1))

    # Evaluate points for smooth curve drawing
    u_fine = np.linspace(u.min(), u.max(), resolution)
    x_fine, y_fine = splev(u_fine, tck)

    # Evaluate N specific points along the spline (evenly spaced in parameter u)
    u_points = np.linspace(u.min(), u.max(), num_points)
    x_points, y_points = splev(u_points, tck)

    return x_fine, y_fine, np.column_stack((x_points, y_points))

# Generate the splines and the N/M specific points on them
x1_curve, y1_curve, spline1_points = generate_spline(control_points1, N_POINTS_SPLINE1, SPLINE_RESOLUTION)
x2_curve, y2_curve, spline2_points = generate_spline(control_points2, M_POINTS_SPLINE2, SPLINE_RESOLUTION)


# --- 2. Generate Connection Matrix Data ---
# Create an (n x m) matrix with positive and negative values
# Example: based on difference in x-coords plus some noise
connection_matrix = np.zeros((N_POINTS_SPLINE1, M_POINTS_SPLINE2))
for i in range(N_POINTS_SPLINE1):
    for j in range(M_POINTS_SPLINE2):
        # Example: Difference in x-coordinates modulated by y-coordinates
        val = (spline1_points[i, 0] - spline2_points[j, 0]) * np.sin(spline1_points[i, 1])
        # Add some noise/other factor to make it more varied
        val += (np.random.rand() - 0.5) * 2.0
        connection_matrix[i, j] = val

# Determine symmetric color limits for the heatmap
max_abs_val = np.max(np.abs(connection_matrix))
vmin = -max_abs_val
vmax = max_abs_val

# --- 3. Animation Setup ---
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle("Spline Point Connections and Interaction Matrix")

# --- Left Plot: Splines and Connection Line ---
ax_spline = axes[0]
# Plot smooth spline curves
ax_spline.plot(x1_curve, y1_curve, 'b-', label=f'Spline 1 ({N_POINTS_SPLINE1} points)')
ax_spline.plot(x2_curve, y2_curve, 'g-', label=f'Spline 2 ({M_POINTS_SPLINE2} points)')
# Plot the specific N and M points
ax_spline.plot(spline1_points[:, 0], spline1_points[:, 1], 'bo', markersize=6)
ax_spline.plot(spline2_points[:, 0], spline2_points[:, 1], 'go', markersize=6)
# Initialize the connection line (initially empty)
connection_line, = ax_spline.plot([], [], 'r--', lw=1.5, label='Connection') # Red dashed line

ax_spline.set_title("Splines")
ax_spline.set_xlabel("X-coordinate")
ax_spline.set_ylabel("Y-coordinate")
ax_spline.legend(loc='best')
ax_spline.axis('equal') # Ensure aspect ratio is equal
ax_spline.grid(True, linestyle=':', alpha=0.6)

# --- Right Plot: Heatmap and Highlight Box ---
ax_heatmap = axes[1]
im_heatmap = ax_heatmap.imshow(connection_matrix, cmap='coolwarm', vmin=vmin, vmax=vmax,
                                aspect='auto', origin='upper', interpolation='nearest')
ax_heatmap.set_title("Connection Matrix (NxM)")
ax_heatmap.set_xlabel("Spline 2 Point Index (j)")
ax_heatmap.set_ylabel("Spline 1 Point Index (i)")

# Add ticks corresponding to indices
ax_heatmap.set_xticks(np.arange(M_POINTS_SPLINE2))
ax_heatmap.set_yticks(np.arange(N_POINTS_SPLINE1))
ax_heatmap.tick_params(axis='x', rotation=90) # Rotate x-ticks if many points

fig.colorbar(im_heatmap, ax=ax_heatmap, fraction=0.046, pad=0.04, label='Connection Value')

# Initialize the highlight rectangle patch
highlight_rect = patches.Rectangle(
    (-0.5, -0.5), 1, 1, # Width=1, Height=1 covers one cell
    linewidth=2.5, edgecolor='yellow', facecolor='none'
)
ax_heatmap.add_patch(highlight_rect)

# Add text to display current connection info
info_text = fig.text(0.5, 0.01, '', ha='center', va='bottom', fontsize=10)

plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout for title and text

# --- 4. Animation Function ---
total_frames = N_POINTS_SPLINE1 * M_POINTS_SPLINE2

def animate(frame):
    if frame >= total_frames:
        return connection_line, highlight_rect, info_text

    # Map frame number to (i, j) indices
    i = frame // M_POINTS_SPLINE2  # Index for spline 1 (rows)
    j = frame % M_POINTS_SPLINE2   # Index for spline 2 (columns)

    # Get the coordinates of the points to connect
    point1 = spline1_points[i]
    point2 = spline2_points[j]

    # Update the connection line data
    connection_line.set_data([point1[0], point2[0]], [point1[1], point2[1]])

    # Update the highlight rectangle position on the heatmap
    # For origin='upper', cell (i,j) is at y=i, x=j. Patch bottom-left needs to be (x-0.5, y-0.5)
    highlight_rect.set_xy((j - 0.5, i - 0.5))

    # Update the info text
    matrix_value = connection_matrix[i, j]
    info_text.set_text(f'Connecting Spline1 Point {i} to Spline2 Point {j}. Matrix Value: {matrix_value:.2f}')

    # Return updated artists for blitting
    return connection_line, highlight_rect, info_text

# --- 5. Create and Save Animation ---
print(f"Creating animation with {total_frames} frames...")

ani = animation.FuncAnimation(fig, animate, frames=total_frames,
                              interval=INTERVAL, blit=True, repeat=False)

# Save as GIF
gif_filename = 'spline_connections_heatmap.gif'
try:
    ani.save(gif_filename, writer='pillow', fps=int(1000 / INTERVAL))
    print(f"Successfully saved animation to {gif_filename}")
except ImportError:
    print("Pillow not found. Install with 'pip install Pillow'. Cannot save GIF.")
except Exception as e:
    print(f"Error saving animation: {e}")

plt.close(fig) # Close the plot window