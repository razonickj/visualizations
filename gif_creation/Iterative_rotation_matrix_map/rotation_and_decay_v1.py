import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

# --- Configuration ---
INITIAL_VECTOR = np.array([2.5, 1.0]) # Starting vector
ANGLE_DEG_PER_STEP = 6.0             # Rotation angle in degrees for each step
SCALE_FACTOR = 0.97                  # Decay factor per step (must be < 1)
NUM_FRAMES = 150                     # Total number of iterations/frames
INTERVAL = 50                        # Milliseconds between frames (animation speed)
FILENAME = "spiraling_decay.gif"     # Output filename

# --- Setup ---
# Convert angle to radians
angle_rad_per_step = np.radians(ANGLE_DEG_PER_STEP)

# Create the 2D rotation matrix
rotation_matrix = np.array([
    [np.cos(angle_rad_per_step), -np.sin(angle_rad_per_step)],
    [np.sin(angle_rad_per_step),  np.cos(angle_rad_per_step)]
])

# Create the scaling matrix (decay towards origin)
scaling_matrix = np.array([
    [SCALE_FACTOR, 0],
    [0, SCALE_FACTOR]
])

# Combine the matrices: Rotate first, then scale
# v_rotated = R @ v_initial
# v_scaled_and_rotated = S @ v_rotated = (S @ R) @ v_initial
transform_matrix = scaling_matrix @ rotation_matrix

# --- Pre-calculate Vector States ---
vector_states = [INITIAL_VECTOR]
current_vector = INITIAL_VECTOR
print(f"Calculating {NUM_FRAMES} vector states...")
for _ in range(NUM_FRAMES - 1):
    # Apply the combined transformation matrix
    current_vector = transform_matrix @ current_vector
    vector_states.append(current_vector)
# Convert the list of vectors into a 2D NumPy array for easier slicing
vector_states = np.array(vector_states)
print("Calculation complete.")

# --- Animation Setup ---
fig, ax = plt.subplots(figsize=(7, 7))

# Determine axis limits based on the initial vector's maximum coordinate
max_coord = np.max(np.abs(INITIAL_VECTOR)) * 1.2 # Add padding

ax.set_xlim(-max_coord, max_coord)
ax.set_ylim(-max_coord, max_coord)
ax.set_aspect('equal', adjustable='box') # Maintain aspect ratio
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.plot(0, 0, 'ko', ms=5) # Mark origin

# --- Artists for Animation Update ---
# Use quiver for the vector arrow
quiver_artist = ax.quiver(0, 0, INITIAL_VECTOR[0], INITIAL_VECTOR[1],
                          angles='xy', scale_units='xy', scale=1,
                          color='red', width=0.008)

# Use plot for the trajectory/path (initialize with empty data)
path_line, = ax.plot([], [], 'b-', lw=1, alpha=0.7) # Blue line for path

# Add text for iteration number
iter_text = ax.text(0.02, 0.95, '', transform=ax.transAxes) # Position relative to axes

# --- Animation Function ---
def update(frame):
    """Updates the plot for each frame of the animation."""
    global quiver_artist # Needed as we re-assign quiver object

    # Get current vector from pre-calculated states
    vector = vector_states[frame]

    # --- Update Vector Arrow ---
    # More efficient quiver update: set U, V components
    # quiver_artist.set_UVC(vector[0], vector[1]) # Preferred method
    # Or remove and redraw (simpler to implement initially)
    quiver_artist.remove()
    quiver_artist = ax.quiver(0, 0, vector[0], vector[1], angles='xy',
                              scale_units='xy', scale=1, color='red', width=0.008)


    # --- Update Path Trace ---
    # Get all points from the start up to the current frame
    path_data = vector_states[0:frame+1]
    # Update the line data efficiently
    path_line.set_data(path_data[:, 0], path_data[:, 1])

    # Update the title or text with iteration number
    # ax.set_title(f'Rotation + Decay Iteration: {frame}') # Option 1: Title
    iter_text.set_text(f'Iteration: {frame}')       # Option 2: Text on plot

    # Return the artists that were modified (needed for blitting)
    return quiver_artist, path_line, iter_text

# --- Create and Save Animation ---
print("Creating animation...")
# blit=True attempts faster rendering but requires the update function
# to return an iterable of all modified artists.
ani = animation.FuncAnimation(fig, update, frames=NUM_FRAMES,
                              interval=INTERVAL, blit=True, repeat=False) # No repeat for decay

try:
    print(f"Saving animation to {FILENAME}...")
    writer = animation.PillowWriter(fps=1000 / INTERVAL)
    ani.save(FILENAME, writer=writer)
    print(f"Animation saved successfully to {FILENAME}")
except Exception as e:
    print(f"\n--- Error saving animation ---")
    print(f"{type(e).__name__}: {e}")
    print("------------------------------")
    print(" Ensure Pillow is installed (`pip install Pillow`).")
    # Fallback or further action
    # plt.show() # Uncomment to try showing plot if save fails

# plt.show() # Uncomment to display plot window regardless of saving

print("Script finished.")