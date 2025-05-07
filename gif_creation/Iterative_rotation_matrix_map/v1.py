import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

# --- Configuration ---
INITIAL_VECTOR = np.array([2.0, 0.5]) # Starting vector (can be anything)
ANGLE_DEG_PER_STEP = 1.5             # Small rotation angle in degrees for each step
NUM_STEPS = 240                      # Total number of iterations/frames (360 / 1.5 = 240 for full circle)
INTERVAL = 40                        # Milliseconds between frames (controls animation speed)
FILENAME = "vector_rotation.gif"     # Output filename

# --- Setup ---
# Convert angle to radians
angle_rad_per_step = np.radians(ANGLE_DEG_PER_STEP)

# Create the 2D rotation matrix
# R = [[cos(theta), -sin(theta)],
#      [sin(theta),  cos(theta)]]
rotation_matrix = np.array([
    [np.cos(angle_rad_per_step), -np.sin(angle_rad_per_step)],
    [np.sin(angle_rad_per_step),  np.cos(angle_rad_per_step)]
])

# --- Pre-calculate Vector States ---
# Store the vector at each step for smooth animation plotting
vector_states = [INITIAL_VECTOR]
current_vector = INITIAL_VECTOR
print(f"Calculating {NUM_STEPS} vector states...")
for i in range(NUM_STEPS - 1):
    # Apply the rotation matrix: v_new = R * v_old
    current_vector = rotation_matrix @ current_vector # Matrix multiplication
    vector_states.append(current_vector)
print("Calculation complete.")

# --- Animation Setup ---
fig, ax = plt.subplots(figsize=(7, 7)) # Slightly larger figure

# Determine axis limits based on the vector magnitude (it traces a circle)
magnitude = np.linalg.norm(INITIAL_VECTOR)
limit = magnitude * 1.3 # Add some padding

ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)
ax.set_aspect('equal', adjustable='box') # Crucial for correct visual rotation
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")

# Add origin marker
ax.plot(0, 0, 'ko', ms=5) # Black dot at origin

# Placeholder for the arrow artist - important for efficient updates
# We use a list because ax.quiver returns a Quiver object, not a list of artists
# If using ax.arrow, it returns a single artist. Let's use quiver for this example.
quiver_artist = None

# --- Animation Function ---
def update(frame):
    """Updates the plot for each frame of the animation."""
    global quiver_artist # Use global to modify the artist object

    vector = vector_states[frame]

    # Remove the previous quiver plot if it exists
    if quiver_artist:
        quiver_artist.remove()

    # Plot the current vector using quiver
    # quiver(x_origin, y_origin, x_component, y_component, ...)
    quiver_artist = ax.quiver(0, 0, vector[0], vector[1],
                              angles='xy', scale_units='xy', scale=1,
                              color='darkcyan', width=0.008)

    # Update the title with the iteration number
    ax.set_title(f'Rotation Matrix Iteration: {frame}')

    # Return the artists that were modified (needed for blitting)
    # If not using blit=True, return value isn't strictly necessary
    return quiver_artist, # Return as a tuple

# --- Create and Save Animation ---
print("Creating animation...")
# blit=True tries to only redraw parts that change for potentially faster rendering
# repeat=True makes the GIF loop
ani = animation.FuncAnimation(fig, update, frames=NUM_STEPS,
                              interval=INTERVAL, blit=True, repeat=True)

try:
    print(f"Saving animation to {FILENAME}...")
    # Use PillowWriter for GIF
    writer = animation.PillowWriter(fps=1000 / INTERVAL)
    ani.save(FILENAME, writer=writer)
    print(f"Animation saved successfully to {FILENAME}")
except Exception as e:
    print(f"\n--- Error saving animation ---")
    print(f"{type(e).__name__}: {e}")
    print("------------------------------")
    print(" PillowWriter often requires Pillow installed (`pip install Pillow`).")
    print(" If saving still fails, try showing the plot instead.")
    # Fallback to showing plot if saving fails
    # plt.show() # Uncomment this line to show interactively if saving fails

# Optionally display the plot window after saving or if saving fails
# plt.show()

print("Script finished.")