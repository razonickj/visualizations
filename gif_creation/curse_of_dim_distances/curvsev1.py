import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import gaussian_kde
from scipy.spatial.distance import pdist  # Efficient pairwise distances
import time

# --- Constants ---
NUM_POINTS = 1000        # Number of points to sample in the hypercube (more points = smoother PDF)
MAX_DIMENSION = 300     # Maximum dimension to simulate
# Define the sequence of dimensions (more steps at lower D for visual timing)
dims_low = list(range(2, 11))          # Steps of 1 (9 frames)
dims_mid = list(range(12, 31, 2))      # Steps of 2 (10 frames)
dims_high = list(range(35, 101, 5))     # Steps of 5 (14 frames)
dims_vhigh = list(range(110, MAX_DIMENSION + 1, 10)) # Steps of 10 (~20 frames)
DIMENSIONS = dims_low + dims_mid + dims_high + dims_vhigh

INTERVAL = 250         # Milliseconds between frames (adjust for desired speed)
FILENAME = "curse_dimensionality_distances.gif"
KDE_POINTS = 200       # Resolution for plotting KDE curve

# --- Pre-calculation ---
print("Pre-calculating pairwise distance distributions...")
results = {}           # Store KDE results (x, y data) for each dimension
all_stats = {}         # Store mean and relative std dev (std/mean)
rng = np.random.default_rng() # Recommended random number generator
max_density = 0        # Track max PDF value for consistent Y axis

start_time = time.time()
for N in DIMENSIONS:
    print(f"  Calculating for Dimension N = {N}...")
    # Generate points uniformly in N-dimensional unit hypercube [0,1]^N
    points = rng.random(size=(NUM_POINTS, N))

    # Calculate all pairwise distances efficiently
    # pdist returns a condensed distance matrix (1D array)
    if NUM_POINTS <= 1: # Need at least 2 points for pairwise distance
         print(f"    Skipping N={N}, NUM_POINTS too small.")
         continue
    distances = pdist(points)

    # Normalize distances by sqrt(N) - theoretical mean scaling factor in hypercube
    # This helps keep the distributions centered for better comparison of shape
    if N > 0:
        normalized_distances = distances / np.sqrt(N)
    else: # Avoid division by zero if N=0 (though N starts at 2 here)
        normalized_distances = distances

    if normalized_distances.size == 0:
         print(f"    Skipping N={N}, no distances calculated.")
         continue

    # Calculate statistics on normalized distances
    mean_norm_dist = np.mean(normalized_distances)
    std_norm_dist = np.std(normalized_distances)
    # Relative Standard Deviation (Coefficient of Variation) - should decrease
    relative_std_dev = std_norm_dist / mean_norm_dist if mean_norm_dist > 1e-9 else 0
    all_stats[N] = {'mean': mean_norm_dist, 'std': std_norm_dist, 'cv': relative_std_dev}

    # Perform Kernel Density Estimation on normalized distances
    try:
        # Bandwidth selection can be important; default 'scott' or 'silverman' is usually ok
        kde = gaussian_kde(normalized_distances)
        # Determine a suitable plotting range for normalized distances.
        # Theory suggests mean ~ sqrt(1/6) ~ 0.4. Range [0, 1] should suffice.
        x_plot = np.linspace(0, 1.0, KDE_POINTS) # Normalized distance axis
        y_plot = kde(x_plot)
        results[N] = {'x': x_plot, 'y': y_plot}
        max_density = max(max_density, np.max(y_plot)) # Find max density for y-axis limit
        print(f"    N={N}: Mean(d/√N) ≈ {mean_norm_dist:.3f}, CV ≈ {relative_std_dev:.3f}")
    except Exception as e:
        print(f"    N={N}: KDE failed! {type(e).__name__}: {e}")
        # Store dummy data if KDE fails
        results[N] = {'x': np.linspace(0, 1.0, KDE_POINTS), 'y': np.zeros(KDE_POINTS)}
        all_stats[N] = {'mean': np.nan, 'std': np.nan, 'cv': np.nan} # Store NaN stats

end_time = time.time()
print(f"Pre-calculation finished in {end_time - start_time:.2f} seconds.")
print(f"Overall max density found: {max_density:.3f}")

# --- Animation Setup ---
fig, ax = plt.subplots(figsize=(9, 6)) # Wider figure
ax.set_xlim(0, 1.0)                   # X-axis is normalized distance
# Set fixed y-limit based on observed max density + padding
ax.set_ylim(0, max_density * 1.15)
ax.set_xlabel("Normalized Pairwise Distance (d / sqrt(N))")
ax.set_ylabel("Probability Density")
ax.grid(True, linestyle='--', alpha=0.6)

# Initialize plot elements to be updated
pdf_line, = ax.plot([], [], 'b-', lw=2.5, alpha=0.8)
title_text = ax.set_title("") # Placeholder, updated each frame
stats_text = ax.text(0.97, 0.97, '', transform=ax.transAxes, ha='right', va='top',
                     fontsize=10, bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

# --- Animation Function ---
def update(frame_index):
    """Updates the plot for each frame based on the dimension index."""
    dimension = DIMENSIONS[frame_index]
    title_text.set_text(f'Distribution of Normalized Pairwise Distances (N = {dimension})')
    print(f" Animating Frame {frame_index+1}/{len(DIMENSIONS)}, Dimension = {dimension}") # Progress

    if dimension in results:
        data = results[dimension]
        stats = all_stats.get(dimension, {'mean': -1, 'cv': -1}) # Default if stats missing

        # Update the PDF line data
        pdf_line.set_data(data['x'], data['y'])

        # Update the statistics text
        stats_str = f"Mean(d/√N) ≈ {stats['mean']:.3f}\nStdDev/Mean ≈ {stats['cv']:.3f}"
        stats_text.set_text(stats_str)
    else:
        # Clear plot if data is missing for this dimension
        pdf_line.set_data([], [])
        stats_text.set_text("Data N/A")

    # Return artists modified for blitting
    return pdf_line, title_text, stats_text

# --- Create Animation ---
print("Creating animation (this may take a moment)...")
fig.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to prevent title overlap
ani = animation.FuncAnimation(fig, update, frames=len(DIMENSIONS),
                              interval=INTERVAL, blit=True, repeat=False)

# --- Save Animation ---
try:
    print(f"Saving animation to {FILENAME}...")
    # Use PillowWriter for GIF
    writer = animation.PillowWriter(fps=(1000 / INTERVAL))
    ani.save(FILENAME, writer=writer)
    print(f"Animation saved successfully to {FILENAME}")
except Exception as e:
    print(f"\n--- Error saving animation ---")
    print(f"{type(e).__name__}: {e}")
    print("------------------------------")
    print(" Ensure Pillow is installed (`pip install Pillow`).")
    print(" You may also need 'imagemagick' or 'ffmpeg' for other writers.")
    # Fallback option
    # print(" Trying to show plot instead...")
    # plt.show()

# plt.show() # Uncomment to display plot window after saving or if saving fails
print("Script finished.")