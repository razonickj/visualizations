# Example vectorized approach
# rng = np.random.default_rng()
# a = rng.standard_normal(size=(num_samples, N))
# b = rng.standard_normal(size=(num_samples, N))
# # Normalize rows (vectors) to unit length
# a /= np.linalg.norm(a, axis=1, keepdims=True)
# b /= np.linalg.norm(b, axis=1, keepdims=True)
# # Dot product of corresponding rows (unit vectors) gives cos(theta)
# cos_theta = np.einsum('ij,ij->i', a, b) # Efficient row-wise dot product
# angles_rad = np.arccos(np.clip(np.abs(cos_theta), 0.0, 1.0)) # Clip ensures valid arccos input
# angles_deg = np.degrees(angles_rad)

#     * **KDE:** `gaussian_kde` works well. Ensure input is 1D array of angles. Handle potential errors if all angles are identical (unlikely here).
#     * **Plotting:** Use `line.set_data()` for efficiency in the `update` function.
#     * **Y-limits:** Automatically adjusting y-limits per frame (`ax.relim()`, `ax.autoscale_view()`) might cause visual "breathing" of the axis. Finding the max density across *all* dimensions first and setting a fixed `ylim` might be better for comparing shapes, even if lower-dimension PDFs look small. Let's calculate the max Y value across all pre-calculated KDEs.
#     * **Blitting:** Use `blit=True` and return modified artists from `update`.

# 6.  **Code Structure:**

#     ```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import gaussian_kde
import time # For timing pre-calculation

# --- Constants ---
NUM_SAMPLES = 10000      # Number of vector pairs per dimension
MAX_DIMENSION = 500     # Maximum dimension to simulate
# Define the sequence of dimensions (more steps at lower D)
dims_low = list(range(2, 11)) # 2-10 (9 steps)
dims_mid = list(range(12, 31, 2)) # 12-30 (10 steps)
dims_high = list(range(35, 101, 5)) # 35-100 (14 steps)
dims_vhigh = list(range(110, MAX_DIMENSION + 1, 10)) # 110-200 (10 steps)
DIMENSIONS = dims_low + dims_mid + dims_high + dims_vhigh
# DIMENSIONS = [2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200] # Simpler sequence

INTERVAL = 350        # Milliseconds between frames
FILENAME = "angle_distribution_dimensionality.gif"
KDE_POINTS = 300      # Resolution for plotting KDE curve

# --- Pre-calculation ---
print("Pre-calculating angle distributions...")
results = {} # Store KDE results (x, y data) for each dimension
rng = np.random.default_rng() # Modern way for random numbers
max_density = 0 # Track max PDF value for consistent Y axis

start_time = time.time()
for N in DIMENSIONS:
    # Generate pairs of random vectors efficiently
    # Use standard normal distribution N(0,1)
    a = rng.standard_normal(size=(NUM_SAMPLES, N))
    b = rng.standard_normal(size=(NUM_SAMPLES, N))

    # Normalize vectors to unit length (optional but simplifies dot product)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)

    # Calculate cosine of the angle (dot product of unit vectors)
    # Use einsum for efficient row-wise dot products
    cos_theta = np.einsum('ij,ij->i', a, b)

    # Ensure cosine is within valid range [-1, 1] due to potential float errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Calculate angle in degrees [0, 90]
    # arccos(abs(cos_theta)) gives the acute angle [0, pi/2]
    angles_rad = np.arccos(np.abs(cos_theta))
    angles_deg = np.degrees(angles_rad)

    # Perform Kernel Density Estimation
    try:
        kde = gaussian_kde(angles_deg)
        x_plot = np.linspace(0, 90, KDE_POINTS) # Angles in degrees
        y_plot = kde(x_plot)
        results[N] = {'x': x_plot, 'y': y_plot}
        max_density = max(max_density, np.max(y_plot))
        print(f"  Dim {N}: Done. Max density ~ {np.max(y_plot):.3f}")
    except Exception as e:
        print(f"  Dim {N}: KDE failed! {e}")
        # Store None or empty data if KDE fails
        results[N] = {'x': np.linspace(0, 90, KDE_POINTS), 'y': np.zeros(KDE_POINTS)}


end_time = time.time()
print(f"Pre-calculation finished in {end_time - start_time:.2f} seconds.")
print(f"Overall max density found: {max_density:.3f}")

# --- Animation Setup ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(0, 90)
# Set fixed y-limit based on observed max density + padding
ax.set_ylim(0, max_density * 1.1)
ax.set_xlabel("Angle between vectors (degrees)")
ax.set_ylabel("Probability Density")
ax.grid(True, linestyle='--', alpha=0.6)

# Initialize the plot line (empty at first)
pdf_line, = ax.plot([], [], 'b-', lw=2)
title_text = ax.set_title("") # Placeholder for title

# --- Animation Function ---
def update(frame_index):
    dimension = DIMENSIONS[frame_index]
    if dimension in results:
        data = results[dimension]
        pdf_line.set_data(data['x'], data['y'])
        title_text.set_text(f'{NUM_SAMPLES} Samples; Angle Distribution in Dimension N = {dimension}')
        print(f" Frame {frame_index+1}/{len(DIMENSIONS)}, Dim = {dimension}") # Progress indicator
    else:
            # Handle cases where KDE might have failed (though unlikely with checks)
            pdf_line.set_data([], [])
            title_text.set_text(f'Angle Distribution in Dimension N = {dimension} (No data)')

    return pdf_line, title_text

# --- Create Animation ---
print("Creating animation...")
ani = animation.FuncAnimation(fig, update, frames=len(DIMENSIONS),
                                interval=INTERVAL, blit=True, repeat=False)

# --- Save ---
try:
    print(f"Saving animation to {FILENAME}...")
    writer = animation.PillowWriter(fps=1000/INTERVAL)
    ani.save(FILENAME, writer=writer)
    print(f"Animation saved successfully to {FILENAME}")
except Exception as e:
    print(f"\n--- Error saving animation ---\n{type(e).__name__}: {e}\n------------------------------")
    # plt.show()

# plt.show()
print("Script finished.")
#     ```

# 7.  **Final Review:** Check the dimension sequence generation. Ensure vectorized angle calculation is correct (`einsum`, `clip`, `abs`, `arccos`, `degrees`). Verify KDE usage and storage of results. Confirm `update` function correctly retrieves data and updates the plot elements (`set_data`). Check `blit=True` requirements (returning artists). Check plot limits and labels. Looks plausible.