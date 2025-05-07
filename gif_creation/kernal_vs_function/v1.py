import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from scipy.signal import convolve2d
from scipy.stats import norm # For Gaussian function

# --- Parameters ---
# Data Generation
GRID_SIZE_REGULAR = 30  # For the regular input grid (e.g., 30x30)
# ---- MODIFIED PARAMETER FOR SHORTER ANIMATION ----
GRID_SIZE_OUTPUT = 15   # For the output smoothed grids (e.g., 15x15 for fewer frames)
# ---- END MODIFICATION ----
N_IRREGULAR_POINTS = 300
NOISE_LEVEL = 0.3
DOMAIN_LIMIT = 5.0 # Data spans from -DOMAIN_LIMIT to +DOMAIN_LIMIT

# Gaussian Filter
SIGMA = 0.8  # Standard deviation of the Gaussian
KERNEL_RADIUS_SIGMAS = 2 # Kernel/influence visualised out to this many sigmas
KERNEL_SIZE_DISCRETE = int(2 * KERNEL_RADIUS_SIGMAS * SIGMA / (2 * DOMAIN_LIMIT / GRID_SIZE_REGULAR) // 2 * 2 + 1)
KERNEL_SIZE_DISCRETE = max(3, KERNEL_SIZE_DISCRETE)


# Animation
INTERVAL_MS = 150 # Kept the same, so individual steps are still visible for the same duration
FPS_GIF = max(1, int(1000 / INTERVAL_MS))
ANIMATION_CYCLES = 1

print(f"Discrete Kernel Size: {KERNEL_SIZE_DISCRETE}x{KERNEL_SIZE_DISCRETE}")
print(f"Sigma: {SIGMA}")
print(f"Output Grid Size (for animation steps): {GRID_SIZE_OUTPUT}x{GRID_SIZE_OUTPUT}")

# --- 1. Data Generation ---
def true_signal(x, y):
    """The underlying clean signal."""
    return (np.exp(-((x - 1.5)**2 + (y - 1.5)**2) / 2) +
            0.8 * np.exp(-((x + 2)**2 + (y + 1.5)**2) / 3) -
            0.5 * np.exp(-((x)**2 + (y - 2.5)**2) / 1.5) )

# Regular Grid Data
x_reg = np.linspace(-DOMAIN_LIMIT, DOMAIN_LIMIT, GRID_SIZE_REGULAR)
y_reg = np.linspace(-DOMAIN_LIMIT, DOMAIN_LIMIT, GRID_SIZE_REGULAR)
X_reg, Y_reg = np.meshgrid(x_reg, y_reg)
original_regular_data = true_signal(X_reg, Y_reg)
noisy_regular_data = original_regular_data + np.random.normal(0, NOISE_LEVEL, X_reg.shape)

# Irregular Grid Data
np.random.seed(42)
x_irreg = np.random.uniform(-DOMAIN_LIMIT, DOMAIN_LIMIT, N_IRREGULAR_POINTS)
y_irreg = np.random.uniform(-DOMAIN_LIMIT, DOMAIN_LIMIT, N_IRREGULAR_POINTS)
values_irreg_clean = true_signal(x_irreg, y_irreg)
noisy_values_irreg = values_irreg_clean + np.random.normal(0, NOISE_LEVEL, N_IRREGULAR_POINTS)

# Output grid for functional smoothing and for revealing kernel smoothing
x_out = np.linspace(-DOMAIN_LIMIT, DOMAIN_LIMIT, GRID_SIZE_OUTPUT)
y_out = np.linspace(-DOMAIN_LIMIT, DOMAIN_LIMIT, GRID_SIZE_OUTPUT)
X_out, Y_out = np.meshgrid(x_out, y_out)

# Pre-calculate smoothed regular data using convolution
def gaussian_kernel_2d(size, sigma_pix):
    """Creates a 2D Gaussian kernel array."""
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma_pix**2))
    return kernel / np.sum(kernel)

sigma_pixels = SIGMA / (2 * DOMAIN_LIMIT / GRID_SIZE_REGULAR)
discrete_gauss_kernel = gaussian_kernel_2d(KERNEL_SIZE_DISCRETE, sigma_pixels)
full_smoothed_regular_data = convolve2d(noisy_regular_data, discrete_gauss_kernel,
                                        mode='same', boundary='symm')

# Data for animation: initially NaN, filled as filter moves
displayed_smoothed_regular = np.full((GRID_SIZE_REGULAR, GRID_SIZE_REGULAR), np.nan)
smoothed_functional_on_grid = np.full(X_out.shape, np.nan)


# Color limits for consistency
vmin = min(np.min(noisy_regular_data), np.min(noisy_values_irreg)) * 0.8
vmax = max(np.max(noisy_regular_data), np.max(noisy_values_irreg)) * 1.2
if vmin > 0 : vmin = -0.1 
if vmax < 0 : vmax = 0.1  
if vmax - vmin < 0.1: 
    vmax = vmin + 1.0


# --- 2. Plot Setup ---
fig, axes = plt.subplots(2, 2, figsize=(12, 12)) 
fig.suptitle("Gaussian Filter: Kernel vs. Function (Shorter)", fontsize=16)

cmap = 'viridis'

# Kernel Method Plots
ax_orig_reg = axes[0, 0]
im_orig_reg = ax_orig_reg.imshow(noisy_regular_data, cmap=cmap, vmin=vmin, vmax=vmax,
                                 extent=[-DOMAIN_LIMIT, DOMAIN_LIMIT, -DOMAIN_LIMIT, DOMAIN_LIMIT], origin='lower')
ax_orig_reg.set_title("Kernel: Original Regular Grid")
kernel_patch_size_world = KERNEL_SIZE_DISCRETE * (2 * DOMAIN_LIMIT / GRID_SIZE_REGULAR)
kernel_patch = patches.Rectangle((-DOMAIN_LIMIT*2, -DOMAIN_LIMIT*2), kernel_patch_size_world, kernel_patch_size_world,
                                 linewidth=1.5, edgecolor='red', facecolor='none', alpha=0.7)
ax_orig_reg.add_patch(kernel_patch)
fig.colorbar(im_orig_reg, ax=ax_orig_reg, fraction=0.046, pad=0.04)

ax_smooth_reg = axes[1, 0]
im_smooth_reg = ax_smooth_reg.imshow(displayed_smoothed_regular, cmap=cmap, vmin=vmin, vmax=vmax,
                                     extent=[-DOMAIN_LIMIT, DOMAIN_LIMIT, -DOMAIN_LIMIT, DOMAIN_LIMIT], origin='lower')
ax_smooth_reg.set_title("Kernel: Smoothed Output (Building)")
fig.colorbar(im_smooth_reg, ax=ax_smooth_reg, fraction=0.046, pad=0.04)

# Functional Method Plots
ax_orig_irreg = axes[0, 1]
sc_orig_irreg = ax_orig_irreg.scatter(x_irreg, y_irreg, c=noisy_values_irreg, cmap=cmap, vmin=vmin, vmax=vmax, s=15, alpha=0.7)
ax_orig_irreg.set_xlim(-DOMAIN_LIMIT, DOMAIN_LIMIT)
ax_orig_irreg.set_ylim(-DOMAIN_LIMIT, DOMAIN_LIMIT)
ax_orig_irreg.set_title("Function: Original Irregular Data")
ax_orig_irreg.set_aspect('equal')
func_influence_patch = patches.Circle((-DOMAIN_LIMIT*2, -DOMAIN_LIMIT*2), SIGMA * KERNEL_RADIUS_SIGMAS,
                                      edgecolor='red', facecolor='red', alpha=0.2)
ax_orig_irreg.add_patch(func_influence_patch)
fig.colorbar(sc_orig_irreg, ax=ax_orig_irreg, fraction=0.046, pad=0.04)


ax_smooth_irreg = axes[1, 1]
im_smooth_irreg = ax_smooth_irreg.imshow(smoothed_functional_on_grid, cmap=cmap, vmin=vmin, vmax=vmax,
                                         extent=[-DOMAIN_LIMIT, DOMAIN_LIMIT, -DOMAIN_LIMIT, DOMAIN_LIMIT], origin='lower')
ax_smooth_irreg.set_title("Function: Smoothed Output (Building)")
fig.colorbar(im_smooth_irreg, ax=ax_smooth_irreg, fraction=0.046, pad=0.04)

for ax_row in axes:
    for ax in ax_row:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

plt.tight_layout(rect=[0, 0, 1, 0.95])

# --- 3. Animation Logic ---
total_frames = GRID_SIZE_OUTPUT * GRID_SIZE_OUTPUT # This is now smaller

def animate(frame_num):
    # Since ANIMATION_CYCLES is 1, frame_num is effectively the frame within the cycle
    # If ANIMATION_CYCLES > 1, use frame_num % total_frames
    frame = frame_num % total_frames

    row_out_idx = frame // GRID_SIZE_OUTPUT
    col_out_idx = frame % GRID_SIZE_OUTPUT

    focal_x = x_out[col_out_idx]
    focal_y = y_out[row_out_idx]

    # --- Kernel Method Update ---
    kernel_center_col_idx_orig = np.searchsorted(x_reg, focal_x)
    kernel_center_row_idx_orig = np.searchsorted(y_reg, focal_y)
    kernel_center_col_idx_orig = np.clip(kernel_center_col_idx_orig, 0, GRID_SIZE_REGULAR -1)
    kernel_center_row_idx_orig = np.clip(kernel_center_row_idx_orig, 0, GRID_SIZE_REGULAR -1)

    patch_x = x_reg[kernel_center_col_idx_orig] - kernel_patch_size_world / 2
    patch_y = y_reg[kernel_center_row_idx_orig] - kernel_patch_size_world / 2
    kernel_patch.set_xy((patch_x, patch_y))

    displayed_smoothed_regular[kernel_center_row_idx_orig, kernel_center_col_idx_orig] = \
        full_smoothed_regular_data[kernel_center_row_idx_orig, kernel_center_col_idx_orig]
    im_smooth_reg.set_data(displayed_smoothed_regular)

    # --- Functional Method Update ---
    func_influence_patch.center = (focal_x, focal_y)

    distances_sq = (x_irreg - focal_x)**2 + (y_irreg - focal_y)**2
    weights = np.exp(-distances_sq / (2 * SIGMA**2)) 

    weighted_sum_values = np.sum(weights * noisy_values_irreg)
    sum_of_weights = np.sum(weights)

    if sum_of_weights > 1e-9: 
        smoothed_val_functional = weighted_sum_values / sum_of_weights
    else: 
        smoothed_val_functional = np.nan 

    smoothed_functional_on_grid[row_out_idx, col_out_idx] = smoothed_val_functional
    im_smooth_irreg.set_data(smoothed_functional_on_grid)

    # Reset data for the next cycle if ANIMATION_CYCLES > 1 (not strictly needed for CYCLES=1)
    if frame == total_frames -1 and ANIMATION_CYCLES > 1: # End of a cycle, and more cycles to come
        # This reset logic is mainly if we want the build-up to repeat visually per cycle
        # For ANIMATION_CYCLES=1, it doesn't affect the output GIF much.
        displayed_smoothed_regular[:] = np.nan
        smoothed_functional_on_grid[:] = np.nan


    return kernel_patch, im_smooth_reg, func_influence_patch, im_smooth_irreg


# --- Create and Save Animation ---
print(f"Total frames for animation cycle: {total_frames}")
print(f"Animation interval: {INTERVAL_MS}ms, GIF FPS: {FPS_GIF}")
# The `frames` argument in FuncAnimation determines the total length of the GIF
# For multiple cycles, it should be total_frames * ANIMATION_CYCLES
# For a single cycle (ANIMATION_CYCLES=1), it's just total_frames.
animation_length_frames = total_frames * ANIMATION_CYCLES

ani = animation.FuncAnimation(fig, animate, frames=animation_length_frames,
                              interval=INTERVAL_MS, blit=False, repeat=True) # repeat=True in FuncAnimation for live viewing

gif_filename = 'gaussian_filter_comparison_shorter.gif'
print(f"Attempting to save animation to {gif_filename}...")
try:
    ani.save(gif_filename, writer='pillow', fps=FPS_GIF)
    print(f"Successfully saved animation to {gif_filename}")
except Exception as e:
    print(f"Error saving animation: {e}")
    print("Ensure you have 'pillow' installed (pip install pillow).")

plt.close(fig)