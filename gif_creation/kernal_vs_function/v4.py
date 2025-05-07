import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from scipy.signal import convolve2d

# --- Parameters ---
GRID_SIZE_REGULAR = 10   # Input grid for kernel method
GRID_SIZE_OUTPUT = 7    # Output grid for animation steps AND for kernel's displayed smoothed output
N_IRREGULAR_POINTS = 300
NOISE_LEVEL = 0.3
DOMAIN_LIMIT = 5.0

SIGMA = 0.8
KERNEL_RADIUS_SIGMAS = 2

pixel_world_size_regular = (2 * DOMAIN_LIMIT) / GRID_SIZE_REGULAR
kernel_world_extent = KERNEL_RADIUS_SIGMAS * SIGMA * 2
KERNEL_SIZE_DISCRETE = int(round(kernel_world_extent / pixel_world_size_regular))
KERNEL_SIZE_DISCRETE = max(3, KERNEL_SIZE_DISCRETE // 2 * 2 + 1)

INTERVAL_MS = 150
FPS_GIF = max(1, int(1000 / INTERVAL_MS))
ANIMATION_CYCLES = 1

print(f"Regular Grid Size (Input for Kernel Method): {GRID_SIZE_REGULAR}x{GRID_SIZE_REGULAR}")
print(f"Discrete Kernel Size (pixels on regular grid): {KERNEL_SIZE_DISCRETE}x{KERNEL_SIZE_DISCRETE}")
print(f"Sigma (world units): {SIGMA}")
print(f"Output Grid Size (for animation steps & kernel display): {GRID_SIZE_OUTPUT}x{GRID_SIZE_OUTPUT}")

# --- 1. Data Generation ---
def true_signal(x, y):
    return (np.exp(-((x - 1.5)**2 + (y - 1.5)**2) / 2) +
            0.8 * np.exp(-((x + 2)**2 + (y + 1.5)**2) / 3) -
            0.5 * np.exp(-((x)**2 + (y - 2.5)**2) / 1.5) )

x_reg = np.linspace(-DOMAIN_LIMIT, DOMAIN_LIMIT, GRID_SIZE_REGULAR)
y_reg = np.linspace(-DOMAIN_LIMIT, DOMAIN_LIMIT, GRID_SIZE_REGULAR)
X_reg, Y_reg = np.meshgrid(x_reg, y_reg)
original_regular_data = true_signal(X_reg, Y_reg)
noisy_regular_data = original_regular_data + np.random.normal(0, NOISE_LEVEL, X_reg.shape)

np.random.seed(42)
x_irreg = np.random.uniform(-DOMAIN_LIMIT, DOMAIN_LIMIT, N_IRREGULAR_POINTS)
y_irreg = np.random.uniform(-DOMAIN_LIMIT, DOMAIN_LIMIT, N_IRREGULAR_POINTS)
values_irreg_clean = true_signal(x_irreg, y_irreg)
noisy_values_irreg = values_irreg_clean + np.random.normal(0, NOISE_LEVEL, N_IRREGULAR_POINTS)

x_out = np.linspace(-DOMAIN_LIMIT, DOMAIN_LIMIT, GRID_SIZE_OUTPUT)
y_out = np.linspace(-DOMAIN_LIMIT, DOMAIN_LIMIT, GRID_SIZE_OUTPUT)
X_out, Y_out = np.meshgrid(x_out, y_out)

def gaussian_kernel_2d(size, sigma_val):
    ax_range = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax_range, ax_range)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma_val**2))
    return kernel / np.sum(kernel)

sigma_pixels_for_kernel = SIGMA / pixel_world_size_regular
discrete_gauss_kernel = gaussian_kernel_2d(KERNEL_SIZE_DISCRETE, sigma_pixels_for_kernel)
# This is the high-resolution smoothed data, on the GRID_SIZE_REGULAR grid
full_smoothed_regular_data = convolve2d(noisy_regular_data, discrete_gauss_kernel,
                                        mode='same', boundary='symm')

# ---- MODIFIED: Kernel method's display grid is now GRID_SIZE_OUTPUT ----
displayed_smoothed_kernel_output = np.full((GRID_SIZE_OUTPUT, GRID_SIZE_OUTPUT), np.nan)
# ---- END MODIFICATION ----
smoothed_functional_on_grid = np.full(X_out.shape, np.nan) # This was already (GRID_SIZE_OUTPUT, GRID_SIZE_OUTPUT)

vmin = min(np.min(noisy_regular_data), np.min(noisy_values_irreg)) * 0.8
vmax = max(np.max(noisy_regular_data), np.max(noisy_values_irreg)) * 1.2
if vmin >= 0 and vmax > 0 : vmin = -0.01 * vmax 
elif vmax <= 0 and vmin < 0 : vmax = -0.01 * vmin 
elif abs(vmax - vmin) < 0.1: vmax = vmin + 1.0

# --- 2. Plot Setup ---
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
fig.suptitle("Gaussian Filter: Kernel vs. Function", fontsize=16)
cmap = 'viridis'

ax_orig_reg = axes[0, 0]
im_orig_reg = ax_orig_reg.imshow(noisy_regular_data, cmap=cmap, vmin=vmin, vmax=vmax,
                                 extent=[-DOMAIN_LIMIT, DOMAIN_LIMIT, -DOMAIN_LIMIT, DOMAIN_LIMIT], origin='lower')
ax_orig_reg.set_title("Kernel: Original Regular Grid")
fig.colorbar(im_orig_reg, ax=ax_orig_reg, fraction=0.046, pad=0.04)

num_kernel_visual_layers = 3
kernel_visual_patches = []
base_kernel_patch_size_world = KERNEL_SIZE_DISCRETE * pixel_world_size_regular
alphas = np.linspace(0.4, 0.15, num_kernel_visual_layers)
for i in range(num_kernel_visual_layers):
    layer_fraction = (num_kernel_visual_layers - i) / num_kernel_visual_layers
    patch_size = base_kernel_patch_size_world * layer_fraction
    patch = patches.Rectangle((-DOMAIN_LIMIT*2, -DOMAIN_LIMIT*2), patch_size, patch_size,
                              linewidth=1, edgecolor='red', facecolor='red', alpha=alphas[i])
    ax_orig_reg.add_patch(patch)
    kernel_visual_patches.append(patch)

ax_smooth_reg = axes[1, 0]
# ---- MODIFIED: im_smooth_reg now displays the GRID_SIZE_OUTPUT resolution data ----
im_smooth_reg = ax_smooth_reg.imshow(displayed_smoothed_kernel_output, cmap=cmap, vmin=vmin, vmax=vmax,
                                     extent=[-DOMAIN_LIMIT, DOMAIN_LIMIT, -DOMAIN_LIMIT, DOMAIN_LIMIT], origin='lower')
# ---- END MODIFICATION ----
ax_smooth_reg.set_title(f"Kernel: Smoothed Output ({GRID_SIZE_OUTPUT}x{GRID_SIZE_OUTPUT})")
fig.colorbar(im_smooth_reg, ax=ax_smooth_reg, fraction=0.046, pad=0.04)

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
ax_smooth_irreg.set_title(f"Function: Smoothed Output ({GRID_SIZE_OUTPUT}x{GRID_SIZE_OUTPUT})")
fig.colorbar(im_smooth_irreg, ax=ax_smooth_irreg, fraction=0.046, pad=0.04)

for ax_row in axes:
    for ax in ax_row:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
plt.tight_layout(rect=[0, 0, 1, 0.95])

# --- 3. Animation Logic ---
total_frames = GRID_SIZE_OUTPUT * GRID_SIZE_OUTPUT

def animate(frame_num):
    frame = frame_num % total_frames

    row_out_idx = frame // GRID_SIZE_OUTPUT  # This is the row index for the *output* display grid
    col_out_idx = frame % GRID_SIZE_OUTPUT   # This is the col index for the *output* display grid
    
    focal_x_world = x_out[col_out_idx] # World coordinate of the center of the current output pixel
    focal_y_world = y_out[row_out_idx]

    # --- Kernel Method Update ---
    # 1. Determine where the kernel is centered on the high-res INPUT grid
    kernel_center_col_idx_orig = np.argmin(np.abs(x_reg - focal_x_world))
    kernel_center_row_idx_orig = np.argmin(np.abs(y_reg - focal_y_world))
    
    center_x_on_input_grid_world = x_reg[kernel_center_col_idx_orig]
    center_y_on_input_grid_world = y_reg[kernel_center_row_idx_orig]

    # 2. Update visual kernel patches on the input display
    for i, patch in enumerate(kernel_visual_patches):
        layer_fraction = (num_kernel_visual_layers - i) / num_kernel_visual_layers
        current_patch_size = base_kernel_patch_size_world * layer_fraction
        patch.set_width(current_patch_size)
        patch.set_height(current_patch_size)
        patch.set_xy((center_x_on_input_grid_world - current_patch_size / 2,
                      center_y_on_input_grid_world - current_patch_size / 2))

    # ---- MODIFIED: Get smoothed value and update the OUTPUT resolution display ----
    # 3. Get the pre-calculated smoothed value from the high-res smoothed data 
    #    at the point where the kernel is centered.
    smoothed_value_at_focal = full_smoothed_regular_data[kernel_center_row_idx_orig, kernel_center_col_idx_orig]
    
    # 4. Place this value into the corresponding cell of the kernel's output display grid.
    displayed_smoothed_kernel_output[row_out_idx, col_out_idx] = smoothed_value_at_focal
    im_smooth_reg.set_data(displayed_smoothed_kernel_output)
    # ---- END MODIFICATION ----

    # --- Functional Method Update ---
    func_influence_patch.center = (focal_x_world, focal_y_world)

    distances_sq = (x_irreg - focal_x_world)**2 + (y_irreg - focal_y_world)**2
    weights = np.exp(-distances_sq / (2 * SIGMA**2))
    weighted_sum_values = np.sum(weights * noisy_values_irreg)
    sum_of_weights = np.sum(weights)
    smoothed_val_functional = weighted_sum_values / sum_of_weights if sum_of_weights > 1e-9 else np.nan
    smoothed_functional_on_grid[row_out_idx, col_out_idx] = smoothed_val_functional # This is already GRID_SIZE_OUTPUT resolution
    im_smooth_irreg.set_data(smoothed_functional_on_grid)

    if frame == total_frames - 1 and ANIMATION_CYCLES > 1 and (frame_num + 1) < (total_frames * ANIMATION_CYCLES) :
        # Reset for next cycle if rendering multiple loops in one GIF
        displayed_smoothed_kernel_output[:] = np.nan
        smoothed_functional_on_grid[:] = np.nan

    return (*kernel_visual_patches, im_smooth_reg, func_influence_patch, im_smooth_irreg)

# --- Create and Save Animation ---
print(f"Total frames for animation cycle: {total_frames}")
print(f"Animation interval: {INTERVAL_MS}ms, GIF FPS: {FPS_GIF}")
animation_length_frames = total_frames * ANIMATION_CYCLES

ani = animation.FuncAnimation(fig, animate, frames=animation_length_frames,
                              interval=INTERVAL_MS, blit=False, repeat=True)

gif_filename = 'gaussian_filter_comparison_synced.gif'
print(f"Attempting to save animation to {gif_filename}...")
try:
    ani.save(gif_filename, writer='pillow', fps=FPS_GIF)
    print(f"Successfully saved animation to {gif_filename}")
except Exception as e:
    print(f"Error saving animation: {e}")
    print("Ensure you have 'pillow' installed (pip install pillow).")

plt.close(fig)