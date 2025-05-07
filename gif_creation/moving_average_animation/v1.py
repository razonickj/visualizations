import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

# --- Parameters ---
DATA_SIZE = (10, 10)  # Height, Width of the heatmap
KERNEL_SIZE = 3       # Size of the moving average box (should be odd)
INTERVAL = 200         # Milliseconds between frames for the animation

# --- Data Initialization ---
# Create some sample data (e.g., random noise)
np.random.seed(1984) # for reproducibility
original_data = np.random.rand(DATA_SIZE[0], DATA_SIZE[1]) * 10
# Add a smoother feature for visual interest
center_y, center_x = DATA_SIZE[0] // 3, DATA_SIZE[1] // 2
y, x = np.ogrid[:DATA_SIZE[0], :DATA_SIZE[1]]
original_data += 15 * np.exp(-((x - center_x)**2 + (y - center_y)**2) / 80.0)


# Initialize filtered data array with NaNs (or zeros)
# NaNs are good because they won't be plotted until calculated
filtered_data = np.full_like(original_data, np.nan)

# --- Filter Calculations Setup ---
if KERNEL_SIZE % 2 == 0:
    raise ValueError("KERNEL_SIZE must be odd")
kernel_radius = KERNEL_SIZE // 2

# Calculate the range of valid center positions for the kernel
min_row_center = kernel_radius
max_row_center = DATA_SIZE[0] - kernel_radius - 1
min_col_center = kernel_radius
max_col_center = DATA_SIZE[1] - kernel_radius - 1

valid_height = max_row_center - min_row_center + 1
valid_width = max_col_center - min_col_center + 1
total_frames = valid_height * valid_width

print(f"Heatmap Size: {DATA_SIZE}")
print(f"Kernel Size: {KERNEL_SIZE}x{KERNEL_SIZE}")
print(f"Number of frames to calculate: {total_frames}")

# Determine consistent color limits based on original data
vmin = np.min(original_data)
vmax = np.max(original_data)

# --- Animation Setup ---
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("Moving Average Filter Animation")

# Plot Original Data (Left)
ax_orig = axes[0]
im_orig = ax_orig.imshow(original_data, cmap='viridis', vmin=vmin, vmax=vmax,
                         interpolation='nearest')
ax_orig.set_title("Original Data + Filter Box")
ax_orig.set_xticks([])
ax_orig.set_yticks([])
fig.colorbar(im_orig, ax=ax_orig, fraction=0.046, pad=0.04)

# Add the moving box patch (Rectangle) - initially off-screen or hidden
# Note: imshow plots with (0,0) at top-left, but patch coords need careful handling
# Rectangle(xy=(col,row), width, height)
# For imshow, the pixel (row, col) covers the area [col-0.5, col+0.5] x [row-0.5, row+0.5]
box_patch = patches.Rectangle(
    (-0.5, -0.5), KERNEL_SIZE, KERNEL_SIZE, # Width, Height
    linewidth=1.5, edgecolor='red', facecolor='none',
    label=f'{KERNEL_SIZE}x{KERNEL_SIZE} Box'
)
ax_orig.add_patch(box_patch)
ax_orig.legend(handles=[box_patch], loc='lower center', bbox_to_anchor=(0.5, -0.15))

# Plot Filtered Data (Right)
ax_filt = axes[1]
# Use the same vmin/vmax for comparison
im_filt = ax_filt.imshow(filtered_data, cmap='viridis', vmin=vmin, vmax=vmax,
                         interpolation='nearest')
ax_filt.set_title("Filtered Data (Updating)")
ax_filt.set_xticks([])
ax_filt.set_yticks([])
fig.colorbar(im_filt, ax=ax_filt, fraction=0.046, pad=0.04)

plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout

# --- Animation Function ---
def animate(frame):
    if frame >= total_frames: # Should not happen with repeat=False, but good practice
        return im_filt, box_patch

    # Calculate the row and column of the center pixel for this frame
    # Iterating through the *valid* center positions
    valid_row_idx = frame // valid_width
    valid_col_idx = frame % valid_width

    center_row = min_row_center + valid_row_idx
    center_col = min_col_center + valid_col_idx

    # Define kernel boundaries (inclusive start, exclusive end for slicing)
    row_start = center_row - kernel_radius
    row_end = center_row + kernel_radius + 1
    col_start = center_col - kernel_radius
    col_end = center_col + kernel_radius + 1

    # Extract the data patch under the kernel
    data_patch = original_data[row_start:row_end, col_start:col_end]

    # Calculate the average and update the filtered data array
    average_value = np.mean(data_patch)
    filtered_data[center_row, center_col] = average_value

    # Update the filtered heatmap display
    im_filt.set_data(filtered_data)

    # Update the position of the red box on the original heatmap
    # The bottom-left corner of the rectangle patch needs to be at (col_start - 0.5, row_start - 0.5)
    # because imshow centers pixels.
    box_patch.set_xy((col_start - 0.5, row_start - 0.5))

    # Return the modified plot elements for blitting
    return im_filt, box_patch

# --- Create and Save Animation ---
print("Creating animation... This might take a while.")

ani = animation.FuncAnimation(fig, animate, frames=total_frames,
                              interval=INTERVAL, blit=True, repeat=False)

# Save as GIF
gif_filename = 'moving_average_heatmap.gif'
try:
    ani.save(gif_filename, writer='pillow', fps=int(1000 / INTERVAL))
    print(f"Successfully saved animation to {gif_filename}")
except ImportError:
    print("Pillow not found. Install with 'pip install Pillow'. Cannot save GIF.")
except Exception as e:
    print(f"Error saving animation: {e}")

plt.close(fig) # Close the plot window