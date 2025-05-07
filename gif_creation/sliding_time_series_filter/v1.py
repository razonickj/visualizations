import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter1d
# from scipy.signal import windows # Alternative for kernel generation

# --- Parameters ---
N_POINTS = 200         # Number of points in the time series
SIGMA_FILTER = 5.0     # Standard deviation for the Gaussian filter (in terms of data points)
NOISE_LEVEL = 0.5
# Kernel width: typically cover +/- 3 to 4 sigma.
# The effective kernel size used by gaussian_filter1d is determined by `truncate` parameter,
# which defaults to 4.0. So, the window is about 4*sigma on each side.
KERNEL_DISPLAY_RADIUS_SIGMAS = 3 # How many sigmas to display for the kernel shape plot
                                 # and for the sliding window visual.
KERNEL_VISUAL_WIDTH_POINTS = int(2 * KERNEL_DISPLAY_RADIUS_SIGMAS * SIGMA_FILTER + 1)


INTERVAL_MS = 50       # Milliseconds between frames for the animation
FPS_GIF = int(1000 / INTERVAL_MS)

# --- 1. Generate Time Series Data ---
np.random.seed(42)
x_time = np.arange(N_POINTS)
# Create a base signal (e.g., sine wave + step)
signal_clean = (np.sin(x_time * 0.1) * 2 +
                np.sin(x_time * 0.03) * 3 +
                2 * (x_time > N_POINTS * 0.4) -
                3 * (x_time > N_POINTS * 0.7))
signal_noisy = signal_clean + np.random.normal(0, NOISE_LEVEL, N_POINTS)

# --- 2. Pre-calculate Filtered Data ---
# scipy.ndimage.gaussian_filter1d handles edges well.
# The 'truncate' parameter determines how many sigmas the kernel extends. Default is 4.0.
filtered_signal = gaussian_filter1d(signal_noisy, sigma=SIGMA_FILTER, mode='reflect')

# Data for animation: initially NaN, filled as filter "moves"
revealed_filtered_signal = np.full(N_POINTS, np.nan)

# --- 3. Generate Gaussian Kernel for Display ---
kernel_display_radius_points = int(KERNEL_DISPLAY_RADIUS_SIGMAS * SIGMA_FILTER)
kernel_display_x = np.arange(-kernel_display_radius_points, kernel_display_radius_points + 1)
# Using the Gaussian PDF formula for the kernel shape for display
# (Not directly used for filtering if using scipy.ndimage.gaussian_filter1d)
kernel_display_y = (1 / (SIGMA_FILTER * np.sqrt(2 * np.pi))) * \
                   np.exp(-kernel_display_x**2 / (2 * SIGMA_FILTER**2))
kernel_display_y /= np.max(kernel_display_y) # Normalize for display height

# --- 4. Plot Setup ---
fig = plt.figure(figsize=(12, 7))
gs = fig.add_gridspec(3, 1, height_ratios=[1, 5, 1]) # Gridspec for layout

# Top plot: Gaussian Kernel Shape
ax_kernel = fig.add_subplot(gs[0])
ax_kernel.plot(kernel_display_x, kernel_display_y, color='green', lw=2)
ax_kernel.set_title(f"Gaussian Kernel Shape (Displayed for $\pm{KERNEL_DISPLAY_RADIUS_SIGMAS}\sigma$, $\sigma={SIGMA_FILTER:.1f}$ points)")
ax_kernel.set_xlabel("Kernel Offset (points)")
ax_kernel.set_ylabel("Weight (Normalized)")
ax_kernel.grid(True, linestyle=':', alpha=0.7)
# Remove y-ticks for cleaner kernel plot if desired
ax_kernel.set_yticks([])


# Main plot: Time Series
ax_main = fig.add_subplot(gs[1])
line_original, = ax_main.plot(x_time, signal_noisy, label='Original Noisy Signal', color='lightblue', lw=1.5)
line_filtered, = ax_main.plot(x_time, revealed_filtered_signal, label='Filtered Signal (Gaussian)', color='orange', lw=2)
ax_main.set_title("Time Series Filtering with Gaussian Filter")
ax_main.set_xlabel("Time / Sample Index")
ax_main.set_ylabel("Value")
ax_main.legend(loc='upper right')
ax_main.grid(True, linestyle=':', alpha=0.7)
y_min, y_max = np.min(signal_noisy), np.max(signal_noisy)
ax_main.set_ylim(y_min - (y_max-y_min)*0.1, y_max + (y_max-y_min)*0.1)


# Sliding box visualization (two vertical lines)
kernel_half_width_visual = KERNEL_DISPLAY_RADIUS_SIGMAS * SIGMA_FILTER
vline_left = ax_main.axvline(x=0 - kernel_half_width_visual, color='red', linestyle='--', lw=1.5, alpha=0.7)
vline_right = ax_main.axvline(x=0 + kernel_half_width_visual, color='red', linestyle='--', lw=1.5, alpha=0.7)

# Bottom plot: Progress text (optional)
ax_progress = fig.add_subplot(gs[2])
ax_progress.axis('off') # Hide axis
progress_text = ax_progress.text(0.5, 0.5, "", ha='center', va='center', fontsize=10)


plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout for suptitle if any, or main title

# --- 5. Animation Logic ---
# The animation will iterate from point 0 to N_POINTS-1
total_frames = N_POINTS

def animate(frame_idx):
    # Update sliding box position
    current_center_x = x_time[frame_idx]
    left_edge = current_center_x - kernel_half_width_visual
    right_edge = current_center_x + kernel_half_width_visual
    
    vline_left.set_xdata([left_edge, left_edge])
    vline_right.set_xdata([right_edge, right_edge])

    # Reveal the filtered signal up to the current point
    # The filtered_signal[frame_idx] is the value corresponding to input at frame_idx
    revealed_filtered_signal[frame_idx] = filtered_signal[frame_idx]
    line_filtered.set_data(x_time[:frame_idx+1], revealed_filtered_signal[:frame_idx+1])
    
    # Update progress text
    progress_text.set_text(f"Processing point {frame_idx+1}/{N_POINTS}")

    # Handle resetting for the next cycle if ANIMATION_CYCLES > 1
    # For a single pass GIF, this isn't strictly needed for the final output,
    # but useful if `repeat=True` in FuncAnimation for live viewing.
    if frame_idx == total_frames - 1:
        # For the last frame of the cycle, ensure the full line is plotted if we want it to persist
        # And then prepare for reset if live looping
        line_filtered.set_data(x_time, filtered_signal) # Show full on last frame of cycle
        if (animate.runs_completed + 1) * total_frames < ani.event_source.get_frameCount(): # Check if not the very last frame of entire animation
             # Reset for next visual loop in live animation (not affecting GIF if ANIMATION_CYCLES=1)
            revealed_filtered_signal[:] = np.nan


    return line_filtered, vline_left, vline_right, progress_text

animate.runs_completed = 0 # Counter for cycles if needed for more complex reset logic

def on_frame_end(frame_num): # To manage state for multi-cycle live viewing
    if (frame_num + 1) % total_frames == 0:
        animate.runs_completed += 1
    return []


# --- 6. Create and Save Animation ---
print(f"Total frames for animation cycle: {total_frames}")
print(f"Animation interval: {INTERVAL_MS}ms, GIF FPS: {FPS_GIF}")

ani = animation.FuncAnimation(fig, animate, frames=total_frames, # For one pass
                              init_func=lambda: on_frame_end(-1), # Call on_frame_end before first frame if needed
                              interval=INTERVAL_MS, blit=False, repeat=True)
# Using blit=False as it's generally more robust with complex updates and text.

gif_filename = 'timeseries_gaussian_filter.gif'
print(f"Attempting to save animation to {gif_filename}...")
try:
    # For saving, typically want repeat=False in the GIF itself unless specified
    # FuncAnimation's repeat parameter is for live viewing.
    # Pillow writer by default doesn't loop. To make GIF loop: writer.loop = 0
    ani.save(gif_filename, writer='pillow', fps=FPS_GIF)
    print(f"Successfully saved animation to {gif_filename}")
except Exception as e:
    print(f"Error saving animation: {e}")
    print("Ensure you have 'pillow' installed (pip install pillow).")

plt.close(fig)