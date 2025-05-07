import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import imageio
import os
import shutil
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection # <<< CORRECT IMPORT ADDED HERE
import warnings # To suppress UserWarning about forcing aspect ratio


# --- Parameters ---
# Define P and Q (Normal distributions)
mu_p, sigma_p = 0.0, 1.0
mu_q, sigma_q = 2.0, 1.5
P_dist = norm(loc=mu_p, scale=sigma_p)
Q_dist = norm(loc=mu_q, scale=sigma_q)

# Plotting and Calculation Range
x_min = -5
x_max = 7
x_res = 200  # Resolution for plots and numerical integration
x = np.linspace(x_min, x_max, x_res)
dx = x[1] - x[0] # Step size for numerical integration

# Numerical stability epsilon
eps = 1e-10

# Animation parameters
N_FRAMES_PER_PHASE = 60 # Frames for each visualization phase (P||Q, Q||P)
N_PAUSE_FRAMES = 30     # Frames for static display phases (Intro, Results)
FPS = 15                # Frames per second for the output video
GIF_FILENAME = 'kl_divergence_asymmetry.gif'
TEMP_FRAME_DIR = 'kl_frames' # Temporary directory

# --- Pre-calculate PDF values ---
p_x = P_dist.pdf(x)
q_x = Q_dist.pdf(x)

# --- Pre-calculate KL components and values ---
# Ensure PDFs don't go to true zero for stable calculation
p_x_safe = np.maximum(p_x, eps)
q_x_safe = np.maximum(q_x, eps)

log_ratio_p_q = np.log2(p_x_safe / q_x_safe)
log_ratio_q_p = np.log2(q_x_safe / p_x_safe)

# Integrands for KL divergence
integrand_p_q = p_x * log_ratio_p_q
integrand_q_p = q_x * log_ratio_q_p

# Calculate total KL values numerically (summing weighted contributions)
# Filter out regions where the weighting probability is near zero
kl_p_q = np.sum(integrand_p_q[p_x > eps] * dx)
kl_q_p = np.sum(integrand_q_p[q_x > eps] * dx)

# For coloring based on log ratio
norm_p_q = mcolors.Normalize(vmin=min(0, np.min(log_ratio_p_q)), vmax=max(0.1, np.max(log_ratio_p_q)))
norm_q_p = mcolors.Normalize(vmin=min(0, np.min(log_ratio_q_p)), vmax=max(0.1, np.max(log_ratio_q_p)))
cmap = plt.get_cmap('coolwarm') # Blue (low ratio) to Red (high ratio)

# --- Generate Frames ---
if os.path.exists(TEMP_FRAME_DIR):
    shutil.rmtree(TEMP_FRAME_DIR)
os.makedirs(TEMP_FRAME_DIR)

frame_files = []
total_frames = 2 * N_PAUSE_FRAMES + 2 * N_FRAMES_PER_PHASE # Intro, P||Q, Q||P, Result
phase_starts = [0, N_PAUSE_FRAMES, N_PAUSE_FRAMES + N_FRAMES_PER_PHASE,
                N_PAUSE_FRAMES + 2 * N_FRAMES_PER_PHASE]

print(f"Generating {total_frames} frames...")
for i in range(total_frames):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, max(p_x.max(), q_x.max()) * 1.1)
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density')
    ax.grid(True, alpha=0.3)

    # Plot base distributions lightly
    ax.plot(x, p_x, color='blue', linestyle='--', linewidth=1.5, alpha=0.4, label='P(x)')
    ax.plot(x, q_x, color='orangered', linestyle='--', linewidth=1.5, alpha=0.4, label='Q(x)')

    phase_title = ""
    kl_text = ""

    # Determine phase
    if i < phase_starts[1]: # Phase 0: Intro - Show P and Q
        phase = 0
        phase_title = "Distributions P(x) and Q(x)"
        ax.plot(x, p_x, color='blue', linewidth=2.5, alpha=0.9) # Emphasize
        ax.plot(x, q_x, color='orangered', linewidth=2.5, alpha=0.9) # Emphasize
        kl_text = "KL Divergence measures difference between distributions.\nIt is asymmetric: D(P||Q) ≠ D(Q||P)"

    elif i < phase_starts[2]: # Phase 1: Visualize D_KL(P || Q)
        phase = 1
        phase_title = "Visualizing D_KL(P || Q) = Σ P(x) * log₂(P(x) / Q(x))"
        kl_text = f"Focus on P(x). Penalty where Q(x) is low but P(x) is high.\nD(P||Q) = {kl_p_q:.3f} bits"
        # Emphasize P
        ax.plot(x, p_x, color='blue', linewidth=2.5, alpha=0.9)
        # Shade under P, colored by log_ratio_p_q
        points = np.array([x, p_x]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # Create a LineCollection object with varying colors
        # vvv CORRECTED LINE vvv
        lc = LineCollection(segments, cmap=cmap, norm=norm_p_q, linewidth=0) # Linewidth 0 just for color norm
        lc.set_array(log_ratio_p_q) # Set colors based on log ratio
        # Apply the colored shading using facecolors derived from LineCollection
        try:
            # Need to add lc to axes for facecolors to populate if using this method
            # ax.add_collection(lc) # Add the collection (though linewidth=0 means it's invisible)
            fig.canvas.draw() # Need to draw to populate facecolors? May vary.
            poly = ax.fill_between(x, 0, p_x, facecolors=cmap(norm_p_q(log_ratio_p_q)), alpha=0.6)
        except Exception as e:
            # print(f"Warning: fill_between with facecolors failed: {e}. Using fallback.") # Optional debug
            poly = ax.fill_between(x, 0, p_x, color='lightblue', alpha=0.6) # Fallback simple shade


    elif i < phase_starts[3]: # Phase 2: Visualize D_KL(Q || P)
        phase = 2
        phase_title = "Visualizing D_KL(Q || P) = Σ Q(x) * log₂(Q(x) / P(x))"
        kl_text = f"Focus on Q(x). Penalty where P(x) is low but Q(x) is high.\nD(Q||P) = {kl_q_p:.3f} bits"
        # Emphasize Q
        ax.plot(x, q_x, color='orangered', linewidth=2.5, alpha=0.9)
        # Shade under Q, colored by log_ratio_q_p
        points = np.array([x, q_x]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # vvv CORRECTED LINE vvv
        lc = LineCollection(segments, cmap=cmap, norm=norm_q_p, linewidth=0)
        lc.set_array(log_ratio_q_p)
        try:
            # ax.add_collection(lc)
            fig.canvas.draw()
            poly = ax.fill_between(x, 0, q_x, facecolors=cmap(norm_q_p(log_ratio_q_p)), alpha=0.6)
        except Exception as e:
            # print(f"Warning: fill_between with facecolors failed: {e}. Using fallback.") # Optional debug
            poly = ax.fill_between(x, 0, q_x, color='lightsalmon', alpha=0.6) # Fallback simple shade


    else: # Phase 3: Show Results
        phase = 3
        phase_title = "KL Divergence Asymmetry Result"
        ax.plot(x, p_x, color='blue', linewidth=2.5, alpha=0.9) # Emphasize
        ax.plot(x, q_x, color='orangered', linewidth=2.5, alpha=0.9) # Emphasize
        kl_text = f"D_KL(P || Q) = {kl_p_q:.3f} bits\n" \
                  f"D_KL(Q || P) = {kl_q_p:.3f} bits\n" \
                  f"Note: D_KL(P || Q) ≠ D_KL(Q || P)"


    # Add titles and text
    ax.set_title(phase_title, fontsize=14)
    ax.text(0.02, 0.95, kl_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.8))
    # Only show legend in intro and result phases for clarity
    if phase in [0, 3]:
        ax.legend(loc='upper right')
    plt.tight_layout()

    # Save frame
    frame_filename = os.path.join(TEMP_FRAME_DIR, f'frame_{i:04d}.png')
    plt.savefig(frame_filename)
    frame_files.append(frame_filename)
    plt.close(fig) # Close plot to free memory

print("Frames generated.")

# --- Compile GIF ---
print(f"Compiling GIF: {GIF_FILENAME}...")
with imageio.get_writer(GIF_FILENAME, mode='I', duration=int(1000/FPS), loop=0) as writer:
    for filename in frame_files:
        image = imageio.imread(filename)
        writer.append_data(image)

print("GIF compiled.")

# --- Clean up temporary frames ---
print(f"Cleaning up temporary files in {TEMP_FRAME_DIR}...")
shutil.rmtree(TEMP_FRAME_DIR)
print("Done.")