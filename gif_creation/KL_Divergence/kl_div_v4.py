import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
import imageio # Use imageio-ffmpeg if you have issues: pip install imageio-ffmpeg
import os
import shutil
import warnings

# --- Parameters ---
# Target Bimodal Distribution P (Gaussian Mixture Model)
mu1, sigma1, w1 = -2.5, 0.8, 0.5
mu2, sigma2, w2 = 2.5, 0.8, 0.5
assert np.isclose(w1 + w2, 1.0) # Weights must sum to 1

# Plotting and Calculation Range
x_min = -7
x_max = 7
x_res = 300  # Resolution for plots and numerical integration
x = np.linspace(x_min, x_max, x_res)
dx = x[1] - x[0] # Step size for numerical integration

# Numerical stability epsilon
eps = 1e-10

# Animation parameters
N_FRAMES_TO_MIN = 100 # Frames to reach the minimum KL
N_OVERSHOOT_FRAMES = 30 # Frames to continue past the minimum
FPS = 20            # Frames per second for the output video
GIF_FILENAME = 'kl_fit_bimodal_animated_v4.gif'
TEMP_FRAME_DIR = 'kl_fit_anim_frames_v4' # Temporary directory

# Sigma constraint factor (allow sigma to change by +/- this factor * initial sigma)
SIGMA_CONSTRAINT_FACTOR = 0.3

# --- Define Target Distribution P ---
def p_pdf(x_vals):
    return w1 * norm.pdf(x_vals, loc=mu1, scale=sigma1) + \
           w2 * norm.pdf(x_vals, loc=mu2, scale=sigma2)

p_x = p_pdf(x)
p_x_safe = np.maximum(p_x, eps) # For stable calculations

# --- Define KL Divergence Objective Functions (Same as before) ---
def kl_pq_objective(params, p_pdf_vals, x_vals, dx_step, epsilon):
    mu, sigma = params
    if sigma <= epsilon: return np.inf # Sigma must be positive
    q_pdf_vals = norm.pdf(x_vals, loc=mu, scale=sigma)
    q_pdf_vals_safe = np.maximum(q_pdf_vals, epsilon)
    integrand = p_pdf_vals * np.log2(np.maximum(p_pdf_vals, epsilon) / q_pdf_vals_safe)
    kl_divergence = np.sum(integrand[p_pdf_vals > epsilon] * dx_step)
    return kl_divergence if np.isfinite(kl_divergence) else np.inf

def kl_qp_objective(params, p_pdf_vals_safe, x_vals, dx_step, epsilon):
    mu, sigma = params
    if sigma <= epsilon: return np.inf # Sigma must be positive
    q_pdf_vals = norm.pdf(x_vals, loc=mu, scale=sigma)
    q_pdf_vals_safe = np.maximum(q_pdf_vals, epsilon)
    integrand = q_pdf_vals * np.log2(q_pdf_vals_safe / p_pdf_vals_safe)
    kl_divergence = np.sum(integrand[q_pdf_vals > epsilon] * dx_step)
    return kl_divergence if np.isfinite(kl_divergence) else np.inf

# --- Optimize to Find Best Q for Each KL Direction (Same as before) ---
print("Optimizing for min D_KL(P || Q)...")
initial_guess_pq = [-5.0, 1.5]
mu0_pq_opt, sigma0_pq_opt = initial_guess_pq
sigma_min_pq = max(eps, sigma0_pq_opt * (1 - SIGMA_CONSTRAINT_FACTOR))
sigma_max_pq = sigma0_pq_opt * (1 + SIGMA_CONSTRAINT_FACTOR)
bounds_pq = [(None, None), (sigma_min_pq, sigma_max_pq)]
print(f"  Initial Guess: mu={mu0_pq_opt:.3f}, sigma={sigma0_pq_opt:.3f}")
print(f"  Sigma bounds: ({sigma_min_pq:.3f}, {sigma_max_pq:.3f})")
result_pq = minimize(kl_pq_objective, initial_guess_pq,
                     args=(p_x, x, dx, eps), method='L-BFGS-B',
                     bounds=bounds_pq, options={'disp': False})
if not result_pq.success: print(f"Warning: Optimization for D_KL(P || Q) failed: {result_pq.message}")
params_pq = result_pq.x
mu_pq, sigma_pq = params_pq
print(f"  Found Q_pq: mu={mu_pq:.3f}, sigma={sigma_pq:.3f}")

print("Optimizing for min D_KL(Q || P)...")
initial_guess_qp1_opt = [mu1, sigma1]
initial_guess_qp2_opt = [mu2, sigma2]
sigma0_qp1_opt = initial_guess_qp1_opt[1]
sigma_min_qp1 = max(eps, sigma0_qp1_opt * (1 - SIGMA_CONSTRAINT_FACTOR))
sigma_max_qp1 = sigma0_qp1_opt * (1 + SIGMA_CONSTRAINT_FACTOR)
bounds_qp1 = [(None, None), (sigma_min_qp1, sigma_max_qp1)]
sigma0_qp2_opt = initial_guess_qp2_opt[1]
sigma_min_qp2 = max(eps, sigma0_qp2_opt * (1 - SIGMA_CONSTRAINT_FACTOR))
sigma_max_qp2 = sigma0_qp2_opt * (1 + SIGMA_CONSTRAINT_FACTOR)
bounds_qp2 = [(None, None), (sigma_min_qp2, sigma_max_qp2)]
print(f"  Initial Guess 1: mu={initial_guess_qp1_opt[0]:.3f}, sigma={sigma0_qp1_opt:.3f}, bounds=({sigma_min_qp1:.3f}, {sigma_max_qp1:.3f})")
result_qp1 = minimize(kl_qp_objective, initial_guess_qp1_opt,
                      args=(p_x_safe, x, dx, eps), method='L-BFGS-B',
                      bounds=bounds_qp1, options={'disp': False})
print(f"  Initial Guess 2: mu={initial_guess_qp2_opt[0]:.3f}, sigma={sigma0_qp2_opt:.3f}, bounds=({sigma_min_qp2:.3f}, {sigma_max_qp2:.3f})")
result_qp2 = minimize(kl_qp_objective, initial_guess_qp2_opt,
                      args=(p_x_safe, x, dx, eps), method='L-BFGS-B',
                      bounds=bounds_qp2, options={'disp': False})
best_result_qp = None
qp1_success = result_qp1.success and np.isfinite(result_qp1.fun)
qp2_success = result_qp2.success and np.isfinite(result_qp2.fun)
if qp1_success and qp2_success: best_result_qp = result_qp1 if result_qp1.fun < result_qp2.fun else result_qp2
elif qp1_success: best_result_qp = result_qp1
elif qp2_success: best_result_qp = result_qp2
else:
    print(f"Warning: Optimization for D_KL(Q || P) failed or yielded non-finite result.")
    best_result_qp = type('obj', (object,), {'x': initial_guess_qp1_opt, 'fun': np.inf}) # Dummy
params_qp = best_result_qp.x
mu_qp, sigma_qp = params_qp
print(f"  Found Q_qp: mu={mu_qp:.3f}, sigma={sigma_qp:.3f}")

# Define initial parameters for the *ANIMATION*
mu0_pq_anim, sigma0_pq_anim = initial_guess_pq
mu0_qp_anim, sigma0_qp_anim = [0.0, 1.5] # Start Q||P anim from center

# --- Generate Frames ---
if os.path.exists(TEMP_FRAME_DIR):
    shutil.rmtree(TEMP_FRAME_DIR)
os.makedirs(TEMP_FRAME_DIR)

frame_files = []
total_frames = N_FRAMES_TO_MIN + N_OVERSHOOT_FRAMES

# Set up plot figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Determine consistent Y limit
max_y_list = [p_x.max()]
if result_pq.success: max_y_list.append(norm.pdf(x, loc=mu_pq, scale=sigma_pq).max())
if best_result_qp.success: max_y_list.append(norm.pdf(x, loc=mu_qp, scale=sigma_qp).max())
max_y_list.append(norm.pdf(x, loc=mu0_pq_anim, scale=sigma0_pq_anim).max())
max_y_list.append(norm.pdf(x, loc=mu0_qp_anim, scale=sigma0_qp_anim).max())
t_max = (total_frames - 1) / (N_FRAMES_TO_MIN - 1) if N_FRAMES_TO_MIN > 1 else 1.0
mu_overshoot_pq = (1 - t_max) * mu0_pq_anim + t_max * mu_pq
sigma_overshoot_pq = (1 - t_max) * sigma0_pq_anim + t_max * sigma_pq
if sigma_overshoot_pq > eps: max_y_list.append(norm.pdf(x, loc=mu_overshoot_pq, scale=sigma_overshoot_pq).max())
mu_overshoot_qp = (1 - t_max) * mu0_qp_anim + t_max * mu_qp
sigma_overshoot_qp = (1 - t_max) * sigma0_qp_anim + t_max * sigma_qp
if sigma_overshoot_qp > eps: max_y_list.append(norm.pdf(x, loc=mu_overshoot_qp, scale=sigma_overshoot_qp).max())
max_y = max(max_y_list) * 1.1


# Plot static P(x) on both axes
axes[0].plot(x, p_x, color='black', linestyle='-', linewidth=2, label='Target P(x)')
axes[1].plot(x, p_x, color='black', linestyle='-', linewidth=2, label='Target P(x)')

# Initialize lines for the animated Q distributions (moving dashed, optimal solid)
line_pq_moving, = axes[0].plot([], [], color='blue', linewidth=2, linestyle='--', label='Q(x) Anim')
# *** Initialize optimal line WITHOUT label ***
line_pq_optimal,= axes[0].plot([], [], color='blue', linewidth=2.5, linestyle='-', visible=False)
line_qp_moving, = axes[1].plot([], [], color='red', linewidth=2, linestyle='--', label='Q(x) Anim')
# *** Initialize optimal line WITHOUT label ***
line_qp_optimal,= axes[1].plot([], [], color='red', linewidth=2.5, linestyle='-', visible=False)

# Set titles and initial plot appearance
title_pq = axes[0].set_title("Minimize D_KL(P || Q)", fontsize=12)
title_qp = axes[1].set_title("Minimize D_KL(Q || P)", fontsize=12)

# Create initial legends (will only show Target and Anim lines)
legend_pq = axes[0].legend(loc='upper right')
legend_qp = axes[1].legend(loc='upper right')

for ax in axes:
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, max_y)
    ax.set_xlabel('x')
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel('Probability Density')

print(f"Generating {total_frames} frames...")
plot_error_count = 0
optimal_pq_pdf = norm.pdf(x, loc=mu_pq, scale=sigma_pq) # Pre-calculate optimal PDF
optimal_qp_pdf = norm.pdf(x, loc=mu_qp, scale=sigma_qp) # Pre-calculate optimal PDF
legend_updated_pq = False # Flag to update legend only once
legend_updated_qp = False # Flag to update legend only once

for i in range(total_frames):
    try:
        # Interpolation factor
        t_extended = i / (N_FRAMES_TO_MIN - 1) if N_FRAMES_TO_MIN > 1 else 1.0

        # Interpolate parameters for P || Q fit
        mu_current_pq = (1 - t_extended) * mu0_pq_anim + t_extended * mu_pq
        sigma_current_pq = (1 - t_extended) * sigma0_pq_anim + t_extended * sigma_pq
        if sigma_current_pq <= eps: sigma_current_pq = eps
        q_current_pq = norm.pdf(x, loc=mu_current_pq, scale=sigma_current_pq)
        current_kl_pq = kl_pq_objective([mu_current_pq, sigma_current_pq], p_x, x, dx, eps)

        # Interpolate parameters for Q || P fit
        mu_current_qp = (1 - t_extended) * mu0_qp_anim + t_extended * mu_qp
        sigma_current_qp = (1 - t_extended) * sigma0_qp_anim + t_extended * sigma_qp
        if sigma_current_qp <= eps: sigma_current_qp = eps
        q_current_qp = norm.pdf(x, loc=mu_current_qp, scale=sigma_current_qp)
        current_kl_qp = kl_qp_objective([mu_current_qp, sigma_current_qp], p_x_safe, x, dx, eps)

        # Update moving plot lines
        line_pq_moving.set_data(x, q_current_pq)
        line_qp_moving.set_data(x, q_current_qp)

        # Check if the minimum frame is reached or passed
        if i >= N_FRAMES_TO_MIN - 1:
             # Ensure the optimal line is plotted, visible, labeled, and legend updated ONCE
             if not legend_updated_pq:
                 line_pq_optimal.set_data(x, optimal_pq_pdf)
                 line_pq_optimal.set_label('Q(x) Optimal') # Set label now
                 line_pq_optimal.set_visible(True)
                 axes[0].legend(loc='upper right') # Redraw legend
                 legend_updated_pq = True # Prevent future updates

             if not legend_updated_qp:
                 line_qp_optimal.set_data(x, optimal_qp_pdf)
                 line_qp_optimal.set_label('Q(x) Optimal') # Set label now
                 line_qp_optimal.set_visible(True)
                 axes[1].legend(loc='upper right') # Redraw legend
                 legend_updated_qp = True # Prevent future updates

        # Update titles with current KL
        title_pq.set_text(f"Minimize D_KL(P || Q)\nKL={current_kl_pq:.3f}")
        title_qp.set_text(f"Minimize D_KL(Q || P)\nKL={current_kl_qp:.3f}")

        plt.tight_layout()

        # Save frame
        frame_filename = os.path.join(TEMP_FRAME_DIR, f'frame_{i:04d}.png')
        plt.savefig(frame_filename)
        frame_files.append(frame_filename)

    except Exception as e:
        plot_error_count += 1
        if plot_error_count <= 5: print(f"Error generating frame {i}: {e}")
        if plot_error_count == 6: print("Suppressing further frame generation errors...")


if 'fig' in locals() and plt.fignum_exists(fig.number):
     plt.close(fig)

if plot_error_count > 0: print(f"Warning: Encountered {plot_error_count} errors during frame generation.")

if not frame_files: print("Error: No frames were generated. Cannot compile GIF.")
else:
    print(f"Frames generated: {len(frame_files)}.")
    # --- Compile GIF ---
    print(f"Compiling GIF: {GIF_FILENAME}...")
    duration_sec = 1.0 / FPS if FPS > 0 else 0.1

    try:
        with imageio.get_writer(GIF_FILENAME, mode='I', duration=duration_sec, loop=0) as writer:
            print(f"Appending {len(frame_files)} frames...")
            for count, filename in enumerate(frame_files):
                try:
                    image = imageio.imread(filename)
                    writer.append_data(image)
                except FileNotFoundError: print(f"Warning: Frame file not found during GIF compilation: {filename}")
                except Exception as e: print(f"Warning: Error reading/appending frame {filename}: {e}")
        print("Writer finished.")

        if os.path.exists(GIF_FILENAME) and os.path.getsize(GIF_FILENAME) > 0:
            print("GIF compiled successfully.")
            print(f"Cleaning up temporary files in {TEMP_FRAME_DIR}...")
            try:
                shutil.rmtree(TEMP_FRAME_DIR)
                print("Cleanup done.")
            except Exception as e: print(f"Warning: Could not remove temporary directory {TEMP_FRAME_DIR}: {e}")
        else:
            print(f"Error: GIF file '{GIF_FILENAME}' was not created or is empty.")
            print(f"Temporary frame files kept in '{TEMP_FRAME_DIR}' for inspection.")

    except Exception as e:
        print(f"Error during GIF compilation: {e}")
        print(f"Temporary frame files kept in '{TEMP_FRAME_DIR}' for inspection.")

