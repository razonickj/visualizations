import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
import imageio
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
N_FRAMES_PER_PHASE = 75 # Frames for each display phase
FPS = 15                # Frames per second for the output video
GIF_FILENAME = 'kl_fit_bimodal.gif'
TEMP_FRAME_DIR = 'kl_fit_frames' # Temporary directory

# --- Define Target Distribution P ---
def p_pdf(x_vals):
    return w1 * norm.pdf(x_vals, loc=mu1, scale=sigma1) + \
           w2 * norm.pdf(x_vals, loc=mu2, scale=sigma2)

p_x = p_pdf(x)
p_x_safe = np.maximum(p_x, eps) # For stable calculations

# --- Define KL Divergence Objective Functions for Optimization ---

# Objective Function for minimizing D_KL(P || Q)
def kl_pq_objective(params, p_pdf_vals, x_vals, dx_step, epsilon):
    """Calculates D_KL(P || Q) for Q defined by params=[mu, sigma]."""
    mu, sigma = params
    # Constraint: sigma must be positive
    if sigma <= epsilon:
        return np.inf # Penalize invalid sigma heavily

    q_pdf_vals = norm.pdf(x_vals, loc=mu, scale=sigma)
    q_pdf_vals_safe = np.maximum(q_pdf_vals, epsilon)

    # Integrand: P(x) * log2(P(x) / Q(x))
    integrand = p_pdf_vals * np.log2(p_pdf_vals / q_pdf_vals_safe) # Already used p_x_safe implicitly via input

    # Numerical integration (Trapezoidal rule might be slightly better, but sum*dx is often sufficient)
    kl_divergence = np.sum(integrand[p_pdf_vals > epsilon] * dx_step) # Only sum where P > 0

    # Handle potential NaN/Inf results from optimization steps
    if not np.isfinite(kl_divergence):
        return np.inf

    return kl_divergence

# Objective Function for minimizing D_KL(Q || P)
def kl_qp_objective(params, p_pdf_vals_safe, x_vals, dx_step, epsilon):
    """Calculates D_KL(Q || P) for Q defined by params=[mu, sigma]."""
    mu, sigma = params
    # Constraint: sigma must be positive
    if sigma <= epsilon:
        return np.inf

    q_pdf_vals = norm.pdf(x_vals, loc=mu, scale=sigma)
    q_pdf_vals_safe = np.maximum(q_pdf_vals, epsilon)

    # Integrand: Q(x) * log2(Q(x) / P(x))
    integrand = q_pdf_vals * np.log2(q_pdf_vals_safe / p_pdf_vals_safe)

    # Numerical integration
    kl_divergence = np.sum(integrand[q_pdf_vals > epsilon] * dx_step) # Only sum where Q > 0

    if not np.isfinite(kl_divergence):
        return np.inf

    return kl_divergence

# --- Optimize to Find Best Q for Each KL Direction ---
print("Optimizing for min D_KL(P || Q)...")
# Initial guess for Q parameters [mu, sigma]
initial_guess = [0.0, 2.0]
# Bounds for parameters: mu can be anything (None), sigma must be positive (eps, None)
bounds = [(None, None), (eps, None)]

# Minimize D_KL(P || Q)
result_pq = minimize(kl_pq_objective, initial_guess,
                     args=(p_x_safe, x, dx, eps), # Pass fixed arguments
                     method='L-BFGS-B', # Method that handles bounds
                     bounds=bounds,
                     options={'disp': False}) # Turn off verbose output

if not result_pq.success:
    print(f"Warning: Optimization for D_KL(P || Q) failed: {result_pq.message}")
params_pq = result_pq.x
mu_pq, sigma_pq = params_pq
q_pq_x = norm.pdf(x, loc=mu_pq, scale=sigma_pq)
kl_pq_final = result_pq.fun
print(f"  Found Q_pq: mu={mu_pq:.3f}, sigma={sigma_pq:.3f}, KL={kl_pq_final:.3f}")


print("Optimizing for min D_KL(Q || P)...")
# Minimize D_KL(Q || P)
# Try two initial guesses, one near each mode of P, because reverse KL is mode-seeking
initial_guess1 = [mu1, sigma1] # Start near mode 1
result_qp1 = minimize(kl_qp_objective, initial_guess1,
                      args=(p_x_safe, x, dx, eps),
                      method='L-BFGS-B',
                      bounds=bounds,
                      options={'disp': False})

initial_guess2 = [mu2, sigma2] # Start near mode 2
result_qp2 = minimize(kl_qp_objective, initial_guess2,
                      args=(p_x_safe, x, dx, eps),
                      method='L-BFGS-B',
                      bounds=bounds,
                      options={'disp': False})

# Choose the result with the lower KL divergence
if result_qp1.success and result_qp2.success:
    result_qp = result_qp1 if result_qp1.fun < result_qp2.fun else result_qp2
elif result_qp1.success:
    result_qp = result_qp1
elif result_qp2.success:
    result_qp = result_qp2
else:
    print(f"Warning: Optimization for D_KL(Q || P) failed for both initial guesses.")
    # Fallback: Use a reasonable default if optimization fails completely
    result_qp = type('obj', (object,), {'x': initial_guess1, 'fun': np.inf}) # Dummy object

params_qp = result_qp.x
mu_qp, sigma_qp = params_qp
q_qp_x = norm.pdf(x, loc=mu_qp, scale=sigma_qp)
kl_qp_final = result_qp.fun if np.isfinite(result_qp.fun) else kl_qp_objective(params_qp, p_x_safe, x, dx, eps) # Recalculate if Inf
print(f"  Found Q_qp: mu={mu_qp:.3f}, sigma={sigma_qp:.3f}, KL={kl_qp_final:.3f}")

# --- Generate Frames ---
if os.path.exists(TEMP_FRAME_DIR):
    shutil.rmtree(TEMP_FRAME_DIR)
os.makedirs(TEMP_FRAME_DIR)

frame_files = []
total_frames = 3 * N_FRAMES_PER_PHASE # Target P, Fit P||Q, Fit Q||P

print(f"Generating {total_frames} frames...")
for i in range(total_frames):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, max(p_x.max(), q_pq_x.max(), q_qp_x.max()) * 1.1)
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density')
    ax.grid(True, alpha=0.3)

    phase_title = ""
    description = ""

    # Determine phase
    phase_idx = i // N_FRAMES_PER_PHASE

    # Plot Target P always
    ax.plot(x, p_x, color='black', linestyle='-', linewidth=2.5, label='Target P(x) (Bimodal)')

    if phase_idx == 0: # Phase 0: Show Target P
        phase_title = "Target Distribution P(x)"
        description = "Goal: Fit a single Gaussian Q(x | μ, σ) to this bimodal P(x)."

    elif phase_idx == 1: # Phase 1: Show Q minimizing D_KL(P || Q)
        phase_title = "Fit via Minimizing D_KL(P || Q)"
        description = ("Minimizing D_KL(P || Q) penalizes Q for being too low where P is high.\n"
                       "Result tends to 'cover' P, averaging modes.\n"
                       f"Q(μ={mu_pq:.2f}, σ={sigma_pq:.2f})")
        # Plot Q_pq
        ax.plot(x, q_pq_x, color='blue', linewidth=2.5, linestyle='--',
                label=f'Q_pq (min P||Q)')
        ax.fill_between(x, q_pq_x, color='blue', alpha=0.2)

    elif phase_idx == 2: # Phase 2: Show Q minimizing D_KL(Q || P)
        phase_title = "Fit via Minimizing D_KL(Q || P)"
        description = ("Minimizing D_KL(Q || P) penalizes Q for being high where P is low.\n"
                       "Result tends to 'seek' one mode of P.\n"
                       f"Q(μ={mu_qp:.2f}, σ={sigma_qp:.2f})")
        # Plot Q_qp
        ax.plot(x, q_qp_x, color='red', linewidth=2.5, linestyle='--',
                label=f'Q_qp (min Q||P)')
        ax.fill_between(x, q_qp_x, color='red', alpha=0.2)


    # Add titles and text
    ax.set_title(phase_title, fontsize=14)
    ax.text(0.02, 0.95, description, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.8))
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