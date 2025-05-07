import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import imageio
import os
import shutil

# --- Parameters ---
TRUE_MEAN = 5.0       # The actual mean of the data generating process
TRUE_STD_DEV = 1.5    # The actual standard deviation
NUM_SAMPLES = 50      # Number of data points to generate
MEAN_RANGE_MIN = 0.0  # Minimum mean value to test
MEAN_RANGE_MAX = 10.0 # Maximum mean value to test
NUM_FRAMES = 60       # Number of frames in the GIF (steps across the mean range)
GIF_FILENAME = 'mle_gaussian_mean.gif'
TEMP_FRAME_DIR = 'mle_frames' # Temporary directory to store frames

# --- Generate Sample Data ---
np.random.seed(42) # for reproducibility
data = np.random.normal(loc=TRUE_MEAN, scale=TRUE_STD_DEV, size=NUM_SAMPLES)

# --- Define Log-Likelihood Function ---
# Assumes fixed standard deviation (using the true one for simplicity here,
# though in reality it might also be estimated or assumed known).
def log_likelihood(mean_param, std_dev_param, data_points):
    """Calculates the total log-likelihood of the data given model parameters."""
    # Calculate log probability density for each data point
    log_pdfs = norm.logpdf(data_points, loc=mean_param, scale=std_dev_param)
    # Sum the log probabilities
    return np.sum(log_pdfs)

# --- Calculate Log-Likelihood across the parameter range ---
mean_values = np.linspace(MEAN_RANGE_MIN, MEAN_RANGE_MAX, NUM_FRAMES)
log_likelihoods = [log_likelihood(mu, TRUE_STD_DEV, data) for mu in mean_values]

# Find the Maximum Likelihood Estimate (MLE) for the mean numerically
mle_mean_index = np.argmax(log_likelihoods)
mle_mean = mean_values[mle_mean_index]
max_log_likelihood = log_likelihoods[mle_mean_index]

# --- Generate Frames for GIF ---
if os.path.exists(TEMP_FRAME_DIR):
    shutil.rmtree(TEMP_FRAME_DIR)
os.makedirs(TEMP_FRAME_DIR)

frame_files = []
x_pdf = np.linspace(min(data.min(), MEAN_RANGE_MIN - 3*TRUE_STD_DEV),
                    max(data.max(), MEAN_RANGE_MAX + 3*TRUE_STD_DEV), 200) # Range for plotting PDF

print(f"Generating {NUM_FRAMES} frames...")
for i, current_mean in enumerate(mean_values):
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    fig.suptitle('Maximum Likelihood Estimation (Mean of Normal Distribution)', fontsize=14)

    # --- Top Plot: Data and Current Candidate PDF ---
    axes[0].hist(data, bins='auto', density=True, alpha=0.6, label='Observed Data')
    current_pdf = norm.pdf(x_pdf, loc=current_mean, scale=TRUE_STD_DEV)
    axes[0].plot(x_pdf, current_pdf, 'r-', lw=2, label=f'Candidate PDF (μ={current_mean:.2f})')
    axes[0].set_title('Data vs. Candidate Model')
    axes[0].set_xlabel('Data Value')
    axes[0].set_ylabel('Density')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    # Keep x-axis consistent
    axes[0].set_xlim(x_pdf.min(), x_pdf.max())


    # --- Bottom Plot: Log-Likelihood Function ---
    axes[1].plot(mean_values[:i+1], log_likelihoods[:i+1], 'b-', label='Log-Likelihood Function') # Plot explored part
    axes[1].axvline(current_mean, color='red', linestyle='--', label=f'Current μ={current_mean:.2f}')
    axes[1].plot(mle_mean, max_log_likelihood, 'go', markersize=10, label=f'MLE μ={mle_mean:.2f}')
    axes[1].set_title('Log-Likelihood of Data given Mean (μ)')
    axes[1].set_xlabel('Candidate Mean (μ)')
    axes[1].set_ylabel('Log-Likelihood')
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)
    # Keep axes consistent
    axes[1].set_xlim(MEAN_RANGE_MIN, MEAN_RANGE_MAX)
    # Set y-lim based on calculated range, add some padding
    min_ll = np.min(log_likelihoods) if log_likelihoods else 0
    max_ll = np.max(log_likelihoods) if log_likelihoods else 1
    padding = (max_ll - min_ll) * 0.1
    axes[1].set_ylim(min_ll - padding, max_ll + padding)


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    # Save frame
    frame_filename = os.path.join(TEMP_FRAME_DIR, f'frame_{i:03d}.png')
    plt.savefig(frame_filename)
    frame_files.append(frame_filename)
    plt.close(fig) # Close plot to free memory

print("Frames generated.")

# --- Compile GIF ---
print(f"Compiling GIF: {GIF_FILENAME}...")
with imageio.get_writer(GIF_FILENAME, mode='I', duration=0.1) as writer: # duration is seconds per frame
    for filename in frame_files:
        image = imageio.imread(filename)
        writer.append_data(image)

print("GIF compiled.")

# --- Clean up temporary frames ---
print(f"Cleaning up temporary files in {TEMP_FRAME_DIR}...")
shutil.rmtree(TEMP_FRAME_DIR)
print("Done.")