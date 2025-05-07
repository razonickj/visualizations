import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import shutil

# --- Parameters ---
NUM_OUTCOMES = 4
NUM_FRAMES_TOTAL = 120 # Total frames for the GIF (must be even)
GIF_FILENAME = 'shannon_entropy_visualization.gif'
TEMP_FRAME_DIR = 'entropy_frames' # Temporary directory
FPS = 15 # Frames per second

# --- Define Probability Distributions ---
# Start: Low entropy (one outcome likely)
p_start = np.array([0.97, 0.01, 0.01, 0.01])
# Middle: Max entropy (uniform distribution)
p_uniform = np.ones(NUM_OUTCOMES) / NUM_OUTCOMES
# End: Low entropy (different outcome likely)
p_end = np.array([0.01, 0.01, 0.97, 0.01])

# --- Define Shannon Entropy Function ---
def shannon_entropy(probabilities):
    """Calculates Shannon Entropy in bits."""
    # Filter out zero probabilities to avoid log2(0)
    non_zero_probs = probabilities[probabilities > 0]
    # Calculate entropy using base 2 logarithm
    entropy = -np.sum(non_zero_probs * np.log2(non_zero_probs))
    return entropy

# --- Generate Frames ---
if os.path.exists(TEMP_FRAME_DIR):
    shutil.rmtree(TEMP_FRAME_DIR)
os.makedirs(TEMP_FRAME_DIR)

frame_files = []
num_frames_half = NUM_FRAMES_TOTAL // 2

print(f"Generating {NUM_FRAMES_TOTAL} frames...")
for i in range(NUM_FRAMES_TOTAL):
    # Interpolate probabilities
    if i < num_frames_half:
        # Interpolate from start to uniform
        t = i / (num_frames_half - 1) if num_frames_half > 1 else 1.0
        current_probs_unnormalized = (1 - t) * p_start + t * p_uniform
    else:
        # Interpolate from uniform to end
        t = (i - num_frames_half) / (num_frames_half - 1) if num_frames_half > 1 else 1.0
        current_probs_unnormalized = (1 - t) * p_uniform + t * p_end

    # Normalize probabilities to ensure they sum to 1
    current_probs = current_probs_unnormalized / np.sum(current_probs_unnormalized)

    # Calculate current entropy
    current_entropy = shannon_entropy(current_probs)
    max_entropy = np.log2(NUM_OUTCOMES) # Theoretical maximum for N outcomes

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(8, 6))

    # Bar chart for probabilities
    bar_labels = [f'Outcome {j+1}' for j in range(NUM_OUTCOMES)]
    bars = ax.bar(bar_labels, current_probs, color='skyblue', edgecolor='black')

    # Set plot limits and labels
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Probability P(x)')
    ax.set_xlabel('Possible Outcomes (x)')
    ax.set_title('Shannon Entropy Visualization')

    # Display Entropy value
    entropy_text = f'H(X) = {current_entropy:.3f} bits'
    ax.text(0.5, 0.95, entropy_text, ha='center', va='top',
            transform=ax.transAxes, fontsize=14, bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.8))

    # Add Uncertainty indicator text
    uncertainty_level = ""
    if current_entropy < max_entropy * 0.3:
        uncertainty_level = "Low Uncertainty / High Predictability"
    elif current_entropy > max_entropy * 0.85:
         uncertainty_level = "High Uncertainty / Low Predictability"
    else:
        uncertainty_level = "Moderate Uncertainty"

    ax.text(0.5, 0.88, uncertainty_level, ha='center', va='top',
            transform=ax.transAxes, fontsize=12, style='italic', color='gray')


    # Add probability values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}',
                ha='center', va='bottom')

    plt.tight_layout()

    # Save frame
    frame_filename = os.path.join(TEMP_FRAME_DIR, f'frame_{i:04d}.png')
    plt.savefig(frame_filename)
    frame_files.append(frame_filename)
    plt.close(fig) # Close plot to free memory

print("Frames generated.")

# --- Compile GIF ---
print(f"Compiling GIF: {GIF_FILENAME}...")
with imageio.get_writer(GIF_FILENAME, mode='I', duration=1000/FPS, loop=0) as writer: # duration is ms per frame, loop=0 means infinite loop
    for filename in frame_files:
        image = imageio.imread(filename)
        writer.append_data(image)

print("GIF compiled.")

# --- Clean up temporary frames ---
print(f"Cleaning up temporary files in {TEMP_FRAME_DIR}...")
shutil.rmtree(TEMP_FRAME_DIR)
print("Done.")