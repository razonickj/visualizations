import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio
import os
import shutil

# --- Parameters ---
# Define Joint Probabilities P(X, Y)
# Rows represent X (X=0, X=1), Columns represent Y (Y=0, Y=1)
joint_probs = np.array([
    [0.1, 0.25], # P(X=0,Y=0), P(X=0,Y=1)
    [0.4, 0.25]  # P(X=1,Y=0), P(X=1,Y=1)
])

# --- Data Validation ---
if not np.isclose(joint_probs.sum(), 1.0):
    raise ValueError(f"Joint probabilities must sum to 1. Current sum: {joint_probs.sum()}")
if joint_probs.shape != (2, 2):
     raise ValueError("joint_probs must be a 2x2 array.")

# --- Calculate Marginals ---
# P(X) = [P(X=0), P(X=1)] (sum across columns for each row)
p_x = joint_probs.sum(axis=1)
# P(Y) = [P(Y=0), P(Y=1)] (sum across rows for each column)
p_y = joint_probs.sum(axis=0)

# --- Animation Parameters ---
FPS = 10
N_FRAMES_PER_PHASE = 30 # Includes pause at the end
total_frames = 6 * N_FRAMES_PER_PHASE # Intro, P(X=0), P(X=1), P(Y=0), P(Y=1), Outro
GIF_FILENAME = 'marginal_probabilities.gif'
TEMP_FRAME_DIR = 'marginal_prob_frames'

# --- Plotting Setup ---
cell_width = 0.35
cell_height = 0.35
gap = 0.05
x_coords = [0.5 - gap/2 - cell_width, 0.5 + gap/2] # Left edges of columns
y_coords = [0.5 + gap/2, 0.5 - gap/2 - cell_height] # Bottom edges of rows
base_colors = [['#add8e6', '#b0e0e6'], ['#b0c4de', '#e0ffff']] # Light blues/greys
highlight_color = '#fffacd' # Lemon chiffon
text_color = 'black'
result_color = 'darkgreen'
sum_color = 'darkslateblue'

# --- Generate Frames ---
if os.path.exists(TEMP_FRAME_DIR):
    shutil.rmtree(TEMP_FRAME_DIR)
os.makedirs(TEMP_FRAME_DIR)

frame_files = []
print(f"Generating {total_frames} frames...")
plot_error_count = 0

for i in range(total_frames):
    try:
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off') # Turn off axes

        phase = i // N_FRAMES_PER_PHASE
        frame_in_phase = i % N_FRAMES_PER_PHASE

        current_title = ""
        show_sum_text = ""

        # --- Draw Grid Cells and Base Labels ---
        rects = {} # Store rect objects for highlighting
        for r in range(2): # row index for X
            for c in range(2): # column index for Y
                x_pos = x_coords[c]
                y_pos = y_coords[r]
                prob = joint_probs[r, c]
                color = base_colors[r][c]

                # Store rect for potential highlighting
                rect = patches.Rectangle((x_pos, y_pos), cell_width, cell_height,
                                         linewidth=1.5, edgecolor='black',
                                         facecolor=color, alpha=0.8)
                ax.add_patch(rect)
                rects[(r, c)] = rect

                # Add joint probability text inside cell
                ax.text(x_pos + cell_width / 2, y_pos + cell_height / 2,
                        f"P(X={r}, Y={c})\n{prob:.2f}",
                        ha='center', va='center', fontsize=11, color=text_color)

        # Add Axis Labels outside grid
        ax.text(x_coords[0] + cell_width / 2, y_coords[0] + cell_height + gap, "Y=0", ha='center', va='bottom', fontsize=12)
        ax.text(x_coords[1] + cell_width / 2, y_coords[0] + cell_height + gap, "Y=1", ha='center', va='bottom', fontsize=12)
        ax.text(x_coords[0] - gap, y_coords[0] + cell_height / 2, "X=0", ha='right', va='center', fontsize=12)
        ax.text(x_coords[0] - gap, y_coords[1] + cell_height / 2, "X=1", ha='right', va='center', fontsize=12)

        # --- Phase Logic ---
        if phase == 0: # Intro: Show Joint Probabilities
            current_title = "Joint Probability Distribution P(X, Y)"

        elif phase == 1: # Calculate P(X=0)
            current_title = "Calculating Marginal P(X=0)"
            # Highlight row X=0
            rects[(0, 0)].set_facecolor(highlight_color)
            rects[(0, 1)].set_facecolor(highlight_color)
            # Show sum text - appears fully at end of phase
            if frame_in_phase >= N_FRAMES_PER_PHASE // 2:
                 show_sum_text = (f"P(X=0) = P(X=0,Y=0) + P(X=0,Y=1)\n"
                                  f"P(X=0) = {joint_probs[0,0]:.2f} + {joint_probs[0,1]:.2f} = {p_x[0]:.2f}")

        elif phase == 2: # Calculate P(X=1)
            current_title = "Calculating Marginal P(X=1)"
            rects[(1, 0)].set_facecolor(highlight_color)
            rects[(1, 1)].set_facecolor(highlight_color)
            if frame_in_phase >= N_FRAMES_PER_PHASE // 2:
                show_sum_text = (f"P(X=1) = P(X=1,Y=0) + P(X=1,Y=1)\n"
                                 f"P(X=1) = {joint_probs[1,0]:.2f} + {joint_probs[1,1]:.2f} = {p_x[1]:.2f}")

        elif phase == 3: # Calculate P(Y=0)
            current_title = "Calculating Marginal P(Y=0)"
            rects[(0, 0)].set_facecolor(highlight_color)
            rects[(1, 0)].set_facecolor(highlight_color)
            if frame_in_phase >= N_FRAMES_PER_PHASE // 2:
                show_sum_text = (f"P(Y=0) = P(X=0,Y=0) + P(X=1,Y=0)\n"
                                 f"P(Y=0) = {joint_probs[0,0]:.2f} + {joint_probs[1,0]:.2f} = {p_y[0]:.2f}")

        elif phase == 4: # Calculate P(Y=1)
            current_title = "Calculating Marginal P(Y=1)"
            rects[(0, 1)].set_facecolor(highlight_color)
            rects[(1, 1)].set_facecolor(highlight_color)
            if frame_in_phase >= N_FRAMES_PER_PHASE // 2:
                show_sum_text = (f"P(Y=1) = P(X=0,Y=1) + P(X=1,Y=1)\n"
                                 f"P(Y=1) = {joint_probs[0,1]:.2f} + {joint_probs[1,1]:.2f} = {p_y[1]:.2f}")

        elif phase == 5: # Outro: Show Marginals
            current_title = "Joint and Marginal Probabilities"
            # Add marginals to the edges
            # P(X)
            ax.text(x_coords[1] + cell_width + gap, y_coords[0] + cell_height / 2,
                    f"P(X=0)\n= {p_x[0]:.2f}", ha='left', va='center', fontsize=12, color=result_color)
            ax.text(x_coords[1] + cell_width + gap, y_coords[1] + cell_height / 2,
                    f"P(X=1)\n= {p_x[1]:.2f}", ha='left', va='center', fontsize=12, color=result_color)
            # P(Y)
            ax.text(x_coords[0] + cell_width / 2, y_coords[1] - gap,
                    f"P(Y=0)\n= {p_y[0]:.2f}", ha='center', va='top', fontsize=12, color=result_color)
            ax.text(x_coords[1] + cell_width / 2, y_coords[1] - gap,
                    f"P(Y=1)\n= {p_y[1]:.2f}", ha='center', va='top', fontsize=12, color=result_color)


        # Add Title and Sum Text
        fig.suptitle(current_title, fontsize=16)
        if show_sum_text:
            ax.text(0.5, 0.05, show_sum_text, ha='center', va='bottom', fontsize=13,
                    color=sum_color, bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))

        plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout

        # Save frame
        frame_filename = os.path.join(TEMP_FRAME_DIR, f'frame_{i:04d}.png')
        plt.savefig(frame_filename)
        frame_files.append(frame_filename) # Append only if savefig succeeds
        plt.close(fig) # Close plot to free memory

    except Exception as e:
        plot_error_count += 1
        if plot_error_count <= 5: print(f"Error generating frame {i}: {e}")
        if plot_error_count == 6: print("Suppressing further frame generation errors...")


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

