import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

# --- Configuration ---
MATRIX_SHAPE = (4, 3)  # Rows, Columns for the original matrix
MIN_VAL, MAX_VAL = 1, 10  # Range for random matrix values

# Slower animation: Increase interval
INTERVAL_MS = 800       # Milliseconds between animation frames (e.g., 800ms for slower)
FPS_GIF = max(1, int(1000 / INTERVAL_MS)) # FPS for the output GIF, ensure at least 1
N_PAUSE_FRAMES = 5     # Number of frames for pause (N_PAUSE_FRAMES * INTERVAL_MS seconds)
ANIMATION_CYCLES = 2    # Number of times the full process will repeat in the GIF

# --- Matrix Initialization ---
np.random.seed(42) # For reproducibility
matrix_orig = np.random.randint(MIN_VAL, MAX_VAL + 1, size=MATRIX_SHAPE)
matrix_T = matrix_orig.T

R_orig, C_orig = matrix_orig.shape
R_T, C_T = matrix_T.shape

max_rows_to_process = max(R_orig, R_T)

# Total frames for one cycle of animation
TOTAL_FRAMES_PER_CYCLE = 1 + max_rows_to_process + 1 + N_PAUSE_FRAMES

# --- Plot Setup ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7)) # Adjusted figsize
# fig.suptitle("Matrix Row Mean Summation", fontsize=18) # Main title if needed

subplot_states = [{}, {}] # To store state for each subplot

def setup_subplot_display(ax, matrix_data, static_title_text, state_idx):
    """Initial setup for displaying one matrix and its dynamic info text."""
    ax.clear()
    ax.axis('off') # Hide axes ticks and spines

    rows, cols = matrix_data.shape
    state = subplot_states[state_idx]
    state['static_title'] = static_title_text # Store the base title

    # Create table
    cell_text_list = [[f"{val}" for val in row] for row in matrix_data]
    col_widths = [0.15 for _ in range(cols)] # Adjusted col_widths slightly
    table = ax.table(cellText=cell_text_list, loc='center', cellLoc='center',
                     colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 1.8) # Scale table size

    state['table'] = table
    state['matrix_data'] = matrix_data
    state['rows'] = rows
    state['cols'] = cols
    
    # Dynamic text area at the top (where a title would go)
    # Initial text is the static title
    dynamic_info_text_obj = ax.text(0.5, 1.03, static_title_text,
                                     transform=ax.transAxes,
                                     ha='center', va='bottom', fontsize=13, weight='bold', color='black')
    state['dynamic_info_text'] = dynamic_info_text_obj
    
    # Initialize dynamic state variables for calculations
    state['current_sum_val'] = 0.0
    state['calculated_means_list'] = [None] * rows


setup_subplot_display(axes[0], matrix_orig, "Original Matrix", 0)
setup_subplot_display(axes[1], matrix_T, "Transposed Matrix", 1)

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout slightly

# --- Animation Logic ---
def update_subplot_visuals(state_idx, current_frame_in_cycle):
    state = subplot_states[state_idx]
    table = state['table']
    matrix_data = state['matrix_data']
    rows = state['rows']
    cols = state['cols']
    dynamic_info_text = state['dynamic_info_text']
    static_title = state['static_title']

    # 1. Reset visual elements (cell colors)
    for r_idx in range(rows):
        for c_idx in range(cols):
            table[(r_idx, c_idx)].set_facecolor('white')

    # 2. Determine animation stage and update dynamic text and visuals
    processing_stage_active = False
    final_sum_stage_active = False
    # pause_stage_active = False # Not explicitly needed for text change, but for logic flow

    if current_frame_in_cycle == 0: # Initial display / Reset for new cycle
        dynamic_info_text.set_text(static_title)
        state['current_sum_val'] = 0.0
        state['calculated_means_list'] = [None] * rows
    elif 1 <= current_frame_in_cycle <= max_rows_to_process: # Row processing
        processing_stage_active = True
    elif current_frame_in_cycle == max_rows_to_process + 1: # Final sum
        final_sum_stage_active = True
    elif current_frame_in_cycle > max_rows_to_process + 1: # Pause
        # Keep displaying the final sum during pause
        final_sum_stage_active = True # Treat pause stage like final sum for display text

    # 3. Apply updates based on stage
    if processing_stage_active:
        current_overall_row_idx = current_frame_in_cycle - 1
        
        if current_overall_row_idx < rows: # If this matrix is actively processing this row
            # Highlight the current row
            for c_idx in range(cols):
                table[(current_overall_row_idx, c_idx)].set_facecolor(mcolors.to_rgba('skyblue', alpha=0.6))
            
            # Calculate and store mean for THIS specific row
            if state['calculated_means_list'][current_overall_row_idx] is None: # Calculate only once
                 state['calculated_means_list'][current_overall_row_idx] = np.mean(matrix_data[current_overall_row_idx, :])
            
            row_mean = state['calculated_means_list'][current_overall_row_idx]
            dynamic_info_text.set_text(f"Row {current_overall_row_idx} Mean: {row_mean:.2f}")
        else:
            # This matrix has fewer rows than max_rows_to_process, so its row processing is done for this stage.
            # Revert to static title or indicate completion for this matrix.
            all_means_calculated_for_this_matrix = all(m is not None for m in state['calculated_means_list'])
            if all_means_calculated_for_this_matrix:
                dynamic_info_text.set_text(static_title + "\n(Row Means Processed)")
            else:
                dynamic_info_text.set_text(static_title + "\n(Waiting...)")


    if final_sum_stage_active: # Includes pause stage for text display
        # Ensure all means are calculated (for the sum display)
        final_calculated_sum = 0.0
        all_means_available = True
        for r_idx in range(rows):
            if state['calculated_means_list'][r_idx] is None:
                # This can happen if a matrix had fewer rows than max_rows_to_process
                # and its processing stage for some rows was skipped.
                state['calculated_means_list'][r_idx] = np.mean(matrix_data[r_idx, :])
            
            if state['calculated_means_list'][r_idx] is not None:
                 final_calculated_sum += state['calculated_means_list'][r_idx]
            else:
                all_means_available = False # Should not happen if logic is correct

        if all_means_available:
            state['current_sum_val'] = final_calculated_sum
            dynamic_info_text.set_text(f"{static_title}\nSum of Row Means: {state['current_sum_val']:.2f}")
        else:
            dynamic_info_text.set_text(f"{static_title}\n(Calculating sum...)")


def animate(frame_num):
    current_frame_in_cycle = frame_num % TOTAL_FRAMES_PER_CYCLE
    
    update_subplot_visuals(0, current_frame_in_cycle) # Original Matrix
    update_subplot_visuals(1, current_frame_in_cycle) # Transposed Matrix
    
    artists_to_return = []
    for i in range(2):
        state = subplot_states[i]
        artists_to_return.append(state['dynamic_info_text'])
        # Table cells are modified via the table object; table itself isn't "new" each frame.
    return artists_to_return


# --- Create and Save Animation ---
print(f"Total frames per animation cycle: {TOTAL_FRAMES_PER_CYCLE}")
print(f"Animation interval: {INTERVAL_MS}ms, GIF FPS: {FPS_GIF}")
num_total_animation_frames = TOTAL_FRAMES_PER_CYCLE * ANIMATION_CYCLES

ani = animation.FuncAnimation(fig, animate, frames=num_total_animation_frames,
                              interval=INTERVAL_MS, blit=False, repeat=True)

gif_filename = 'matrix_row_means_sum_slower_titles.gif'
print(f"Attempting to save animation to {gif_filename}...")
try:
    ani.save(gif_filename, writer='pillow', fps=FPS_GIF)
    print(f"Successfully saved animation to {gif_filename}")
except Exception as e:
    print(f"Error saving animation: {e}")
    print("Ensure you have 'pillow' installed (pip install pillow).")

plt.close(fig)