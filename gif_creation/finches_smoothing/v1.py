import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

# --- Configuration ---
MATRIX_SHAPE = (4, 3)  # Rows, Columns for the original matrix
MIN_VAL, MAX_VAL = 1, 10  # Range for random matrix values
N_PAUSE_FRAMES = 3     # Number of frames for pause at the end of a cycle (approx 0.5s at 30fps)
FPS = 0.5
ANIMATION_CYCLES = 2    # Number of times the full process will repeat in the GIF

# --- Matrix Initialization ---
np.random.seed(42) # For reproducibility
matrix_orig = np.random.randint(MIN_VAL, MAX_VAL + 1, size=MATRIX_SHAPE)
matrix_T = matrix_orig.T

R_orig, C_orig = matrix_orig.shape
R_T, C_T = matrix_T.shape # R_T = C_orig, C_T = R_orig

max_rows_to_process = max(R_orig, R_T)

# Total frames for one cycle of animation
# 1 (initial display) + max_rows_to_process (row means) + 1 (final sums) + N_PAUSE_FRAMES
TOTAL_FRAMES_PER_CYCLE = 1 + max_rows_to_process + 1 + N_PAUSE_FRAMES

# --- Plot Setup ---
fig, axes = plt.subplots(1, 2, figsize=(16, 8)) # Adjusted figsize
fig.suptitle("Matrix Row Mean Summation (Original vs Transpose)", fontsize=16)

# Store artists and state that need updating for each subplot (original and transpose)
# Each element in the list will be a dictionary for one subplot
subplot_states = [{}, {}]

def setup_subplot_display(ax, matrix_data, title_text, state_idx):
    """Initial setup for displaying one matrix and its related text elements."""
    ax.clear()
    ax.set_title(title_text, fontsize=14)
    ax.axis('off') # Hide axes ticks and spines

    rows, cols = matrix_data.shape
    state = subplot_states[state_idx] # Get the dictionary for this subplot

    # Create table
    cell_text_list = [[f"{val}" for val in row] for row in matrix_data]
    # Use fixed column width to make text positioning easier later
    col_widths = [0.12 for _ in range(cols)]
    table = ax.table(cellText=cell_text_list, loc='center left', cellLoc='center',
                     colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 1.8) # Scale table size

    state['table'] = table
    state['matrix_data'] = matrix_data
    state['rows'] = rows
    state['cols'] = cols
    
    # Placeholders for row means text (position relative to table)
    row_means_texts_list = []
    # Estimate table width for text positioning (highly approximate)
    table_total_width_ax_coords = sum(col_widths) * 1.1 # A bit of guesswork
    
    for r in range(rows):
        # Y position is tricky with table scaling, adjust manually if needed
        # Table cells are indexed 0 to rows-1 from top.
        # ax.text y is 0 (bottom) to 1 (top) of axes.
        # Approximate y position of row r's center: (rows - r - 0.5) / rows
        # Use a fixed x offset from the right of the table visualization
        txt = ax.text(table_total_width_ax_coords + 0.05, 1.0 - (r + 0.5) * (1.8 / rows) * 0.9, # Y adjusted for scale
                      '', transform=ax.transAxes, va='center', ha='left', fontsize=12)
        row_means_texts_list.append(txt)
    state['row_means_texts'] = row_means_texts_list

    # Placeholder for sum of means text
    sum_text_obj = ax.text(0.5, -0.05, '', transform=ax.transAxes, ha='center', va='top', fontsize=13, weight='bold')
    state['sum_text'] = sum_text_obj
    
    # Initialize dynamic state variables
    state['current_sum_val'] = 0.0
    state['calculated_means_list'] = [None] * rows


setup_subplot_display(axes[0], matrix_orig, "Original Matrix", 0)
setup_subplot_display(axes[1], matrix_T, "Transposed Matrix", 1)

plt.tight_layout(rect=[0, 0.05, 1, 0.93])

# --- Animation Logic ---
def update_subplot_visuals(state_idx, current_frame_in_cycle):
    state = subplot_states[state_idx]
    table = state['table']
    matrix_data = state['matrix_data']
    rows = state['rows']
    cols = state['cols']
    row_means_texts = state['row_means_texts']
    sum_text = state['sum_text']

    # 1. Reset visual elements (colors, text visibility)
    for r_idx in range(rows):
        for c_idx in range(cols):
            table[(r_idx, c_idx)].set_facecolor('white')
        if row_means_texts[r_idx]:
             row_means_texts[r_idx].set_text('')
             row_means_texts[r_idx].set_visible(False)
    sum_text.set_text('')
    sum_text.set_visible(False)

    # 2. Determine current animation stage and update visuals accordingly
    processing_stage_active = False
    final_sum_stage_active = False
    pause_stage_active = False

    if current_frame_in_cycle == 0: # Initial display stage / Reset for new cycle
        state['current_sum_val'] = 0.0
        state['calculated_means_list'] = [None] * rows
    elif 1 <= current_frame_in_cycle <= max_rows_to_process: # Row processing stage
        processing_stage_active = True
    elif current_frame_in_cycle == max_rows_to_process + 1: # Final sum display stage
        final_sum_stage_active = True
    elif current_frame_in_cycle > max_rows_to_process + 1: # Pause stage
        pause_stage_active = True

    # 3. Apply updates based on stage
    if processing_stage_active:
        current_overall_row_being_processed = current_frame_in_cycle - 1
        
        # Calculate mean if this matrix's row is the current one
        if current_overall_row_being_processed < rows:
            row_mean = np.mean(matrix_data[current_overall_row_being_processed, :])
            state['calculated_means_list'][current_overall_row_being_processed] = row_mean
            # Highlight the row being actively processed now
            for c_idx in range(cols):
                table[(current_overall_row_being_processed, c_idx)].set_facecolor(mcolors.to_rgba('skyblue', alpha=0.6))

        # Display all means calculated so far in this cycle up to the overall current row
        current_temp_sum = 0.0
        for r_idx in range(rows):
            if r_idx <= current_overall_row_being_processed and state['calculated_means_list'][r_idx] is not None:
                mean_val = state['calculated_means_list'][r_idx]
                row_means_texts[r_idx].set_text(f"Mean: {mean_val:.2f}")
                row_means_texts[r_idx].set_visible(True)
                current_temp_sum += mean_val
            elif r_idx <= current_overall_row_being_processed and state['calculated_means_list'][r_idx] is None and r_idx < rows:
                # This row should have been processed for this matrix, but wasn't (e.g. matrix has fewer rows)
                # So, "fast-forward" its mean calculation for display
                row_mean = np.mean(matrix_data[r_idx, :])
                state['calculated_means_list'][r_idx] = row_mean
                mean_val = state['calculated_means_list'][r_idx]
                row_means_texts[r_idx].set_text(f"Mean: {mean_val:.2f}")
                row_means_texts[r_idx].set_visible(True)
                current_temp_sum += mean_val

        state['current_sum_val'] = current_temp_sum # Store the sum of currently displayed means


    if final_sum_stage_active or pause_stage_active:
        # Ensure all means are calculated and displayed
        final_calculated_sum = 0.0
        for r_idx in range(rows):
            if state['calculated_means_list'][r_idx] is None: # Should have been calculated by now
                state['calculated_means_list'][r_idx] = np.mean(matrix_data[r_idx, :])
            
            mean_val = state['calculated_means_list'][r_idx]
            row_means_texts[r_idx].set_text(f"Mean: {mean_val:.2f}")
            row_means_texts[r_idx].set_visible(True)
            final_calculated_sum += mean_val
        
        state['current_sum_val'] = final_calculated_sum
        sum_text.set_text(f"Sum of Means: {state['current_sum_val']:.2f}")
        sum_text.set_visible(True)


def animate(frame_num):
    current_frame_in_cycle = frame_num % TOTAL_FRAMES_PER_CYCLE
    
    update_subplot_visuals(0, current_frame_in_cycle) # Original Matrix
    update_subplot_visuals(1, current_frame_in_cycle) # Transposed Matrix
    
    # Collect all artists that might have been updated for blitting (if blit=True)
    # For blit=False, this list isn't strictly necessary but good practice.
    artists_to_return = []
    for i in range(2):
        state = subplot_states[i]
        # For table, returning the table object itself is usually enough.
        # Individual cell artists are managed by the table.
        # artists_to_return.append(state['table']) # Table object doesn't change, its cells do.
        artists_to_return.extend(state['row_means_texts'])
        artists_to_return.append(state['sum_text'])
    return artists_to_return


# --- Create and Save Animation ---
print(f"Total frames per animation cycle: {TOTAL_FRAMES_PER_CYCLE}")
num_total_animation_frames = TOTAL_FRAMES_PER_CYCLE * ANIMATION_CYCLES

ani = animation.FuncAnimation(fig, animate, frames=num_total_animation_frames,
                              interval=1000/FPS, blit=False, repeat=True)
# Using blit=False for robustness with table cell modifications.

gif_filename = 'matrix_row_means_sum_animation.gif'
print(f"Attempting to save animation to {gif_filename}...")
try:
    ani.save(gif_filename, writer='pillow', fps=FPS)
    print(f"Successfully saved animation to {gif_filename}")
except Exception as e:
    print(f"Error saving animation: {e}")
    print("Ensure you have 'pillow' installed (pip install pillow).")
    # For debugging, you might want to see the plot:
    # plt.show()

plt.close(fig)