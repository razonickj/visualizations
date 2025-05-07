import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

# --- Configuration ---
MATRIX_SHAPE = (4, 3)  # Rows, Columns for the original matrix
MIN_VAL, MAX_VAL = 1, 10  # Range for random matrix values
# Slower animation: Increase interval, decrease FPS for saving
INTERVAL_MS = 400       # Milliseconds between animation frames (e.g., 400ms for slower)
FPS_GIF = int(1000 / INTERVAL_MS) if INTERVAL_MS > 0 else 10 # FPS for the output GIF
N_PAUSE_FRAMES = 10     # Number of frames for pause (approx. N_PAUSE_FRAMES * INTERVAL_MS / 1000 seconds)
ANIMATION_CYCLES = 2    # Number of times the full process will repeat in the GIF

if FPS_GIF < 1: FPS_GIF = 1 # Ensure FPS is at least 1

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
fig, axes = plt.subplots(1, 2, figsize=(17, 8)) # Adjusted figsize for clarity
fig.suptitle("Matrix Row Mean Summation (Original vs Transpose)", fontsize=16)

subplot_states = [{}, {}]

def setup_subplot_display(ax, matrix_data, title_text, state_idx):
    ax.clear()
    ax.set_title(title_text, fontsize=14)
    ax.axis('off')

    rows, cols = matrix_data.shape
    state = subplot_states[state_idx]

    cell_text_list = [[f"{val}" for val in row] for row in matrix_data]
    col_widths = [0.12 for _ in range(cols)]
    table = ax.table(cellText=cell_text_list, loc='center left', cellLoc='center',
                     colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 1.8)

    state['table'] = table
    state['matrix_data'] = matrix_data
    state['rows'] = rows
    state['cols'] = cols
    
    # Single placeholder for the currently calculated row mean
    # Positioned to the right of the table, vertically centered somewhat
    table_total_width_ax_coords = sum(col_widths) * 1.1
    current_row_mean_text_obj = ax.text(table_total_width_ax_coords + 0.05, 0.5, # Vertically centered
                                     '', transform=ax.transAxes, va='center', ha='left', fontsize=12, color='blue')
    state['current_row_mean_text'] = current_row_mean_text_obj
    
    # Placeholder for sum of means text (below the table)
    sum_text_obj = ax.text(0.5, -0.15, '', transform=ax.transAxes, ha='center', va='top', fontsize=13, weight='bold') # Adjusted y for spacing
    state['sum_text'] = sum_text_obj
    
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
    current_row_mean_text = state['current_row_mean_text']
    sum_text = state['sum_text']

    # 1. Reset visual elements
    for r_idx in range(rows):
        for c_idx in range(cols):
            table[(r_idx, c_idx)].set_facecolor('white')
    current_row_mean_text.set_text('')
    current_row_mean_text.set_visible(False)
    sum_text.set_text('')
    sum_text.set_visible(False)

    # 2. Determine animation stage
    processing_stage_active = False
    final_sum_stage_active = False
    pause_stage_active = False

    if current_frame_in_cycle == 0: # Initial display / Reset
        state['current_sum_val'] = 0.0
        state['calculated_means_list'] = [None] * rows
    elif 1 <= current_frame_in_cycle <= max_rows_to_process: # Row processing
        processing_stage_active = True
    elif current_frame_in_cycle == max_rows_to_process + 1: # Final sum
        final_sum_stage_active = True
    elif current_frame_in_cycle > max_rows_to_process + 1: # Pause
        pause_stage_active = True

    # 3. Apply updates based on stage
    if processing_stage_active:
        current_overall_row_idx = current_frame_in_cycle - 1
        
        # If it's this matrix's turn to process a row
        if current_overall_row_idx < rows:
            # Highlight the current row
            for c_idx in range(cols):
                table[(current_overall_row_idx, c_idx)].set_facecolor(mcolors.to_rgba('skyblue', alpha=0.6))
            
            # Calculate and display mean for THIS specific row
            row_mean = np.mean(matrix_data[current_overall_row_idx, :])
            state['calculated_means_list'][current_overall_row_idx] = row_mean # Store for final sum
            
            current_row_mean_text.set_text(f"Row {current_overall_row_idx} Mean: {row_mean:.2f}")
            current_row_mean_text.set_visible(True)
        else:
            # This matrix has fewer rows than max_rows_to_process, so its row processing is done
            current_row_mean_text.set_text("Mean: ---") # Or keep it hidden
            current_row_mean_text.set_visible(True)
        
        # Update sum based on all means calculated so far *for this matrix*
        # This sum isn't displayed yet, but 'calculated_means_list' is populated
        # for the final sum stage.

    if final_sum_stage_active or pause_stage_active:
        # Ensure all means are calculated (for sum, even if not shown individually)
        final_calculated_sum = 0.0
        for r_idx in range(rows):
            if state['calculated_means_list'][r_idx] is None: 
                state['calculated_means_list'][r_idx] = np.mean(matrix_data[r_idx, :])
            final_calculated_sum += state['calculated_means_list'][r_idx]
        
        state['current_sum_val'] = final_calculated_sum
        sum_text.set_text(f"Sum of All Row Means:\n{state['current_sum_val']:.2f}")
        sum_text.set_visible(True)
        
        # Hide current row mean text during sum/pause
        current_row_mean_text.set_text('')
        current_row_mean_text.set_visible(False)


def animate(frame_num):
    current_frame_in_cycle = frame_num % TOTAL_FRAMES_PER_CYCLE
    
    update_subplot_visuals(0, current_frame_in_cycle) # Original Matrix
    update_subplot_visuals(1, current_frame_in_cycle) # Transposed Matrix
    
    artists_to_return = []
    for i in range(2):
        state = subplot_states[i]
        artists_to_return.append(state['current_row_mean_text'])
        artists_to_return.append(state['sum_text'])
        # Table cells are modified via the table object, so table itself doesn't need to be returned for blit=False
    return artists_to_return


# --- Create and Save Animation ---
print(f"Total frames per animation cycle: {TOTAL_FRAMES_PER_CYCLE}")
print(f"Animation interval: {INTERVAL_MS}ms, GIF FPS: {FPS_GIF}")
num_total_animation_frames = TOTAL_FRAMES_PER_CYCLE * ANIMATION_CYCLES

ani = animation.FuncAnimation(fig, animate, frames=num_total_animation_frames,
                              interval=INTERVAL_MS, blit=False, repeat=True)

gif_filename = 'matrix_row_means_sum_slow.gif'
print(f"Attempting to save animation to {gif_filename}...")
try:
    ani.save(gif_filename, writer='pillow', fps=FPS_GIF)
    print(f"Successfully saved animation to {gif_filename}")
except Exception as e:
    print(f"Error saving animation: {e}")
    print("Ensure you have 'pillow' installed (pip install pillow).")

plt.close(fig)