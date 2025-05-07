import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
import sys

# --- Configuration ---
# Use the same parameters as the filling animation
SEQ1 = "GATTACA"
SEQ2 = "GCATGCU"
MATCH_SCORE = 1
MISMATCH_SCORE = -1
GAP_PENALTY = -2
OUTPUT_FILENAME = "needleman_wunsch_traceback.gif"
INTERVAL = 600 # Milliseconds between frames

# --- Needleman-Wunsch Table Calculation (from previous step) ---
def calculate_nw_matrices(seq1, seq2, match, mismatch, gap):
    n = len(seq1)
    m = len(seq2)
    score_matrix = np.zeros((n + 1, m + 1), dtype=int)
    traceback_matrix = np.full((n + 1, m + 1), '', dtype=object)

    # Initialization
    for i in range(1, n + 1):
        score_matrix[i, 0] = score_matrix[i-1, 0] + gap
        traceback_matrix[i, 0] = '↑'
    for j in range(1, m + 1):
        score_matrix[0, j] = score_matrix[0, j-1] + gap
        traceback_matrix[0, j] = '←'
    traceback_matrix[0, 0] = ' '

    # Filling the matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_val = match if seq1[i-1] == seq2[j-1] else mismatch
            diag_score = score_matrix[i-1, j-1] + match_val
            up_score = score_matrix[i-1, j] + gap
            left_score = score_matrix[i, j-1] + gap

            scores = [diag_score, up_score, left_score]
            max_score = max(scores)
            score_matrix[i, j] = max_score

            # Determine traceback (simplified: picks one if scores are equal)
            # Preference: Diagonal > Up > Left
            if max_score == diag_score:
                traceback_matrix[i, j] = '↖'
            elif max_score == up_score:
                traceback_matrix[i, j] = '↑'
            else:
                traceback_matrix[i, j] = '←'

    return score_matrix, traceback_matrix, n, m # Return n and m

# --- Perform Traceback ---
def perform_traceback(seq1, seq2, traceback_matrix):
    n = len(seq1)
    m = len(seq2)
    i, j = n, m
    path_cells = []
    align1_rev = []
    align2_rev = []

    while i > 0 or j > 0:
        path_cells.append((i, j))
        direction = traceback_matrix[i, j]

        if direction == '↖':
            align1_rev.append(seq1[i-1])
            align2_rev.append(seq2[j-1])
            i -= 1
            j -= 1
        elif direction == '↑':
            align1_rev.append(seq1[i-1])
            align2_rev.append('-')
            i -= 1
        elif direction == '←':
            align1_rev.append('-')
            align2_rev.append(seq2[j-1])
            j -= 1
        else: # Reached (0,0) or boundary
             if i == 0 and j == 0:
                 break
             elif i == 0 and j > 0: # Only left possible
                  align1_rev.append('-')
                  align2_rev.append(seq2[j-1])
                  j -= 1
                  #path_cells.append((i, j)) # Optional: Add boundary cell if needed
             elif j == 0 and i > 0: # Only up possible
                  align1_rev.append(seq1[i-1])
                  align2_rev.append('-')
                  i -= 1
                  #path_cells.append((i, j)) # Optional: Add boundary cell if needed
             else:
                  print(f"Warning: Unexpected state or empty trace at ({i},{j}) with trace '{direction}'")
                  break


    path_cells.append((0, 0)) # Add the start cell
    path_cells.reverse() # Path from (0,0) to (n,m)
    align1 = "".join(align1_rev[::-1])
    align2 = "".join(align2_rev[::-1])

    return path_cells, align1, align2

# Calculate the matrices and perform traceback
# Now also get n and m back from calculate_nw_matrices
score_matrix, traceback_matrix, n, m = calculate_nw_matrices(
    SEQ1, SEQ2, MATCH_SCORE, MISMATCH_SCORE, GAP_PENALTY
)
traceback_path, alignment1, alignment2 = perform_traceback(SEQ1, SEQ2, traceback_matrix)

# --- Animation Setup ---
fig = plt.figure(figsize=(12, 6), constrained_layout=True)
gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[3, 2],
                       hspace=0.3, wspace=0.3)

ax_table = fig.add_subplot(gs[:, 0])       # Table on the left (spans both rows)
ax_alignment = fig.add_subplot(gs[:, 1]) # Alignment on the right (spans both rows)

# Store artists
cell_texts = {}
path_highlights = [] # Will store lines and rectangles
alignment_text_artist1 = None
alignment_text_artist2 = None
alignment_match_artist = None

# --- Plotting Functions ---
# MODIFIED: Accept n and m as arguments
def setup_plot(n, m):
    """Initial setup of the plot elements."""
    global alignment_text_artist1, alignment_text_artist2, alignment_match_artist

    # --- Main Table Axis (ax_table) ---
    # Now uses the passed n and m
    ax_table.set_xticks(np.arange(m + 1) - 0.5, minor=True)
    ax_table.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax_table.set_xticks(np.arange(m + 1))
    ax_table.set_yticks(np.arange(n + 1))
    ax_table.set_xticklabels([''] + list(SEQ2))
    ax_table.set_yticklabels([''] + list(SEQ1))
    ax_table.xaxis.tick_top()
    ax_table.xaxis.set_label_position('top')
    ax_table.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax_table.set_xlim(-0.5, m + 0.5)
    ax_table.set_ylim(n + 0.5, -0.5) # Inverted y-axis
    ax_table.tick_params(axis='both', which='major', length=0)
    ax_table.set_title("Needleman-Wunsch Traceback")

    # Fill table with final scores and arrows (also uses n, m implicitly via loops)
    for r in range(n + 1):
        for c in range(m + 1):
            score = score_matrix[r, c]
            trace = traceback_matrix[r, c]
            cell_texts[(r, c)] = ax_table.text(c, r, f"{score}\n{trace}",
                                               ha='center', va='center', fontsize=9, zorder=1)

    # --- Alignment Axis (ax_alignment) ---
    ax_alignment.set_axis_off()
    ax_alignment.set_title("Resulting Alignment")
    ax_alignment.set_xlim(0, 1)
    ax_alignment.set_ylim(0, 1)

    # Placeholder for alignment text (use monospace for alignment)
    alignment_text_artist1 = ax_alignment.text(0.1, 0.7, "", fontsize=12, family='monospace', va='center')
    alignment_match_artist = ax_alignment.text(0.1, 0.5, "", fontsize=12, family='monospace', va='center')
    alignment_text_artist2 = ax_alignment.text(0.1, 0.3, "", fontsize=12, family='monospace', va='center')


# --- Animation Update Function ---
total_frames = len(traceback_path) # Number of steps in the path

def update(frame):
    """Update function called for each frame of the traceback animation."""
    global path_highlights

    # Clear previous highlights (lines and rectangles)
    for artist in path_highlights:
        artist.remove()
    path_highlights.clear()

    # Determine current step in path
    current_cell_index = frame
    if current_cell_index >= total_frames:
         current_cell_index = total_frames - 1 # Stay on last frame


    # Highlight path segments up to the current frame
    path_segments = [] # Store (x, y) coordinates for drawing lines
    # Ensure we don't exceed path length if animation runs longer
    max_k = min(current_cell_index, total_frames - 1)
    for k in range(max_k + 1):
        r, c = traceback_path[k]
        path_segments.append((c, r)) # Use (column, row) for plotting

        # Highlight the cell itself
        rect = Rectangle((c - 0.5, r - 0.5), 1, 1, linewidth=1, edgecolor='none', facecolor='cyan', alpha=0.5, zorder=0)
        ax_table.add_patch(rect)
        path_highlights.append(rect)

    # Draw lines connecting the path segments
    if len(path_segments) > 1:
        path_x, path_y = zip(*path_segments)
        line, = ax_table.plot(path_x, path_y, color='red', linewidth=2.5, zorder=2)
        path_highlights.append(line)

    # Highlight the very current cell more prominently
    if current_cell_index < total_frames:
        curr_r, curr_c = traceback_path[current_cell_index]
        current_rect = Rectangle((curr_c - 0.5, curr_r - 0.5), 1, 1, linewidth=2, edgecolor='magenta', facecolor='none', zorder=3)
        ax_table.add_patch(current_rect)
        path_highlights.append(current_rect)

    # Update alignment text dynamically
    # Determine how much of the alignment to show
    path_index_rev = total_frames - 1 - current_cell_index
    if path_index_rev < 0: path_index_rev = 0 # Keep it valid for slicing

    # Simple slicing based on number of steps back from end
    # Note: This assumes each step in path corresponds to one alignment char/gap
    # More robust way is still to re-trace, but let's try slicing:
    current_len = len(alignment1) - path_index_rev

    partial_align1 = alignment1[:current_len]
    partial_align2 = alignment2[:current_len]


    # Build match string
    match_str = ""
    for k in range(len(partial_align1)):
        if k >= len(partial_align2): break # Should not happen with NW
        if partial_align1[k] == partial_align2[k]:
            match_str += "|"
        elif partial_align1[k] == '-' or partial_align2[k] == '-':
            match_str += " " # Gap
        else:
            match_str += "." # Mismatch

    alignment_text_artist1.set_text(f"Seq1: {partial_align1}")
    alignment_match_artist.set_text(f"      {match_str}")
    alignment_text_artist2.set_text(f"Seq2: {partial_align2}")

    # Return list of artists that were modified
    artists = (list(cell_texts.values()) + path_highlights +
               [alignment_text_artist1, alignment_match_artist, alignment_text_artist2])
    return artists

# --- Create and Save Animation ---
# MODIFIED: Pass n and m to setup_plot
setup_plot(n, m)

# Add a few extra frames at the end to display the final state
num_anim_frames = total_frames + 5

ani = animation.FuncAnimation(fig, update, frames=num_anim_frames,
                              interval=INTERVAL, blit=True, repeat=False)

print(f"Attempting to save traceback animation to {OUTPUT_FILENAME}...")
try:
    ani.save(OUTPUT_FILENAME, writer='pillow', fps=1000/INTERVAL)
    print(f"Animation saved successfully to {OUTPUT_FILENAME}")
except Exception as e:
    print(f"Error saving animation: {e}", file=sys.stderr)
    print("Ensure you have 'pillow' installed ('pip install pillow').", file=sys.stderr)

# To display the plot interactively (optional, remove if running non-interactively)
# plt.show()