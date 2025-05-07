import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
import sys

# --- Configuration ---
SEQ1 = "GATTACA"
SEQ2 = "GCATGCU"
MATCH_SCORE = 1
MISMATCH_SCORE = -1
GAP_PENALTY = -2
OUTPUT_FILENAME = "needleman_wunsch_traceback_reverse.gif" # Changed filename
INTERVAL = 600 # Milliseconds between frames

# --- Needleman-Wunsch Table Calculation ---
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
            if max_score == diag_score: traceback_matrix[i, j] = '↖'
            elif max_score == up_score: traceback_matrix[i, j] = '↑'
            else: traceback_matrix[i, j] = '←'

    return score_matrix, traceback_matrix, n, m

# --- Perform Traceback ---
def perform_traceback(seq1, seq2, traceback_matrix):
    n = len(seq1)
    m = len(seq2)
    i, j = n, m
    # path_cells will store the path from (n, m) back to (0, 0)
    path_cells_reverse = []
    align1_rev = []
    align2_rev = []

    while i > 0 or j > 0:
        path_cells_reverse.append((i, j)) # Store current cell
        direction = traceback_matrix[i, j]

        if direction == '↖':
            align1_rev.append(seq1[i-1])
            align2_rev.append(seq2[j-1])
            i -= 1; j -= 1
        elif direction == '↑':
            align1_rev.append(seq1[i-1])
            align2_rev.append('-') # GAP character
            i -= 1
        elif direction == '←':
            align1_rev.append('-') # GAP character
            align2_rev.append(seq2[j-1])
            j -= 1
        else: # Boundary or error
            if i == 0 and j > 0: direction = '←' # Force left if on top edge
            elif j == 0 and i > 0: direction = '↑' # Force up if on left edge
            else:
                 print(f"Warning: Reached ({i},{j}) with unexpected trace '{direction}'. Stopping traceback.")
                 break # Stop if unexpected state

            # Handle boundary moves if forced direction
            if direction == '←':
                 align1_rev.append('-')
                 align2_rev.append(seq2[j-1])
                 j -= 1
            elif direction == '↑':
                 align1_rev.append(seq1[i-1])
                 align2_rev.append('-')
                 i -= 1

    path_cells_reverse.append((0, 0)) # Add the final (0,0) cell

    # The final alignments are built reversed, so flip them
    align1 = "".join(align1_rev[::-1])
    align2 = "".join(align2_rev[::-1])

    # path_cells_reverse is ALREADY in the order (n,m) -> (0,0)
    return path_cells_reverse, align1, align2

# Calculate the matrices and perform traceback
score_matrix, traceback_matrix, n, m = calculate_nw_matrices(
    SEQ1, SEQ2, MATCH_SCORE, MISMATCH_SCORE, GAP_PENALTY
)
# traceback_path now goes from (n,m) back to (0,0)
traceback_path_rev, alignment1, alignment2 = perform_traceback(SEQ1, SEQ2, traceback_matrix)

# --- Animation Setup ---
fig = plt.figure(figsize=(12, 6), constrained_layout=True)
gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[3, 2],
                       hspace=0.3, wspace=0.3)

ax_table = fig.add_subplot(gs[:, 0])
ax_alignment = fig.add_subplot(gs[:, 1])

# Store artists
cell_texts = {}
path_highlights = []
alignment_text_artist1 = None
alignment_text_artist2 = None
alignment_match_artist = None

# --- Plotting Functions ---
def setup_plot(n, m):
    """Initial setup of the plot elements."""
    global alignment_text_artist1, alignment_text_artist2, alignment_match_artist

    # --- Main Table Axis (ax_table) ---
    ax_table.set_xticks(np.arange(m + 1) - 0.5, minor=True); ax_table.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax_table.set_xticks(np.arange(m + 1)); ax_table.set_yticks(np.arange(n + 1))
    ax_table.set_xticklabels([''] + list(SEQ2)); ax_table.set_yticklabels([''] + list(SEQ1))
    ax_table.xaxis.tick_top(); ax_table.xaxis.set_label_position('top')
    ax_table.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax_table.set_xlim(-0.5, m + 0.5); ax_table.set_ylim(n + 0.5, -0.5)
    ax_table.tick_params(axis='both', which='major', length=0)
    ax_table.set_title("Needleman-Wunsch Traceback (Bottom-Right to Top-Left)")

    # Fill table with final scores and arrows
    for r in range(n + 1):
        for c in range(m + 1):
            score = score_matrix[r, c]; trace = traceback_matrix[r, c]
            cell_texts[(r, c)] = ax_table.text(c, r, f"{score}\n{trace}", ha='center', va='center', fontsize=9, zorder=1)

    # --- Alignment Axis (ax_alignment) ---
    ax_alignment.set_axis_off(); ax_alignment.set_title("Alignment Being Built (Reversed)")
    ax_alignment.set_xlim(0, 1); ax_alignment.set_ylim(0, 1)
    alignment_text_artist1 = ax_alignment.text(0.1, 0.7, "", fontsize=12, family='monospace', va='center')
    alignment_match_artist = ax_alignment.text(0.1, 0.5, "", fontsize=12, family='monospace', va='center')
    alignment_text_artist2 = ax_alignment.text(0.1, 0.3, "", fontsize=12, family='monospace', va='center')


# --- Animation Update Function ---
# total_frames is the number of steps *from* (n,m)
total_frames = len(traceback_path_rev)

def update(frame):
    """Update function called for each frame of the traceback animation."""
    global path_highlights

    for artist in path_highlights: artist.remove()
    path_highlights.clear()

    # Current index in the path (starts at 0, which is cell (n,m))
    current_path_index = frame
    if current_path_index >= total_frames:
        current_path_index = total_frames - 1 # Hold on last frame

    # Highlight path segments from frame 0 up to current frame
    path_segments_coords = [] # Store (col, row) for plotting
    # Iterate through the path taken SO FAR in this animation
    max_k = min(current_path_index, total_frames - 1)
    for k in range(max_k + 1):
        # traceback_path_rev[k] gives cell (r, c) for step k
        r, c = traceback_path_rev[k]
        path_segments_coords.append((c, r)) # Use (col, row) for plotting

        # Highlight the cells visited so far
        rect_color = 'lightcoral' if k==0 else 'cyan' # Start cell different color
        rect = Rectangle((c - 0.5, r - 0.5), 1, 1, linewidth=1, edgecolor='none', facecolor=rect_color, alpha=0.5, zorder=0)
        ax_table.add_patch(rect)
        path_highlights.append(rect)

    # Draw lines connecting the path segments visited so far
    if len(path_segments_coords) > 1:
        path_x, path_y = zip(*path_segments_coords)
        line, = ax_table.plot(path_x, path_y, color='red', linewidth=2.5, zorder=2)
        path_highlights.append(line)

    # Highlight the very current cell being processed in this frame
    if current_path_index < total_frames:
        curr_r, curr_c = traceback_path_rev[current_path_index]
        current_rect = Rectangle((curr_c - 0.5, curr_r - 0.5), 1, 1, linewidth=2, edgecolor='magenta', facecolor='none', zorder=3)
        ax_table.add_patch(current_rect)
        path_highlights.append(current_rect)


    # --- Update alignment text dynamically ---
    # frame 0 = 1st step back = last char of alignment
    # frame N = (N+1)th step back = last (N+1) chars
    num_chars_to_show = min(frame + 1, len(alignment1))

    # We want the last 'num_chars_to_show' characters from the final alignment
    partial_align1 = alignment1[-num_chars_to_show:]
    partial_align2 = alignment2[-num_chars_to_show:]

    # Build match string for the partial alignment
    match_str = ""
    for k in range(len(partial_align1)):
        # Ensure comparison is valid (should be for NW)
        if k >= len(partial_align2): break
        if partial_align1[k] == partial_align2[k]: match_str += "|"
        elif partial_align1[k] == '-' or partial_align2[k] == '-': match_str += " " # Gap uses '-'
        else: match_str += "."
    # Displaying the alignment as it's found (which is reverse of conventional display)
    # To make it look like it's building left-to-right use the reversed temp strings
    align1_building = "".join(reversed(partial_align1))
    align2_building = "".join(reversed(partial_align2))
    match_building = "".join(reversed(match_str))


    # Update the text artists - show the alignment being built (reversed order)
    alignment_text_artist1.set_text(f"Aln1: {align1_building}")
    alignment_match_artist.set_text(f"      {match_building}")
    alignment_text_artist2.set_text(f"Aln2: {align2_building}")

    # Return list of artists that were modified
    artists = (list(cell_texts.values()) + path_highlights +
               [alignment_text_artist1, alignment_match_artist, alignment_text_artist2])
    return artists

# --- Create and Save Animation ---
setup_plot(n, m)

# Animation frames = number of steps in traceback path + pause frames
num_anim_frames = total_frames + 5

ani = animation.FuncAnimation(fig, update, frames=num_anim_frames,
                              interval=INTERVAL, blit=True, repeat=False)

print(f"Attempting to save reverse traceback animation to {OUTPUT_FILENAME}...")
try:
    ani.save(OUTPUT_FILENAME, writer='pillow', fps=1000/INTERVAL)
    print(f"Animation saved successfully to {OUTPUT_FILENAME}")
except Exception as e:
    print(f"Error saving animation: {e}", file=sys.stderr)
    print("Ensure you have 'pillow' installed ('pip install pillow').", file=sys.stderr)

# plt.show()
