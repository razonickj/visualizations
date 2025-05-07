import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
import sys

# --- Configuration ---
SEQ1 = "GCATGCGAA"
SEQ2 = "GATTACA"
MATCH_SCORE = 1
MISMATCH_SCORE = -1
GAP_PENALTY = -1
OUTPUT_FILENAME = "needleman_wunsch_traceback_rtl.gif" # Changed filename again
INTERVAL = 600 # Milliseconds between frames

# --- Needleman-Wunsch Table Calculation (Same as before) ---
def calculate_nw_matrices(seq1, seq2, match, mismatch, gap):
    n = len(seq1)
    m = len(seq2)
    score_matrix = np.zeros((n + 1, m + 1), dtype=int)
    traceback_matrix = np.full((n + 1, m + 1), '', dtype=object)
    # Initialization
    for i in range(1, n + 1):
        score_matrix[i, 0] = score_matrix[i-1, 0] + gap; traceback_matrix[i, 0] = '↑'
    for j in range(1, m + 1):
        score_matrix[0, j] = score_matrix[0, j-1] + gap; traceback_matrix[0, j] = '←'
    traceback_matrix[0, 0] = ' '
    # Filling
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_val = match if seq1[i-1] == seq2[j-1] else mismatch
            diag_score = score_matrix[i-1, j-1] + match_val
            up_score = score_matrix[i-1, j] + gap
            left_score = score_matrix[i, j-1] + gap
            scores = [diag_score, up_score, left_score]; max_score = max(scores)
            score_matrix[i, j] = max_score
            if max_score == diag_score: traceback_matrix[i, j] = '↖'
            elif max_score == up_score: traceback_matrix[i, j] = '↑'
            else: traceback_matrix[i, j] = '←'
    return score_matrix, traceback_matrix, n, m

# --- Perform Traceback (Same as before) ---
def perform_traceback(seq1, seq2, traceback_matrix):
    n = len(seq1); m = len(seq2)
    i, j = n, m
    path_cells_reverse = [] # Path from (n,m) back to (0,0)
    align1_rev = []; align2_rev = []
    while i > 0 or j > 0:
        path_cells_reverse.append((i, j))
        direction = traceback_matrix[i, j]
        if direction == '↖':
            align1_rev.append(seq1[i-1]); align2_rev.append(seq2[j-1])
            i -= 1; j -= 1
        elif direction == '↑':
            align1_rev.append(seq1[i-1]); align2_rev.append('-') # Gap char
            i -= 1
        elif direction == '←':
            align1_rev.append('-'); align2_rev.append(seq2[j-1]) # Gap char
            j -= 1
        else: # Boundary or error
            if i == 0 and j > 0: direction = '←'
            elif j == 0 and i > 0: direction = '↑'
            else: print(f"Warning: Reached ({i},{j}) trace '{direction}'"); break
            if direction == '←': align1_rev.append('-'); align2_rev.append(seq2[j-1]); j -= 1
            elif direction == '↑': align1_rev.append(seq1[i-1]); align2_rev.append('-'); i -= 1
    path_cells_reverse.append((0, 0))
    align1 = "".join(align1_rev[::-1]) # Final alignment (correct L-to-R order)
    align2 = "".join(align2_rev[::-1])
    return path_cells_reverse, align1, align2 # path is (n,m)->(0,0)

# Calculate matrices and perform traceback
score_matrix, traceback_matrix, n, m = calculate_nw_matrices(
    SEQ1, SEQ2, MATCH_SCORE, MISMATCH_SCORE, GAP_PENALTY
)
traceback_path_rev, alignment1, alignment2 = perform_traceback(SEQ1, SEQ2, traceback_matrix)

# --- Animation Setup ---
fig = plt.figure(figsize=(12, 6), constrained_layout=True)
gs = gridspec.GridSpec(5, 2, figure=fig, height_ratios=[1, 1, 0.5, 1, 1], width_ratios=[3, 2],
                       hspace=0.1, wspace=0.3) # Adjusted grid for 5 rows

ax_table = fig.add_subplot(gs[:, 0])       # Table still on left
# Reserve rows on the right for alignment display
ax_seq1_static = fig.add_subplot(gs[0, 1])
ax_aln1_dynamic = fig.add_subplot(gs[1, 1])
ax_match_dynamic = fig.add_subplot(gs[2, 1])
ax_aln2_dynamic = fig.add_subplot(gs[3, 1])
ax_seq2_static = fig.add_subplot(gs[4, 1])

# Store artists
cell_texts = {}
path_highlights = []
# Artists for the 5 text lines on the right
seq1_static_artist = None
aln1_dynamic_artist = None
match_dynamic_artist = None
aln2_dynamic_artist = None
seq2_static_artist = None

# --- Plotting Functions ---
def setup_plot(n, m):
    """Initial setup of the plot elements."""
    global seq1_static_artist, aln1_dynamic_artist, match_dynamic_artist, aln2_dynamic_artist, seq2_static_artist

    # --- Main Table Axis (ax_table) - Unchanged ---
    ax_table.set_xticks(np.arange(m + 1) - 0.5, minor=True); ax_table.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax_table.set_xticks(np.arange(m + 1)); ax_table.set_yticks(np.arange(n + 1))
    ax_table.set_xticklabels([''] + list(SEQ2)); ax_table.set_yticklabels([''] + list(SEQ1))
    ax_table.xaxis.tick_top(); ax_table.xaxis.set_label_position('top')
    ax_table.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax_table.set_xlim(-0.5, m + 0.5); ax_table.set_ylim(n + 0.5, -0.5)
    ax_table.tick_params(axis='both', which='major', length=0)
    ax_table.set_title("Needleman-Wunsch Traceback (Bottom-Right to Top-Left)")
    # Fill table texts - Unchanged
    for r in range(n + 1):
        for c in range(m + 1):
            score = score_matrix[r, c]; trace = traceback_matrix[r, c]
            cell_texts[(r, c)] = ax_table.text(c, r, f"{score}\n{trace}", ha='center', va='center', fontsize=9, zorder=1)

    # --- Alignment Axes (Right Side) ---
    # Configure all 5 axes on the right
    axes_right = [ax_seq1_static, ax_aln1_dynamic, ax_match_dynamic, ax_aln2_dynamic, ax_seq2_static]
    for ax in axes_right:
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Static Sequences (aligned right for consistency)
    seq1_static_artist = ax_seq1_static.text(0.95, 0.5, f"Seq1: {SEQ1}", fontsize=11, family='monospace', va='center', ha='right')
    seq2_static_artist = ax_seq2_static.text(0.95, 0.5, f"Seq2: {SEQ2}", fontsize=11, family='monospace', va='center', ha='right')

    # Dynamic Alignment Placeholders (aligned right)
    aln1_dynamic_artist = ax_aln1_dynamic.text(0.95, 0.5, "Aln1: ", fontsize=12, family='monospace', va='center', ha='right')
    match_dynamic_artist = ax_match_dynamic.text(0.95, 0.5, "", fontsize=12, family='monospace', va='center', ha='right') # No label for match line
    aln2_dynamic_artist = ax_aln2_dynamic.text(0.95, 0.5, "Aln2: ", fontsize=12, family='monospace', va='center', ha='right')

    axes_right[0].set_title("Alignment (Grows Right-to-Left)") # Title for the block


# --- Animation Update Function ---
total_frames = len(traceback_path_rev)

def update(frame):
    """Update function called for each frame of the traceback animation."""
    global path_highlights

    # --- Path Highlighting on Table (Unchanged from previous version) ---
    for artist in path_highlights: artist.remove()
    path_highlights.clear()
    current_path_index = frame
    if current_path_index >= total_frames: current_path_index = total_frames - 1
    path_segments_coords = []
    max_k = min(current_path_index, total_frames - 1)
    for k in range(max_k + 1):
        r, c = traceback_path_rev[k]
        path_segments_coords.append((c, r))
        rect_color = 'lightcoral' if k==0 else 'cyan'
        rect = Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=rect_color, alpha=0.5, zorder=0, edgecolor='none')
        ax_table.add_patch(rect); path_highlights.append(rect)
    if len(path_segments_coords) > 1:
        path_x, path_y = zip(*path_segments_coords)
        line, = ax_table.plot(path_x, path_y, color='red', linewidth=2.5, zorder=2)
        path_highlights.append(line)
    if current_path_index < total_frames:
        curr_r, curr_c = traceback_path_rev[current_path_index]
        current_rect = Rectangle((curr_c - 0.5, curr_r - 0.5), 1, 1, linewidth=2, edgecolor='magenta', facecolor='none', zorder=3)
        ax_table.add_patch(current_rect); path_highlights.append(current_rect)
    # --- End Path Highlighting ---


    # --- Update alignment text dynamically (Right-to-Left Growth) ---
    # frame 0 = 1st step back = last char of alignment
    num_chars_to_show = min(frame + 1, len(alignment1))

    # Get the rightmost 'num_chars_to_show' characters from the final alignment
    # These already include the '-' gap characters
    partial_align1 = alignment1[-num_chars_to_show:]
    partial_align2 = alignment2[-num_chars_to_show:]

    # Build match string for the partial alignment
    match_str = ""
    for k in range(len(partial_align1)):
        if k >= len(partial_align2): break
        if partial_align1[k] == partial_align2[k]: match_str += "|"
        elif partial_align1[k] == '-' or partial_align2[k] == '-': match_str += " "
        else: match_str += "."

    # Update the text artists WITHOUT reversing the partial strings.
    # The ha='right' alignment handles the right-to-left growth appearance.
    aln1_dynamic_artist.set_text(f"Aln1: {partial_align1}")
    match_dynamic_artist.set_text(f"      {match_str}") # Maintain spacing for label
    aln2_dynamic_artist.set_text(f"Aln2: {partial_align2}")

    # Return list of artists that were modified
    artists = (list(cell_texts.values()) + path_highlights +
               [aln1_dynamic_artist, match_dynamic_artist, aln2_dynamic_artist])
               # Static text artists don't change, no need to return if blit=True
    return artists

# --- Create and Save Animation ---
setup_plot(n, m)

num_anim_frames = total_frames + 5 # Pause at the end

ani = animation.FuncAnimation(fig, update, frames=num_anim_frames,
                              interval=INTERVAL, blit=True, repeat=False)

print(f"Attempting to save right-to-left traceback animation to {OUTPUT_FILENAME}...")
try:
    ani.save(OUTPUT_FILENAME, writer='pillow', fps=1000/INTERVAL)
    print(f"Animation saved successfully to {OUTPUT_FILENAME}")
except Exception as e:
    print(f"Error saving animation: {e}", file=sys.stderr)
    print("Ensure you have 'pillow' installed ('pip install pillow').", file=sys.stderr)

# plt.show()