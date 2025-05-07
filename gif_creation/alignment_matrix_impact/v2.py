import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
import sys
import re
from pathlib import Path
import math

# --- Configuration ---
# Using shorter sequences for better GIF readability
PROTEIN_SEQ1 = "MVHLTPEEKSAVTALWGKVN"
PROTEIN_SEQ2 = "MVLSPADKTNVKAAWGKVG"

# Specify paths to your scoring matrix files
MATRIX_FILE_1 = "./alignment_matrix_impact/BLOSUM62.mat" # <--- SET PATH TO YOUR BLOSUM62 FILE
MATRIX_FILE_2 = "./alignment_matrix_impact/PAM250.mat"  # <--- SET PATH TO YOUR PAM250 FILE
MATRIX_NAME_1 = "BLOSUM62"
MATRIX_NAME_2 = "PAM250"

GAP_PENALTY = -8 # Example linear gap penalty

OUTPUT_FILENAME = "protein_alignment_scan_compare.gif"
INTERVAL = 100 # Milliseconds between frames (adjust for speed)
HIGHLIGHT_DIFF_COLOR = 'red'
SCAN_BAR_COLOR = 'yellow'
FONT_SIZE = 9
FONT_FAMILY = 'monospace'

# --- 1. Parse Scoring Matrix File (Same as before) ---
def parse_scoring_matrix(filepath):
    matrix = {}; amino_acids = []
    filepath = Path(filepath)
    if not filepath.is_file(): raise FileNotFoundError(f"Scoring matrix file not found: {filepath}")
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split()
            if not amino_acids: amino_acids = [aa.upper() for aa in parts]
            else:
                row_aa = parts[0].upper(); scores = list(map(int, parts[1:]))
                if row_aa not in amino_acids or len(scores) != len(amino_acids): continue
                for col_idx, col_aa in enumerate(amino_acids):
                    score = scores[col_idx]
                    matrix[(row_aa, col_aa)] = score; matrix[(col_aa, row_aa)] = score
    if not matrix: raise ValueError(f"Could not parse matrix from {filepath}.")
    # print(f"Parsed {filepath}, found {len(amino_acids)} AAs.")
    return matrix, amino_acids

# --- 2. Adapt Needleman-Wunsch (Same as before) ---
def calculate_nw_protein(seq1, seq2, scoring_matrix, gap_penalty):
    n = len(seq1); m = len(seq2)
    score_mat = np.zeros((n + 1, m + 1), dtype=int)
    trace_mat = np.full((n + 1, m + 1), '', dtype=object)
    for i in range(1, n + 1): score_mat[i, 0] = score_mat[i-1, 0] + gap_penalty; trace_mat[i, 0] = '↑'
    for j in range(1, m + 1): score_mat[0, j] = score_mat[0, j-1] + gap_penalty; trace_mat[0, j] = '←'
    trace_mat[0, 0] = ' '
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            aa1 = seq1[i-1].upper(); aa2 = seq2[j-1].upper()
            match_val = scoring_matrix.get((aa1, aa2), -99) # Default penalty
            diag_score = score_mat[i-1, j-1] + match_val
            up_score = score_mat[i-1, j] + gap_penalty
            left_score = score_mat[i, j-1] + gap_penalty
            scores = [diag_score, up_score, left_score]; max_score = max(scores)
            score_mat[i, j] = max_score
            if max_score == diag_score: trace_mat[i, j] = '↖'
            elif max_score == up_score: trace_mat[i, j] = '↑'
            else: trace_mat[i, j] = '←'
    final_score = score_mat[n, m]
    return score_mat, trace_mat, n, m, final_score

def perform_traceback_protein(seq1, seq2, trace_mat):
    n = len(seq1); m = len(seq2); i, j = n, m
    path_rev = []; aln1_rev = []; aln2_rev = []
    while i > 0 or j > 0:
        path_rev.append((i, j)); direction = trace_mat[i, j]
        if direction == '↖': aln1_rev.append(seq1[i-1]); aln2_rev.append(seq2[j-1]); i -= 1; j -= 1
        elif direction == '↑': aln1_rev.append(seq1[i-1]); aln2_rev.append('-'); i -= 1
        elif direction == '←': aln1_rev.append('-'); aln2_rev.append(seq2[j-1]); j -= 1
        else:
            if i == 0 and j > 0: direction = '←'
            elif j == 0 and i > 0: direction = '↑'
            else: break
            if direction == '←': aln1_rev.append('-'); aln2_rev.append(seq2[j-1]); j -= 1
            elif direction == '↑': aln1_rev.append(seq1[i-1]); aln2_rev.append('-'); i -= 1
    path_rev.append((0, 0))
    aln1 = "".join(aln1_rev[::-1]); aln2 = "".join(aln2_rev[::-1])
    return path_rev, aln1, aln2

# --- 3. Run Alignments (Same as before) ---
try:
    matrix1, _ = parse_scoring_matrix(MATRIX_FILE_1)
    matrix2, _ = parse_scoring_matrix(MATRIX_FILE_2)
except (FileNotFoundError, ValueError) as e: print(f"Error: {e}"); sys.exit(1)

_, trace_m1, n1, m1, score_m1 = calculate_nw_protein(PROTEIN_SEQ1, PROTEIN_SEQ2, matrix1, GAP_PENALTY)
_, aln1_m1, aln2_m1 = perform_traceback_protein(PROTEIN_SEQ1, PROTEIN_SEQ2, trace_m1)

_, trace_m2, n2, m2, score_m2 = calculate_nw_protein(PROTEIN_SEQ1, PROTEIN_SEQ2, matrix2, GAP_PENALTY)
_, aln1_m2, aln2_m2 = perform_traceback_protein(PROTEIN_SEQ1, PROTEIN_SEQ2, trace_m2)

print(f"{MATRIX_NAME_1} Score: {score_m1}")
print(f"{MATRIX_NAME_2} Score: {score_m2}")

# --- 4. Compare Alignments & Prepare ---
max_len = max(len(aln1_m1), len(aln1_m2)) # Use the longest alignment length

# Pad shorter alignments with trailing spaces (or could use leading)
aln1_m1 = aln1_m1.ljust(max_len)
aln2_m1 = aln2_m1.ljust(max_len)
aln1_m2 = aln1_m2.ljust(max_len)
aln2_m2 = aln2_m2.ljust(max_len)

# Generate match strings
match_str_m1 = "".join(["|" if a1 == a2 and a1 != '-' else ("." if a1 != '-' and a2 != '-' else " ")
                        for a1, a2 in zip(aln1_m1, aln2_m1)])
match_str_m2 = "".join(["|" if a1 == a2 and a1 != '-' else ("." if a1 != '-' and a2 != '-' else " ")
                        for a1, a2 in zip(aln1_m2, aln2_m2)])

# Create difference flags for each COLUMN
# A column differs if seq1 alignment OR seq2 alignment differs
diff_cols = [False] * max_len
for k in range(max_len):
    if aln1_m1[k] != aln1_m2[k] or aln2_m1[k] != aln2_m2[k]:
        diff_cols[k] = True

# --- 5. Animate ---
# Estimate character width for positioning (highly approximate)
# This is tricky and depends on figure size, dpi, font.
# We might need to adjust this based on output.
estimated_char_width = 0.01 # Fraction of the axes width

fig = plt.figure(figsize=(12, 4), constrained_layout=True) # Adjust size as needed
# Fewer rows needed now
gs = gridspec.GridSpec(5, 1, figure=fig, height_ratios=[1, 1, 0.5, 1, 1], hspace=0.1)

ax_m1_aln1 = fig.add_subplot(gs[0, 0])
ax_m1_match = fig.add_subplot(gs[1, 0])
ax_m1_aln2 = fig.add_subplot(gs[2, 0])
# Add a little space? Maybe combine match/aln2 later if needed
ax_m2_aln1 = fig.add_subplot(gs[3, 0])
# ax_m2_match = fig.add_subplot(gs[3, 0]) # Maybe overlay match? Skip for now
ax_m2_aln2 = fig.add_subplot(gs[4, 0])

# Store artists: scan bar, reveal patch, highlight patches
scan_bar = None
reveal_patch = None
diff_highlight_patches = []
static_text_artists = []

def setup_plot():
    """Setup static plot elements and text."""
    global scan_bar, reveal_patch, static_text_artists
    static_text_artists = []

    axes = [ax_m1_aln1, ax_m1_match, ax_m1_aln2, ax_m2_aln1, ax_m2_aln2]
    alignments = [aln1_m1, match_str_m1, aln2_m1, aln1_m2, aln2_m2]
    labels = [f"{MATRIX_NAME_1} Aln1:", " "*len(f"{MATRIX_NAME_1} Match:"), f"{MATRIX_NAME_1} Aln2:",
              f"{MATRIX_NAME_2} Aln1:", f"{MATRIX_NAME_2} Aln2:"]

    # Determine plot limits based on alignment length
    # Use index as x-coordinate
    x_limit = max_len

    for i, ax in enumerate(axes):
        ax.set_axis_off()
        # Set xlim based on number of characters/columns
        ax.set_xlim(-1, x_limit) # Use index as coordinate
        ax.set_ylim(0, 1)

        # Draw the FULL alignment string (will be covered by reveal_patch)
        # Position text slightly offset from the left edge (x=0)
        t = ax.text(0, 0.5, labels[i] + " " + alignments[i],
                    fontsize=FONT_SIZE, family=FONT_FAMILY, va='center', ha='left')
        static_text_artists.append(t) # Store to potentially adjust later if needed

    # Initial Scan Bar (off-screen left) - using axvline for simplicity
    # We'll use the top axes (ax_m1_aln1) for the scan bar coordinate system
    scan_bar = ax_m1_aln1.axvline(-1, color=SCAN_BAR_COLOR, linewidth=4, zorder=10)

    # Initial Reveal Patch (covers everything)
    # Spans all axes vertically. Get bounding box of all axes.
    fig.canvas.draw() # Ensure layout is calculated
    bboxes = [ax.get_window_extent() for ax in axes]
    if not bboxes: return # Should not happen
    full_extent = Bbox.union(bboxes)
    inv = fig.transFigure.inverted()
    fig_bbox = full_extent.transformed(inv)

    # Get data coords for x range. Y range in axes coords (0 to 1)
    # Use the top axes for data coord reference
    reveal_patch = ax_m1_aln1.axvspan(-0.5, x_limit - 0.5, facecolor='white', alpha=1.0, zorder=5, ymin=0, ymax=5) # ymax > 1 covers vertically somewhat
    # A better way for vertical span might be a figure-level patch, but axvspan is easier
    # We need to manually adjust ymin/ymax if axes aren't perfectly stacked/aligned
    # Let's try a figure patch instead
    reveal_patch = Rectangle((0, 0), 1, 1, transform=fig.transFigure, facecolor='white', alpha=1.0, zorder=5, clip_on=False)

    # Manually set figure patch coords later in update based on scan bar x

def update(frame):
    """Update function for animation."""
    global diff_highlight_patches, reveal_patch

    # Clear previous difference highlights
    for patch in diff_highlight_patches:
        patch.remove()
    diff_highlight_patches = []

    # Current column index (0 to max_len - 1)
    current_col = frame
    if current_col >= max_len:
        current_col = max_len # Keep bar at the end

    # --- Update Scan Bar ---
    # Position bar just AFTER the column being revealed
    scan_bar_x_pos = current_col + 0.5
    scan_bar.set_xdata([scan_bar_x_pos, scan_bar_x_pos])

    # --- Update Reveal Patch ---
    # Convert scan bar x position (data coords in top ax) to figure coords
    # We want the white patch to cover from scan_bar_x_pos to the right edge
    display_coords_start = ax_m1_aln1.transData.transform((scan_bar_x_pos, 0))
    fig_coords_start = fig.transFigure.inverted().transform(display_coords_start)
    reveal_x_start = fig_coords_start[0]

    # Make patch cover from reveal_x_start to right edge of figure (1.0)
    reveal_patch.set_bounds(reveal_x_start, 0, 1.0 - reveal_x_start, 1)

    # --- Add Difference Highlights for newly revealed column ---
    if current_col < max_len and diff_cols[current_col]:
        # Add a red vertical span for this column
        # Calculate xmin, xmax for the highlight patch in data coords
        xmin = current_col - 0.5
        xmax = current_col + 0.5
        # Add patch to each relevant axes row (better than one tall one?)
        axes_to_highlight = [ax_m1_aln1, ax_m1_aln2, ax_m2_aln1, ax_m2_aln2] # Skip match lines
        for ax in axes_to_highlight:
             # Use axvspan which respects axis limits better
             patch = ax.axvspan(xmin, xmax, facecolor=HIGHLIGHT_DIFF_COLOR, alpha=0.3, zorder=1, ymin=0.05, ymax=0.95)
             diff_highlight_patches.append(patch)


    # Artists to return (scan bar updates itself, need reveal + highlights)
    return [scan_bar, reveal_patch] + diff_highlight_patches + static_text_artists # Include static if not blitting


# --- Matplotlib setup for animation ---
# Needed for Bbox calculation in setup
from matplotlib.transforms import Bbox

# --- Create and Save Animation ---
setup_plot()

# Total frames = max alignment length + pause frames
num_anim_frames = max_len + 15

ani = animation.FuncAnimation(fig, update, frames=num_anim_frames,
                              interval=INTERVAL, blit=False, repeat=False) # MUST use blit=False

print(f"Attempting to save scan comparison animation to {OUTPUT_FILENAME}...")
try:
    ani.save(OUTPUT_FILENAME, writer='pillow', fps=1000/INTERVAL, dpi=120) # Increase dpi for clarity
    print(f"Animation saved successfully to {OUTPUT_FILENAME}")
except Exception as e:
    print(f"Error saving animation: {e}", file=sys.stderr)
    print("Ensure you have 'pillow' installed ('pip install pillow').")

# plt.show()