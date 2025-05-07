import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
import sys
import re # For parsing matrix files
from pathlib import Path

# Check current working directoys
import os
print(f"Current directory Python is running in: {os.getcwd()}")

# --- Configuration ---
PROTEIN_SEQ1 = "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFASFGNLSSPTAILGNPMVRAHGKKVLTSFGDAVKNLDNIKNTFSQLSELHCDKLHVDPENFRLLGNVLVCVLARNFGKEFTPQMQAAYQKVVAGVANALAHKYH"
PROTEIN_SEQ2 = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"

# Specify paths to your scoring matrix files
MATRIX_FILE_1 = "./alignment_matrix_impact/BLOSUM62.mat" # <--- SET PATH TO YOUR BLOSUM62 FILE
MATRIX_FILE_2 = "./alignment_matrix_impact/PAM250.mat"  # <--- SET PATH TO YOUR PAM250 FILE
MATRIX_NAME_1 = "BLOSUM62"
MATRIX_NAME_2 = "PAM250"

GAP_PENALTY = -8 # Example linear gap penalty

OUTPUT_FILENAME = "protein_alignment_comparison.gif"
INTERVAL = 150 # Milliseconds between frames (adjust for speed)
HIGHLIGHT_COLOR = 'yellow'

# --- 1. Parse Scoring Matrix File ---
def parse_scoring_matrix(filepath):
    """Parses a NCBI-style scoring matrix file."""
    matrix = {}
    amino_acids = []
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f"Scoring matrix file not found: {filepath}")

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue # Skip comments and empty lines

            parts = line.split()
            if not amino_acids:
                # Assume the first non-comment line is the AA header
                amino_acids = [aa.upper() for aa in parts]
            else:
                row_aa = parts[0].upper()
                if row_aa not in amino_acids: continue # Could be a footer or different format

                scores = list(map(int, parts[1:]))
                if len(scores) != len(amino_acids):
                    print(f"Warning: Score count mismatch in row {row_aa} of {filepath}. Skipping row.")
                    continue

                for col_idx, col_aa in enumerate(amino_acids):
                    score = scores[col_idx]
                    matrix[(row_aa, col_aa)] = score
                    matrix[(col_aa, row_aa)] = score # Ensure symmetry

    if not matrix:
         raise ValueError(f"Could not parse scoring matrix from {filepath}. Check format.")
    print(f"Parsed {filepath}, found {len(amino_acids)} amino acids.")
    return matrix, amino_acids

# --- 2. Adapt Needleman-Wunsch ---
def calculate_nw_protein(seq1, seq2, scoring_matrix, gap_penalty):
    """Calculates NW score and traceback matrices for proteins."""
    n = len(seq1)
    m = len(seq2)
    score_mat = np.zeros((n + 1, m + 1), dtype=int)
    trace_mat = np.full((n + 1, m + 1), '', dtype=object)

    # Initialization
    for i in range(1, n + 1):
        score_mat[i, 0] = score_mat[i-1, 0] + gap_penalty
        trace_mat[i, 0] = '↑'
    for j in range(1, m + 1):
        score_mat[0, j] = score_mat[0, j-1] + gap_penalty
        trace_mat[0, j] = '←'
    trace_mat[0, 0] = ' '

    # Filling the matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            aa1 = seq1[i-1].upper()
            aa2 = seq2[j-1].upper()

            # Get match/mismatch score from the matrix dictionary
            # Use a default very negative score if AA pair not in matrix
            match_val = scoring_matrix.get((aa1, aa2), -99)

            diag_score = score_mat[i-1, j-1] + match_val
            up_score = score_mat[i-1, j] + gap_penalty
            left_score = score_mat[i, j-1] + gap_penalty

            scores = [diag_score, up_score, left_score]
            max_score = max(scores)
            score_mat[i, j] = max_score

            # Tie-breaking: Diag > Up > Left (same as before)
            if max_score == diag_score: trace_mat[i, j] = '↖'
            elif max_score == up_score: trace_mat[i, j] = '↑'
            else: trace_mat[i, j] = '←'

    final_score = score_mat[n, m]
    return score_mat, trace_mat, n, m, final_score

def perform_traceback_protein(seq1, seq2, trace_mat):
    """Performs traceback for protein sequences."""
    n = len(seq1); m = len(seq2)
    i, j = n, m
    path_rev = [] # Path from (n,m) back to (0,0)
    aln1_rev = []; aln2_rev = []

    while i > 0 or j > 0:
        path_rev.append((i, j))
        direction = trace_mat[i, j]
        if direction == '↖':
            aln1_rev.append(seq1[i-1]); aln2_rev.append(seq2[j-1])
            i -= 1; j -= 1
        elif direction == '↑':
            aln1_rev.append(seq1[i-1]); aln2_rev.append('-')
            i -= 1
        elif direction == '←':
            aln1_rev.append('-'); aln2_rev.append(seq2[j-1])
            j -= 1
        else: # Boundary cases
             current_cell_score = -float('inf') # Need score matrix to check boundary properly, simplified here
             # Simplified boundary handling (assuming valid trace)
             if i == 0 and j > 0: direction = '←'
             elif j == 0 and i > 0: direction = '↑'
             else: break # Stop if at (0,0) or error
             if direction == '←': aln1_rev.append('-'); aln2_rev.append(seq2[j-1]); j -= 1
             elif direction == '↑': aln1_rev.append(seq1[i-1]); aln2_rev.append('-'); i -= 1

    path_rev.append((0, 0))
    aln1 = "".join(aln1_rev[::-1])
    aln2 = "".join(aln2_rev[::-1])
    return path_rev, aln1, aln2 # path is (n,m)->(0,0)

# --- 3. Run Alignments ---
try:
    matrix1, _ = parse_scoring_matrix(MATRIX_FILE_1)
    matrix2, _ = parse_scoring_matrix(MATRIX_FILE_2)
except (FileNotFoundError, ValueError) as e:
    print(f"Error loading scoring matrices: {e}")
    sys.exit(1)

print(f"Running NW with {MATRIX_NAME_1}...")
_, trace_m1, n1, m1, score_m1 = calculate_nw_protein(PROTEIN_SEQ1, PROTEIN_SEQ2, matrix1, GAP_PENALTY)
path_rev_m1, aln1_m1, aln2_m1 = perform_traceback_protein(PROTEIN_SEQ1, PROTEIN_SEQ2, trace_m1)

print(f"Running NW with {MATRIX_NAME_2}...")
_, trace_m2, n2, m2, score_m2 = calculate_nw_protein(PROTEIN_SEQ1, PROTEIN_SEQ2, matrix2, GAP_PENALTY)
path_rev_m2, aln1_m2, aln2_m2 = perform_traceback_protein(PROTEIN_SEQ1, PROTEIN_SEQ2, trace_m2)

print(f"{MATRIX_NAME_1} Score: {score_m1}")
print(f"{MATRIX_NAME_2} Score: {score_m2}")

# --- 4. Compare Alignments & Prepare for Animation ---
max_len = max(len(aln1_m1), len(aln1_m2)) # Alignments for seq1 should have same length, same for seq2

# Pad shorter alignments with leading spaces (for comparison)
# Note: NW alignment lengths for seq1 vs seq1 (and seq2 vs seq2) should be the same
# between the two runs if implemented correctly. Padding might not be strictly needed
# but is safer. Let's assume lengths can differ slightly due to traceback variants.
aln1_m1_padded = aln1_m1.rjust(max_len)
aln2_m1_padded = aln2_m1.rjust(max_len)
aln1_m2_padded = aln1_m2.rjust(max_len)
aln2_m2_padded = aln2_m2.rjust(max_len)

# Create difference flags (True if different at this position)
diff1 = [aln1_m1_padded[k] != aln1_m2_padded[k] for k in range(max_len)]
diff2 = [aln2_m1_padded[k] != aln2_m2_padded[k] for k in range(max_len)]
# Also consider if one has gap and other doesn't as difference
diff_gap1 = [(aln1_m1_padded[k] == '-') != (aln1_m2_padded[k] == '-') for k in range(max_len)]
diff_gap2 = [(aln2_m1_padded[k] == '-') != (aln2_m2_padded[k] == '-') for k in range(max_len)]

diff_combined1 = [(d or g) for d, g in zip(diff1, diff_gap1)]
diff_combined2 = [(d or g) for d, g in zip(diff2, diff_gap2)]


# Determine max path length for animation frames
max_path_len = max(len(path_rev_m1), len(path_rev_m2))

# --- 5. Animate ---
fig = plt.figure(figsize=(15, 8), constrained_layout=True)
# Grid spec: More rows for alignment, slightly wider right column
gs = gridspec.GridSpec(6, 2, figure=fig, height_ratios=[1, 1, 0.5, 1, 1, 1], width_ratios=[1, 3],
                       hspace=0.1, wspace=0.1)

# Left Column: Info
ax_info = fig.add_subplot(gs[0:2, 0]) # Span top two rows
# Right Column: Alignments
ax_aln1_m1 = fig.add_subplot(gs[0, 1]) # Matrix 1 - Seq1 Align
ax_match_m1 = fig.add_subplot(gs[1, 1]) # Matrix 1 - Match
ax_aln2_m1 = fig.add_subplot(gs[2, 1]) # Matrix 1 - Seq2 Align

ax_aln1_m2 = fig.add_subplot(gs[3, 1]) # Matrix 2 - Seq1 Align
ax_match_m2 = fig.add_subplot(gs[4, 1]) # Matrix 2 - Match
ax_aln2_m2 = fig.add_subplot(gs[5, 1]) # Matrix 2 - Seq2 Align

# Store artists for dynamic text and highlights
dynamic_artists = [] # Store text artists and highlight patches

def setup_plot():
    """Setup static plot elements."""
    global dynamic_artists
    dynamic_artists = [] # Clear for setup

    # --- Info Panel (Left) ---
    ax_info.set_axis_off()
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    info_text = (
        f"Seq1: {PROTEIN_SEQ1[:30]}...\n"
        f"Seq2: {PROTEIN_SEQ2[:30]}...\n\n"
        f"{MATRIX_NAME_1} Score: {score_m1}\n"
        f"{MATRIX_NAME_2} Score: {score_m2}\n\n"
        f"Gap Penalty: {GAP_PENALTY}\n\n"
        f"Highlight: {HIGHLIGHT_COLOR}"
    )
    ax_info.text(0.05, 0.95, info_text, va='top', ha='left', wrap=True, fontsize=9)
    ax_info.set_title("NW Alignment Comparison", fontsize=10)

    # --- Alignment Panels (Right) ---
    axes_right = [ax_aln1_m1, ax_match_m1, ax_aln2_m1, ax_aln1_m2, ax_match_m2, ax_aln2_m2]
    titles = [f"{MATRIX_NAME_1} Alignment", "", "", f"{MATRIX_NAME_2} Alignment", "", ""]
    labels = ["Aln1:", "     ", "Aln2:", "Aln1:", "     ", "Aln2:"] # Placeholders for labels

    for i, ax in enumerate(axes_right):
        ax.set_axis_off()
        ax.set_xlim(0, 1) # Use relative coordinates for text placement
        ax.set_ylim(0, 1)
        if titles[i]: ax.set_title(titles[i], fontsize=10)
        # Create placeholder text artists, aligned RIGHT, positioned near right edge (x=1.0)
        # We will draw char-by-char later, so placeholders are less critical
        # but create one for structure if needed
        t = ax.text(1.0, 0.5, labels[i], fontsize=10, family='monospace', va='center', ha='right')
        # Store these base labels if needed, or just redraw everything in update
        # dynamic_artists.append(t) # Don't add static labels to dynamic list

def draw_dynamic_alignment(ax, label, alignment_str, diff_flags, frame_len):
    """Draws the alignment string char-by-char with highlighting."""
    # Clear previous chars/highlights on this axis
    for artist in ax.findobj(match=lambda x: hasattr(x, 'set_gid') and x.get_gid() == 'dynamic_char'):
        artist.remove()

    max_chars_to_display = 80 # Limit display width
    current_len = len(alignment_str)
    start_index = max(0, current_len - max_chars_to_display)
    display_str = alignment_str[start_index:]
    display_diff = diff_flags[start_index:]

    # Calculate starting x position based on fixed-width font assumption
    char_width_approx = 0.011 # Adjust based on font/figure size
    current_x = 1.0 # Start from right edge

    # Draw label first
    lbl = ax.text(current_x - (len(display_str)+len(label))*char_width_approx , 0.5, label, fontsize=10, family='monospace', va='center', ha='right', transform=ax.transAxes, gid='dynamic_char')
    dynamic_artists.append(lbl)

    # Draw chars right to left
    for i in range(len(display_str) - 1, -1, -1):
        char = display_str[i]
        is_diff = display_diff[i]

        # Add highlight patch if different
        if is_diff:
            rect = Rectangle((current_x - char_width_approx, 0.1), char_width_approx*1.1, 0.8,
                             facecolor=HIGHLIGHT_COLOR, edgecolor='none', alpha=0.5, zorder=-1,
                             transform=ax.transAxes, gid='dynamic_char')
            ax.add_patch(rect)
            dynamic_artists.append(rect)

        # Add character text
        t = ax.text(current_x - char_width_approx / 2, 0.5, char, fontsize=10, family='monospace',
                    va='center', ha='center', transform=ax.transAxes, gid='dynamic_char')
        dynamic_artists.append(t)

        current_x -= char_width_approx


def update(frame):
    """Update function for animation."""
    global dynamic_artists
    # Clear artists from previous frame (safer than relying on blit for complex scenes)
    # Especially needed for the char-by-char drawing
    for artist in dynamic_artists:
        try: artist.remove()
        except (ValueError, AttributeError): pass # Handle cases where artist might be gone
    dynamic_artists = []

    # Determine number of alignment characters to show based on frame
    # Use max_len as the total number of steps/chars in the final alignment
    num_chars_to_show = min(frame + 1, max_len)

    # Get the rightmost 'num_chars_to_show' characters
    partial_aln1_m1 = aln1_m1_padded[-num_chars_to_show:]
    partial_aln2_m1 = aln2_m1_padded[-num_chars_to_show:]
    partial_aln1_m2 = aln1_m2_padded[-num_chars_to_show:]
    partial_aln2_m2 = aln2_m2_padded[-num_chars_to_show:]

    # Get corresponding difference flags
    partial_diff1 = diff_combined1[-num_chars_to_show:]
    partial_diff2 = diff_combined2[-num_chars_to_show:]

    # Build partial match strings
    match_str_m1 = "".join(["|" if p1 == p2 and p1 != '-' else ("." if p1 != '-' and p2 != '-' else " ")
                           for p1, p2 in zip(partial_aln1_m1, partial_aln2_m1)])
    match_str_m2 = "".join(["|" if p1 == p2 and p1 != '-' else ("." if p1 != '-' and p2 != '-' else " ")
                           for p1, p2 in zip(partial_aln1_m2, partial_aln2_m2)])

    # Draw the alignments dynamically using char-by-char function
    draw_dynamic_alignment(ax_aln1_m1, "Aln1:", partial_aln1_m1, partial_diff1, frame)
    draw_dynamic_alignment(ax_match_m1, "     ", match_str_m1, [False]*len(match_str_m1), frame) # No diffs on match line
    draw_dynamic_alignment(ax_aln2_m1, "Aln2:", partial_aln2_m1, partial_diff2, frame)

    draw_dynamic_alignment(ax_aln1_m2, "Aln1:", partial_aln1_m2, partial_diff1, frame)
    draw_dynamic_alignment(ax_match_m2, "     ", match_str_m2, [False]*len(match_str_m2), frame) # No diffs on match line
    draw_dynamic_alignment(ax_aln2_m2, "Aln2:", partial_aln2_m2, partial_diff2, frame)


    # Return list of artists (important for blitting if enabled, good practice otherwise)
    # Since we clear/redraw everything, returning might not be strictly needed if blit=False
    # But let's return the list managed by draw_dynamic_alignment
    return dynamic_artists


# --- Create and Save Animation ---
setup_plot()

# Total frames = max alignment length + pause frames
num_anim_frames = max_len + 10

# blit=False might be necessary due to the complexity of clearing/redrawing
# char-by-char and adding/removing patches. Test with True first.
ani = animation.FuncAnimation(fig, update, frames=num_anim_frames,
                              interval=INTERVAL, blit=False, repeat=False) # Try blit=False if True causes issues

print(f"Attempting to save comparison animation to {OUTPUT_FILENAME}...")
try:
    ani.save(OUTPUT_FILENAME, writer='pillow', fps=1000/INTERVAL, dpi=100) # Adjust dpi if needed
    print(f"Animation saved successfully to {OUTPUT_FILENAME}")
except Exception as e:
    print(f"Error saving animation: {e}", file=sys.stderr)
    print("Ensure you have 'pillow' installed ('pip install pillow').", file=sys.stderr)
    print("If saving fails, try setting blit=False in FuncAnimation.", file=sys.stderr)

# plt.show() # Optional: Show plot interactively