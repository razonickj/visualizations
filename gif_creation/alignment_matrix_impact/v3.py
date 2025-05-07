import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
import sys
import re
from pathlib import Path
import math
import traceback # Import traceback for detailed error


# --- Configuration ---
PROTEIN_SEQ1 = "MVHLTPEEKSAVTALWGKVN"
PROTEIN_SEQ2 = "MVLSPADKTNVKAAWGKVG"
MATRIX_FILE_1 = "./alignment_matrix_impact/BLOSUM62.mat" # <--- SET PATH TO YOUR BLOSUM62 FILE
MATRIX_FILE_2 = "./alignment_matrix_impact/PAM250.mat"  # <--- SET PATH TO YOUR PAM250 FILE
MATRIX_NAME_1 = "BLOSUM62"
MATRIX_NAME_2 = "PAM250"
GAP_PENALTY = -8
OUTPUT_FILENAME = "protein_alignment_scan_compare_v4.gif" # Incremented version
INTERVAL = 100
HIGHLIGHT_DIFF_COLOR = 'red'
SCAN_BAR_COLOR = 'yellow'
FONT_SIZE = 9
FONT_FAMILY = 'monospace'
REVEAL_BG_COLOR = plt.rcParams['figure.facecolor'] # Use figure background color

# --- Parse Scoring Matrix (Same as before) ---
def parse_scoring_matrix(filepath):
    matrix = {}; amino_acids = []
    filepath = Path(filepath)
    if not filepath.is_file(): raise FileNotFoundError(f"Scoring matrix file not found: {filepath}")
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip(); parts = line.split()
            if not line or line.startswith('#'): continue
            if not amino_acids: amino_acids = [aa.upper() for aa in parts]
            else:
                row_aa = parts[0].upper(); scores = list(map(int, parts[1:]))
                if row_aa not in amino_acids or len(scores) != len(amino_acids): continue
                for col_idx, col_aa in enumerate(amino_acids):
                    score = scores[col_idx]; matrix[(row_aa, col_aa)] = score; matrix[(col_aa, row_aa)] = score
    if not matrix: raise ValueError(f"Could not parse matrix from {filepath}.")
    return matrix, amino_acids

# --- Needleman-Wunsch (Same as before) ---
def calculate_nw_protein(seq1, seq2, scoring_matrix, gap_penalty):
    n=len(seq1); m=len(seq2); score_mat=np.zeros((n+1, m+1),dtype=int); trace_mat=np.full((n+1, m+1),'',dtype=object)
    for i in range(1,n+1): score_mat[i,0]=score_mat[i-1,0]+gap_penalty; trace_mat[i,0]='↑'
    for j in range(1,m+1): score_mat[0,j]=score_mat[0,j-1]+gap_penalty; trace_mat[0,j]='←'
    trace_mat[0,0]=' '
    for i in range(1,n+1):
        for j in range(1,m+1):
            aa1=seq1[i-1].upper(); aa2=seq2[j-1].upper(); match_val=scoring_matrix.get((aa1, aa2),-99)
            diag=score_mat[i-1,j-1]+match_val; up=score_mat[i-1,j]+gap_penalty; left=score_mat[i,j-1]+gap_penalty
            scores=[diag,up,left]; max_score=max(scores); score_mat[i,j]=max_score
            if max_score==diag: trace_mat[i,j]='↖'
            elif max_score==up: trace_mat[i,j]='↑'
            else: trace_mat[i,j]='←'
    return score_mat,trace_mat,n,m,score_mat[n,m]

def perform_traceback_protein(seq1, seq2, trace_mat):
    n=len(seq1); m=len(seq2); i,j=n,m; path_rev=[]; aln1_rev=[]; aln2_rev=[]
    while i>0 or j>0:
        path_rev.append((i,j)); direction=trace_mat[i,j]
        if direction=='↖': aln1_rev.append(seq1[i-1]); aln2_rev.append(seq2[j-1]); i-=1; j-=1
        elif direction=='↑': aln1_rev.append(seq1[i-1]); aln2_rev.append('-'); i-=1
        elif direction=='←': aln1_rev.append('-'); aln2_rev.append(seq2[j-1]); j-=1
        else:
            if i==0 and j>0: direction='←'
            elif j==0 and i>0: direction='↑'
            else: break # Should be at (0,0)
            if direction=='←': aln1_rev.append('-'); aln2_rev.append(seq2[j-1]); j-=1
            elif direction=='↑': aln1_rev.append(seq1[i-1]); aln2_rev.append('-'); i-=1
    path_rev.append((0,0)); aln1="".join(aln1_rev[::-1]); aln2="".join(aln2_rev[::-1])
    return path_rev, aln1, aln2

# --- Run Alignments & Compare ---
try:
    matrix1, _ = parse_scoring_matrix(MATRIX_FILE_1)
    matrix2, _ = parse_scoring_matrix(MATRIX_FILE_2)
except (FileNotFoundError, ValueError) as e: print(f"Error: {e}"); sys.exit(1)

_, trace_m1, n1, m1, score_m1 = calculate_nw_protein(PROTEIN_SEQ1, PROTEIN_SEQ2, matrix1, GAP_PENALTY)
_, aln1_m1, aln2_m1 = perform_traceback_protein(PROTEIN_SEQ1, PROTEIN_SEQ2, trace_m1)
_, trace_m2, n2, m2, score_m2 = calculate_nw_protein(PROTEIN_SEQ1, PROTEIN_SEQ2, matrix2, GAP_PENALTY)
_, aln1_m2, aln2_m2 = perform_traceback_protein(PROTEIN_SEQ1, PROTEIN_SEQ2, trace_m2)

max_len = max(len(aln1_m1), len(aln1_m2))
aln1_m1 = aln1_m1.ljust(max_len); aln2_m1 = aln2_m1.ljust(max_len)
aln1_m2 = aln1_m2.ljust(max_len); aln2_m2 = aln2_m2.ljust(max_len)
diff_cols = [False] * max_len
for k in range(max_len):
    if aln1_m1[k] != aln1_m2[k] or aln2_m1[k] != aln2_m2[k]: diff_cols[k] = True

# **** ADD LENGTH CHECKS ****
print("-" * 20 + "\nData Length Verification:")
print(f"Max alignment length (max_len): {max_len}")
print(f"Length aln1_m1: {len(aln1_m1)}, aln2_m1: {len(aln2_m1)}")
print(f"Length aln1_m2: {len(aln1_m2)}, aln2_m2: {len(aln2_m2)}")
print(f"Length diff_cols: {len(diff_cols)}")
if not (len(aln1_m1)==max_len==len(aln2_m1)==len(aln1_m2)==len(aln2_m2)==len(diff_cols)):
    print("ERROR: Length mismatch found!") ; sys.exit(1)
else: print("All lengths consistent."); print("-" * 20)
# **** END LENGTH CHECKS ****

# --- Animate ---
fig = plt.figure(figsize=(14, 4), constrained_layout=True)
gs = gridspec.GridSpec(5, 1, figure=fig, height_ratios=[0.5, 1, 1, 1, 1], hspace=0.1)
ax_title = fig.add_subplot(gs[0, 0])
ax_m1_aln1 = fig.add_subplot(gs[1, 0], sharex=ax_title)
ax_m1_aln2 = fig.add_subplot(gs[2, 0], sharex=ax_title)
ax_m2_aln1 = fig.add_subplot(gs[3, 0], sharex=ax_title)
ax_m2_aln2 = fig.add_subplot(gs[4, 0], sharex=ax_title)

scan_bar = None
reveal_patches = []
diff_highlight_patches = []
static_text_artists = [] # Holds the main alignment text objects

def setup_plot():
    global scan_bar, reveal_patches, static_text_artists
    static_text_artists = []; reveal_patches = []

    ax_title.set_axis_off(); ax_title.set_xlim(-1, max_len); ax_title.set_ylim(0, 1)
    title_text = f"{MATRIX_NAME_1} (Score: {score_m1}) vs {MATRIX_NAME_2} (Score: {score_m2})"
    ax_title.text(0, 0.5, title_text, fontsize=FONT_SIZE+1, va='center', ha='left')

    axes = [ax_m1_aln1, ax_m1_aln2, ax_m2_aln1, ax_m2_aln2]
    alignments = [aln1_m1, aln2_m1, aln1_m2, aln2_m2]
    labels = [f"{MATRIX_NAME_1} S1:", f"{MATRIX_NAME_1} S2:", f"{MATRIX_NAME_2} S1:", f"{MATRIX_NAME_2} S2:"]
    max_label_len = max(len(lbl) for lbl in labels)

    for i, ax in enumerate(axes):
        ax.set_axis_off()
        padded_label = labels[i].ljust(max_label_len) + " "
        t = ax.text(0, 0.5, padded_label + alignments[i],
                    fontsize=FONT_SIZE, family=FONT_FAMILY, va='center', ha='left',
                    clip_on=True, zorder=1) # Text Layer 1
        static_text_artists.append(t)

        patch = ax.axvspan(-0.5, max_len - 0.5,
                           facecolor=fig.get_facecolor(), alpha=1.0, # Use figure bg color
                           zorder=3, label=f"reveal_{i}", visible=True) # Reveal Patch Layer 3
        reveal_patches.append(patch)

    scan_bar = ax_m1_aln1.axvline(-0.5, color=SCAN_BAR_COLOR, linewidth=3, zorder=4) # Scan Bar Layer 4

def update(frame):
    global diff_highlight_patches, reveal_patches # Need to modify reveal_patches list

    # --- Cap frame/col index early ---
    current_col_idx = min(frame, max_len - 1) # Index for accessing data (0 to max_len-1)
    is_final_pause_frame = (frame >= max_len)  # Check original frame number for end state

    # Position defining the *right* edge of the revealed area / left edge of scan bar
    # Add 0.5 because columns are centered at integers 0, 1, 2...
    reveal_ends_at_pos = current_col_idx + 0.5

    # Scan bar position moves up to the end, then stays
    scan_bar_x_pos = reveal_ends_at_pos
    if is_final_pause_frame:
        scan_bar_x_pos = max_len - 0.5 # Keep bar at the very end during pause

    scan_bar.set_xdata([scan_bar_x_pos, scan_bar_x_pos])

    # --- Clear old highlights ---
    for patch in diff_highlight_patches: patch.remove()
    diff_highlight_patches = []

    # --- Update Reveal Patches (Recreate approach) ---
    new_reveal_patches_for_list = [] # Temp list to hold newly created patches
    for i, old_patch in enumerate(reveal_patches):
         if old_patch: old_patch.remove() # Remove the old patch graphically

         # Define the new coverage area for the white patch
         reveal_patch_xmin = reveal_ends_at_pos # White patch starts where scan bar is
         reveal_patch_xmax = max_len - 0.5   # Covers to the end

         # Only create if there is still area to cover
         current_ax = static_text_artists[i].axes # Get axes from text artist (safer)
         if reveal_patch_xmin < reveal_patch_xmax:
              new_patch = current_ax.axvspan(reveal_patch_xmin, reveal_patch_xmax,
                                             facecolor=fig.get_facecolor(), alpha=1.0,
                                             zorder=3, label=f"reveal_{i}")
              new_reveal_patches_for_list.append(new_patch) # Add to temp list
         # If xmin >= xmax, we don't add a patch, effectively revealing fully

    reveal_patches[:] = new_reveal_patches_for_list # Replace global list content


    # --- Add Difference Highlights ---
    # Only highlight the column just revealed (if not in pause frames)
    if not is_final_pause_frame:
        # current_col_idx is already capped and safe for indexing diff_cols
        if diff_cols[current_col_idx]:
            xmin = current_col_idx - 0.5
            xmax = current_col_idx + 0.5
            axes_to_highlight = [ax_m1_aln1, ax_m1_aln2, ax_m2_aln1, ax_m2_aln2]
            for ax in axes_to_highlight:
                 # Check if axes still exist (paranoid check)
                 if ax.figure is None: continue
                 patch = ax.axvspan(xmin, xmax, facecolor=HIGHLIGHT_DIFF_COLOR, alpha=0.4,
                                    zorder=2) # Highlight Layer 2 (below reveal)
                 diff_highlight_patches.append(patch)


    # --- Return Artists ---
    # Need to return all artists that were potentially modified OR added/removed
    return [scan_bar] + reveal_patches + diff_highlight_patches + static_text_artists


# --- Create and Save Animation ---
setup_plot()
num_anim_frames = max_len + 15
# Make sure blit is False
ani = animation.FuncAnimation(fig, update, frames=num_anim_frames,
                              interval=INTERVAL, blit=False, repeat=False)

print(f"Attempting to save scan comparison animation v4 to {OUTPUT_FILENAME}...")
try:
    ani.save(OUTPUT_FILENAME, writer='pillow', fps=1000/INTERVAL, dpi=120)
    print(f"Animation saved successfully to {OUTPUT_FILENAME}")
except Exception as e:
    print(f"Error saving animation: {e}", file=sys.stderr)
    print("Traceback:", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    print("Ensure 'pillow' is installed ('pip install pillow').", file=sys.stderr)

# plt.show()