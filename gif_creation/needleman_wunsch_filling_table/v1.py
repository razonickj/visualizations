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
OUTPUT_FILENAME = "needleman_wunsch.gif"
INTERVAL = 800 # Milliseconds between frames

# --- Helper Functions ---
def get_score(char1, char2):
    """Gets the score for comparing two characters."""
    if char1 == char2:
        return MATCH_SCORE
    else:
        return MISMATCH_SCORE

def format_sequence_text(seq, highlight_index=-1):
    """Formats sequence text for display, highlighting one character."""
    texts = []
    colors = []
    for i, char in enumerate(seq):
        texts.append(char)
        colors.append('red' if i == highlight_index else 'black')
    return texts, colors

# --- Initialization ---
n = len(SEQ1)
m = len(SEQ2)

# Initialize score matrix and traceback matrix
score_matrix = np.zeros((n + 1, m + 1), dtype=int)
traceback_matrix = np.full((n + 1, m + 1), '', dtype=object) # Store arrows or source

# Fill the first row and column based on gap penalties
for i in range(1, n + 1):
    score_matrix[i, 0] = score_matrix[i-1, 0] + GAP_PENALTY
    traceback_matrix[i, 0] = '↑' # Up
for j in range(1, m + 1):
    score_matrix[0, j] = score_matrix[0, j-1] + GAP_PENALTY
    traceback_matrix[0, j] = '←' # Left

traceback_matrix[0, 0] = ' '

# --- Animation Setup ---
fig = plt.figure(figsize=(12, 7), constrained_layout=True)
gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 5, 1], width_ratios=[3, 1])

ax_table = fig.add_subplot(gs[0:2, 0]) # Main NW Table spans 2 rows
ax_calc = fig.add_subplot(gs[0, 1])   # Calculation text
ax_seq1 = fig.add_subplot(gs[2, 0])   # Sequence 1 below table
ax_seq2 = fig.add_subplot(gs[2, 1])   # Sequence 2 below calc text (adjust layout)

# Store artists that need updating
cell_texts = {}
cell_highlights = []
seq1_texts = []
seq2_texts = []
calc_text_artist = None

# --- Plotting Functions ---
def setup_plot():
    """Initial setup of the plot elements."""
    global calc_text_artist, seq1_texts, seq2_texts

    # --- Main Table Axis (ax_table) ---
    ax_table.set_xticks(np.arange(m + 1) - 0.5, minor=True)
    ax_table.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax_table.set_xticks(np.arange(m + 1))
    ax_table.set_yticks(np.arange(n + 1))

    # Add sequence labels to the main table
    ax_table.set_xticklabels([''] + list(SEQ2))
    ax_table.set_yticklabels([''] + list(SEQ1))
    ax_table.xaxis.tick_top()
    ax_table.xaxis.set_label_position('top')
    ax_table.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax_table.set_xlim(-0.5, m + 0.5)
    ax_table.set_ylim(n + 0.5, -0.5) # Inverted y-axis
    ax_table.tick_params(axis='both', which='major', length=0) # Hide major ticks
    ax_table.set_title("Needleman-Wunsch Matrix Filling")

    # Initialize cell text artists
    for r in range(n + 1):
        for c in range(m + 1):
            val = score_matrix[r, c] if (r == 0 or c == 0) else ""
            trace = traceback_matrix[r,c] if (r == 0 or c == 0) else ""
            cell_texts[(r, c)] = ax_table.text(c, r, f"{val}\n{trace}",
                                               ha='center', va='center', fontsize=9)

    # --- Calculation Text Axis (ax_calc) ---
    ax_calc.set_axis_off()
    ax_calc.set_title("Calculation")
    calc_text_artist = ax_calc.text(0.5, 0.5, "Initializing...",
                                    ha='center', va='center', fontsize=10, wrap=True)

    # --- Sequence 1 Axis (ax_seq1) ---
    ax_seq1.set_axis_off()
    ax_seq1.set_title(f"Sequence 1: {SEQ1}")
    ax_seq1.set_xlim(0, len(SEQ1))
    ax_seq1.set_ylim(0, 1)
    texts, colors = format_sequence_text(SEQ1)
    for idx, (char, color) in enumerate(zip(texts, colors)):
        t = ax_seq1.text(idx + 0.5, 0.5, char, ha='center', va='center', fontsize=12, color=color)
        seq1_texts.append(t)

    # --- Sequence 2 Axis (ax_seq2) ---
    ax_seq2.set_axis_off()
    ax_seq2.set_title(f"Sequence 2: {SEQ2}")
    ax_seq2.set_xlim(0, len(SEQ2))
    ax_seq2.set_ylim(0, 1)
    texts, colors = format_sequence_text(SEQ2)
    for idx, (char, color) in enumerate(zip(texts, colors)):
        t = ax_seq2.text(idx + 0.5, 0.5, char, ha='center', va='center', fontsize=12, color=color)
        seq2_texts.append(t)

# --- Animation Update Function ---
current_frame = 0
total_frames = n * m

def update(frame):
    """Update function called for each frame of the animation."""
    global current_frame, cell_highlights, calc_text_artist

    # Calculate current row and column (iterate row by row)
    if frame >= total_frames: # Avoid index errors after last frame
        # Optionally add final traceback visualization here
        calc_text_artist.set_text("Finished!")
        # Clear highlights
        for patch in cell_highlights: patch.remove()
        cell_highlights.clear()
        for txt in seq1_texts: txt.set_color('black')
        for txt in seq2_texts: txt.set_color('black')
        return list(cell_texts.values()) + seq1_texts + seq2_texts + [calc_text_artist]


    i = frame // m + 1
    j = frame % m + 1

    # Clear previous highlights
    for patch in cell_highlights:
        patch.remove()
    cell_highlights.clear()
    for txt in seq1_texts: txt.set_color('black')
    for txt in seq2_texts: txt.set_color('black')


    # --- Calculate Score for current cell (i, j) ---
    match_val = get_score(SEQ1[i-1], SEQ2[j-1])
    diag_score = score_matrix[i-1, j-1] + match_val
    up_score = score_matrix[i-1, j] + GAP_PENALTY
    left_score = score_matrix[i, j-1] + GAP_PENALTY

    scores = [diag_score, up_score, left_score]
    max_score = max(scores)
    score_matrix[i, j] = max_score

    # Determine traceback
    trace = ""
    if max_score == diag_score:
        trace = '↖' # Diagonal
    elif max_score == up_score:
        trace = '↑' # Up
    else: # max_score == left_score
        trace = '←' # Left
    # Could add logic here if multiple paths give the same max score
    traceback_matrix[i, j] = trace

    # --- Update Visualization ---

    # Update calculation text
    calc_str = (f"Cell ({i}, {j}):\n"
                f"Match/Mismatch: {SEQ1[i-1]} vs {SEQ2[j-1]} -> {match_val}\n"
                f"Diag ({i-1},{j-1}): {score_matrix[i-1, j-1]} + {match_val} = {diag_score}\n"
                f"Up   ({i-1},{j}): {score_matrix[i-1, j]} + {GAP_PENALTY} = {up_score}\n"
                f"Left ({i},{j-1}): {score_matrix[i, j-1]} + {GAP_PENALTY} = {left_score}\n"
                f"Max = {max_score} -> Trace: {trace}")
    calc_text_artist.set_text(calc_str)

    # Highlight source cells and current cell
    sources = [(i-1, j-1), (i-1, j), (i, j-1)]
    colors = ['lightblue', 'lightgreen', 'lightcoral'] # Diag, Up, Left colors
    for idx, (r, c) in enumerate(sources):
         rect = Rectangle((c - 0.5, r - 0.5), 1, 1, linewidth=1, edgecolor='none', facecolor=colors[idx], alpha=0.4, zorder=-1)
         ax_table.add_patch(rect)
         cell_highlights.append(rect)

    # Highlight current cell with a border
    current_rect = Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=2, edgecolor='red', facecolor='yellow', alpha=0.6, zorder=0)
    ax_table.add_patch(current_rect)
    cell_highlights.append(current_rect)

    # Update cell text with score and traceback
    cell_texts[(i, j)].set_text(f"{max_score}\n{trace}")
    cell_texts[(i, j)].set_zorder(1) # Ensure text is above highlight

    # Highlight sequence characters
    if i > 0:
        seq1_texts[i-1].set_color('red')
    if j > 0:
        seq2_texts[j-1].set_color('red')

    current_frame += 1

    # Return list of artists that were modified
    artists = list(cell_texts.values()) + cell_highlights + seq1_texts + seq2_texts + [calc_text_artist]
    return artists

# --- Create and Save Animation ---
setup_plot()

# Add +1 to frames to show the final state briefly
ani = animation.FuncAnimation(fig, update, frames=total_frames + 1,
                              interval=INTERVAL, blit=True, repeat=False)

print(f"Attempting to save animation to {OUTPUT_FILENAME}...")
try:
    # You might need to install pillow: pip install pillow
    # Or specify another writer like 'imagemagick' if installed
    ani.save(OUTPUT_FILENAME, writer='pillow', fps=1000/INTERVAL)
    print(f"Animation saved successfully to {OUTPUT_FILENAME}")
except Exception as e:
    print(f"Error saving animation: {e}", file=sys.stderr)
    print("Ensure you have 'pillow' installed ('pip install pillow').", file=sys.stderr)
    print("Alternatively, try installing ImageMagick and use writer='imagemagick'.", file=sys.stderr)

# To display the plot interactively (optional, remove if running non-interactively)
# plt.show()