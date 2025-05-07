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
OUTPUT_FILENAME = "needleman_wunsch_rearranged.gif"
INTERVAL = 800 # Milliseconds between frames

# --- Helper Functions ---
def get_score(char1, char2):
    """Gets the score for comparing two characters."""
    if char1 == char2:
        return MATCH_SCORE
    else:
        return MISMATCH_SCORE

def format_sequence_text(ax, seq, title, highlight_index=-1):
    """Formats sequence text for display, highlighting one character."""
    ax.clear() # Clear previous text/state
    ax.set_axis_off()
    ax.set_title(title)
    ax.set_xlim(-0.5, len(seq) + 0.5) # Adjust xlim for better spacing
    ax.set_ylim(0, 1)
    texts = []
    for i, char in enumerate(seq):
        color = 'red' if i == highlight_index else 'black'
        t = ax.text(i + 0.5, 0.5, char, ha='center', va='center', fontsize=12, color=color)
        texts.append(t)
    return texts


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
# Adjust figsize and layout parameters
fig = plt.figure(figsize=(11, 7), constrained_layout=True)
# GridSpec: 3 rows, 2 columns. Right column narrower. Give calc text more height.
gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[2, 1, 1], width_ratios=[3, 2],
                       hspace=0.4, wspace=0.3) # Added spacing

# Assign axes based on the new layout
ax_table = fig.add_subplot(gs[:, 0])   # Table takes the whole left column
ax_calc = fig.add_subplot(gs[0, 1])   # Calculation text takes top-right
ax_seq1 = fig.add_subplot(gs[1, 1])   # Sequence 1 takes middle-right
ax_seq2 = fig.add_subplot(gs[2, 1])   # Sequence 2 takes bottom-right


# Store artists that need updating
cell_texts = {}
cell_highlights = []
seq1_text_artists = []
seq2_text_artists = []
calc_text_artist = None

# --- Plotting Functions ---
def setup_plot():
    """Initial setup of the plot elements."""
    global calc_text_artist, seq1_text_artists, seq2_text_artists

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
                                    ha='center', va='center', fontsize=10, wrap=True,
                                    transform=ax_calc.transAxes) # Use axes coordinates

    # --- Sequence Axes (ax_seq1, ax_seq2) ---
    seq1_text_artists = format_sequence_text(ax_seq1, SEQ1, f"Sequence 1")
    seq2_text_artists = format_sequence_text(ax_seq2, SEQ2, f"Sequence 2")


# --- Animation Update Function ---
current_frame = 0
total_frames = n * m

def update(frame):
    """Update function called for each frame of the animation."""
    global current_frame, cell_highlights, calc_text_artist, seq1_text_artists, seq2_text_artists

    # Calculate current row and column (iterate row by row)
    if frame >= total_frames: # Avoid index errors after last frame
        calc_text_artist.set_text("Finished!")
        # Clear highlights
        for patch in cell_highlights: patch.remove()
        cell_highlights.clear()
        # Re-render sequences without highlights
        seq1_text_artists = format_sequence_text(ax_seq1, SEQ1, f"Sequence 1", -1)
        seq2_text_artists = format_sequence_text(ax_seq2, SEQ2, f"Sequence 2", -1)
        return list(cell_texts.values()) + seq1_text_artists + seq2_text_artists + [calc_text_artist]


    i = frame // m + 1
    j = frame % m + 1

    # Clear previous highlights from table
    for patch in cell_highlights:
        patch.remove()
    cell_highlights.clear()

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

    # Highlight source cells and current cell in the table
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

    # Update sequence character highlights by re-rendering the sequence text
    seq1_idx_to_highlight = i - 1 if i > 0 else -1
    seq2_idx_to_highlight = j - 1 if j > 0 else -1
    seq1_text_artists = format_sequence_text(ax_seq1, SEQ1, f"Sequence 1", seq1_idx_to_highlight)
    seq2_text_artists = format_sequence_text(ax_seq2, SEQ2, f"Sequence 2", seq2_idx_to_highlight)


    current_frame += 1

    # Return list of artists that were modified
    # Note: Since we recreate seq text artists, we return the new lists
    artists = list(cell_texts.values()) + cell_highlights + seq1_text_artists + seq2_text_artists + [calc_text_artist]
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