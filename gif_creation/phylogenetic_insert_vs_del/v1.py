import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Arrow
import matplotlib.gridspec as gridspec

# --- Configuration ---
OUTPUT_FILENAME = "prank_concept_insertion.gif"
INTERVAL = 300  # Milliseconds between frames (adjust for speed)
FONT_SIZE_SEQ = 10
FONT_SIZE_TEXT = 9
FONT_FAMILY = 'monospace'

# --- Data for Visualization ---
# Simple Tree Structure (A,B sister taxa, C outgroup)
# Coordinates for plotting tree nodes/lines
tree_coords = {
    'A': (0.8, 0.9), 'B': (0.8, 0.7), 'C': (0.8, 0.3),
    'Anc1': (0.5, 0.8), 'Anc2': (0.2, 0.55), 'Root': (0, 0.55)
}
tree_lines = [
    ('Root', 'Anc2'), ('Anc2', 'Anc1'), ('Anc2', 'C'),
    ('Anc1', 'A'), ('Anc1', 'B')
]

# Sequences (Simplified Example)
# State 1: Before Insertion
seq_anc = "AC-GT"
seq_b = "AC-GT"
seq_c = "AC-GT"
seq_a_before = "AC-GT"
# State 2: After Insertion in A
seq_a_after = "ACXGT"
insertion_char = "X"
insertion_idx = 2 # Index where 'X' is inserted

# State 3: Traditional Alignment Result
aln_a_trad = "ACXGT"
aln_b_trad = "AC-GT"
aln_c_trad = "AC-GT"

# State 4: PRANK Explanation (Focus on logic, not result difference here)

# Animation timing (frame numbers)
TIME_INITIAL = 0
TIME_INSERT_STARTS = 10
TIME_INSERT_ENDS = 20
TIME_TRAD_ALIGN_STARTS = 25
TIME_TRAD_ALIGN_ENDS = 40
TIME_PRANK_LOGIC_STARTS = 45
TIME_END = 65 # Total frames = TIME_END + pause

# --- Animation Setup ---
fig = plt.figure(figsize=(10, 6), constrained_layout=True)
gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1], width_ratios=[1, 2])

ax_tree = fig.add_subplot(gs[:, 0])         # Tree on the left
ax_seqs = fig.add_subplot(gs[0, 1])         # Initial Seqs / Event
ax_trad = fig.add_subplot(gs[1, 1])         # Traditional Alignment
ax_prank = fig.add_subplot(gs[2, 1])        # PRANK Logic Explanation

# Artists to manage
dynamic_artists = {} # Dictionary to store artists by name/key

def setup_plot():
    """Setup static plot elements and initial text."""
    global dynamic_artists
    dynamic_artists = {} # Clear artists

    # --- Tree Panel (Same as before) ---
    ax_tree.set_axis_off(); ax_tree.set_xlim(-0.1, 0.9); ax_tree.set_ylim(0.1, 1.0)
    ax_tree.set_title("Phylogenetic Tree")
    for start, end in tree_lines: ax_tree.plot([tree_coords[start][0], tree_coords[end][0]], [tree_coords[start][1], tree_coords[end][1]], 'k-', lw=1.5)
    for name, (x, y) in tree_coords.items(): ax_tree.text(x + 0.03, y, name, va='center', ha='left', fontsize=FONT_SIZE_TEXT)

    # --- Sequence Panel (Same as before) ---
    ax_seqs.set_axis_off(); ax_seqs.set_xlim(0, 1); ax_seqs.set_ylim(0, 1)
    dynamic_artists['seq_a'] = ax_seqs.text(0.1, 0.8, f"Seq A: {seq_a_before}", family=FONT_FAMILY, fontsize=FONT_SIZE_SEQ, va='center')
    dynamic_artists['seq_b'] = ax_seqs.text(0.1, 0.6, f"Seq B: {seq_b}", family=FONT_FAMILY, fontsize=FONT_SIZE_SEQ, va='center')
    dynamic_artists['seq_c'] = ax_seqs.text(0.1, 0.4, f"Seq C: {seq_c}", family=FONT_FAMILY, fontsize=FONT_SIZE_SEQ, va='center')
    dynamic_artists['seq_anc'] = ax_seqs.text(0.1, 0.2, f"Anc1:  {seq_anc}", family=FONT_FAMILY, fontsize=FONT_SIZE_SEQ, va='center', alpha=0.7)
    dynamic_artists['event_text'] = ax_seqs.text(0.5, 0.05, "", family="sans-serif", fontsize=FONT_SIZE_TEXT, ha='center', color='blue')
    dynamic_artists['insertion_flash'] = ax_tree.add_patch(Rectangle((0, 0), 0, 0, color='yellow', alpha=0.0, zorder=-1))

    # --- Traditional Alignment Panel ---
    ax_trad.set_axis_off(); ax_trad.set_xlim(0, 1); ax_trad.set_ylim(0, 1)
    dynamic_artists['trad_title'] = ax_trad.text(0.5, 0.9, "Traditional MSA", ha='center', fontsize=FONT_SIZE_TEXT + 1, visible=False)
    # Create the text artists first
    dynamic_artists['aln_a_trad'] = ax_trad.text(0.1, 0.7, f"A: {aln_a_trad}", family=FONT_FAMILY, fontsize=FONT_SIZE_SEQ, va='center', visible=False)
    dynamic_artists['aln_b_trad'] = ax_trad.text(0.1, 0.5, f"B: {aln_b_trad}", family=FONT_FAMILY, fontsize=FONT_SIZE_SEQ, va='center', visible=False)
    dynamic_artists['aln_c_trad'] = ax_trad.text(0.1, 0.3, f"C: {aln_c_trad}", family=FONT_FAMILY, fontsize=FONT_SIZE_SEQ, va='center', visible=False)
    dynamic_artists['trad_explain'] = ax_trad.text(0.1, 0.1, "", family="sans-serif", fontsize=FONT_SIZE_TEXT, va='top', color='gray', visible=False)

    # **** CORRECTED GAP HIGHLIGHT POSITIONING ****
    # Get the full text content AFTER creating the text artist
    full_aln_b_text = dynamic_artists['aln_b_trad'].get_text()
    full_aln_c_text = dynamic_artists['aln_c_trad'].get_text()
    trad_b_gap_pos = (0, 0) # Default in case gap not found
    trad_c_gap_pos = (0, 0)

    try:
        # Find index of '-' within the full alignment string text
        b_gap_char_index = full_aln_b_text.index('-')
        # Use this index to estimate position
        trad_b_gap_pos = _get_char_pos(dynamic_artists['aln_b_trad'], b_gap_char_index)
    except ValueError:
        print("Warning: Gap character '-' not found in traditional alignment B for highlighting.")

    try:
        c_gap_char_index = full_aln_c_text.index('-')
        trad_c_gap_pos = _get_char_pos(dynamic_artists['aln_c_trad'], c_gap_char_index)
    except ValueError:
        print("Warning: Gap character '-' not found in traditional alignment C for highlighting.")

    # Create highlight patches using the calculated (or default) positions
    char_w = 0.08 # Approximate width for highlight
    dynamic_artists['trad_b_gap_highlight'] = ax_trad.add_patch(Rectangle((trad_b_gap_pos[0]-char_w/2, trad_b_gap_pos[1]-0.1), char_w, 0.2, color='red', alpha=0.0))
    dynamic_artists['trad_c_gap_highlight'] = ax_trad.add_patch(Rectangle((trad_c_gap_pos[0]-char_w/2, trad_c_gap_pos[1]-0.1), char_w, 0.2, color='red', alpha=0.0))
    # **** END CORRECTION ****

    # --- PRANK Logic Panel (Same as before) ---
    ax_prank.set_axis_off(); ax_prank.set_xlim(0, 1); ax_prank.set_ylim(0, 1)
    dynamic_artists['prank_title'] = ax_prank.text(0.5, 0.9, "PRANK Logic", ha='center', fontsize=FONT_SIZE_TEXT + 1, visible=False)
    dynamic_artists['prank_explain1'] = ax_prank.text(0.1, 0.7, "", family="sans-serif", fontsize=FONT_SIZE_TEXT, va='top', color='green', visible=False)
    dynamic_artists['prank_explain2'] = ax_prank.text(0.1, 0.5, "", family="sans-serif", fontsize=FONT_SIZE_TEXT, va='top', color='green', visible=False)
    dynamic_artists['prank_explain3'] = ax_prank.text(0.1, 0.3, "", family="sans-serif", fontsize=FONT_SIZE_TEXT, va='top', color='green', visible=False)
    dynamic_artists['prank_arrow'] = ax_prank.add_patch(Arrow(0.8, 0.6, -0.6, -0.1, width=0.05, color='green', alpha=0.0))

def _get_char_pos(text_artist, char_index):
    """Helper to estimate character position - VERY APPROXIMATE"""
    # This is difficult in Matplotlib. Use relative positioning based on string length.
    full_text = text_artist.get_text()
    total_len = len(full_text)
    x_start, _ = text_artist.get_position() # Assumes ha='left' or similar start point
    # Estimate relative position (needs adjustment based on actual rendering)
    relative_pos = (char_index + 0.5) / total_len # Center of char
    # Map relative pos to axes coords (assuming text roughly fills axes width 0.1 to 0.9)
    axes_width = 0.8
    est_x = x_start + relative_pos * axes_width
    est_y = text_artist.get_position()[1] # Y position is simpler
    return (est_x, est_y)


def update(frame):
    """Update function for animation."""

    # --- Phase 1: Initial State ---
    if frame < TIME_INSERT_STARTS:
        dynamic_artists['event_text'].set_text("Sequences diverge...")

    # --- Phase 2: Insertion Event ---
    elif frame < TIME_INSERT_ENDS:
        # Update text
        dynamic_artists['seq_a'].set_text(f"Seq A: {seq_a_after}")
        dynamic_artists['seq_a'].set_color('blue') # Highlight changed seq
        dynamic_artists['event_text'].set_text(f"'{insertion_char}' inserted in lineage A!")
        # Flash effect on branch A
        flash_alpha = 0.6 * (1 - abs(frame - (TIME_INSERT_STARTS + TIME_INSERT_ENDS)/2) / ((TIME_INSERT_ENDS - TIME_INSERT_STARTS)/2))
        branch_start = tree_coords['Anc1']
        branch_end = tree_coords['A']
        mid_x = (branch_start[0] + branch_end[0]) / 2
        mid_y = (branch_start[1] + branch_end[1]) / 2
        w = abs(branch_start[0] - branch_end[0]) + 0.05
        h = abs(branch_start[1] - branch_end[1]) + 0.05
        x0 = min(branch_start[0], branch_end[0]) - 0.025
        y0 = min(branch_start[1], branch_end[1]) - 0.025
        dynamic_artists['insertion_flash'].set_bounds(x0, y0, w, h)
        dynamic_artists['insertion_flash'].set_alpha(flash_alpha)

    # --- Phase 3: Traditional Alignment ---
    elif frame < TIME_TRAD_ALIGN_ENDS:
        # Reset previous highlights
        dynamic_artists['seq_a'].set_color('black')
        dynamic_artists['insertion_flash'].set_alpha(0.0)
        dynamic_artists['event_text'].set_text("")
        # Show traditional alignment panel
        dynamic_artists['trad_title'].set_visible(True)
        dynamic_artists['aln_a_trad'].set_visible(True)
        dynamic_artists['aln_b_trad'].set_visible(True)
        dynamic_artists['aln_c_trad'].set_visible(True)
        dynamic_artists['trad_explain'].set_visible(True)
        dynamic_artists['trad_explain'].set_text("Treats insertion/gap symmetrically.\nImplies B and C might lack 'X'.")
        # Highlight the inferred gaps in B and C
        dynamic_artists['trad_b_gap_highlight'].set_alpha(0.4)
        dynamic_artists['trad_c_gap_highlight'].set_alpha(0.4)


    # --- Phase 4: PRANK Logic Explanation ---
    elif frame < TIME_END:
        # Keep traditional visible, add PRANK panel info
        dynamic_artists['prank_title'].set_visible(True)
        dynamic_artists['prank_explain1'].set_visible(True)
        dynamic_artists['prank_explain1'].set_text("PRANK uses the tree!")
        dynamic_artists['prank_explain2'].set_visible(True)
        dynamic_artists['prank_explain2'].set_text(f"Sees '{insertion_char}' is ONLY in A's lineage\n(Ancestor has '-' at that position).")
        dynamic_artists['prank_explain3'].set_visible(True)
        dynamic_artists['prank_explain3'].set_text("Avoids penalizing gaps in B & C\nas if they were homologous deletions.")
        # Show arrow pointing towards ancestor node or branch
        dynamic_artists['prank_arrow'].set_alpha(0.7)

    # --- Hold Final State ---
    else:
        pass # Keep last state visible


    # Return all artists that might have changed
    return list(dynamic_artists.values())

# --- Create and Save Animation ---
setup_plot()
num_anim_frames = TIME_END + 20 # Add pause at the end

ani = animation.FuncAnimation(fig, update, frames=num_anim_frames,
                              interval=INTERVAL, blit=False, repeat=False) # Use blit=False

print(f"Attempting to save PRANK concept animation to {OUTPUT_FILENAME}...")
try:
    ani.save(OUTPUT_FILENAME, writer='pillow', fps=1000/INTERVAL, dpi=100)
    print(f"Animation saved successfully to {OUTPUT_FILENAME}")
except Exception as e:
    import traceback
    print(f"Error saving animation: {e}", file=sys.stderr)
    print("Traceback:", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    print("Ensure 'pillow' is installed ('pip install pillow').", file=sys.stderr)

# plt.show()