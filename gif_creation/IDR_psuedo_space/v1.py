import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from scipy.interpolate import splprep, splev

# --- Parameters ---
N_POINTS_SPLINE1 = 4  # n: Number of points on spline 1
M_POINTS_SPLINE2 = 5  # m: Number of points on spline 2
P_TRIANGLES_PER_PAIR = 3 # P: Number of random triangles per reference pair
SPLINE_RESOLUTION = 100 # For drawing smooth spline curves
INTERVAL_MS = 500       # Milliseconds between frames (slower for clarity)

# --- 1. Generate Spline Data ---
control_points1 = np.array([
    [0.5, 2.0], [1.5, 3.5], [3.0, 3.0], [4.5, 1.5]
])
control_points2 = np.array([
    [1.0, -0.5], [2.5, -1.8], [4.0, -1.5], [5.5, -0.2], [6.5, 0.8]
])

def generate_spline(control_points, num_target_points, resolution):
    control_points_t = np.array(control_points, dtype=float).T
    tck, u = splprep(control_points_t, s=0, k=min(3, len(control_points)-1))
    u_fine = np.linspace(u.min(), u.max(), resolution)
    x_fine, y_fine = splev(u_fine, tck)
    u_points = np.linspace(u.min(), u.max(), num_target_points)
    x_points, y_points = splev(u_points, tck)
    return x_fine, y_fine, np.column_stack((x_points, y_points))

x1_curve, y1_curve, spline1_eval_points = generate_spline(control_points1, N_POINTS_SPLINE1, SPLINE_RESOLUTION)
x2_curve, y2_curve, spline2_eval_points = generate_spline(control_points2, M_POINTS_SPLINE2, SPLINE_RESOLUTION)

# Calculate max possible leg length for scaling triangle plot
max_dist_s1 = 0
for i in range(N_POINTS_SPLINE1):
    for k in range(N_POINTS_SPLINE1):
        max_dist_s1 = max(max_dist_s1, np.linalg.norm(spline1_eval_points[i] - spline1_eval_points[k]))
max_dist_s2 = 0
for i in range(M_POINTS_SPLINE2):
    for k in range(M_POINTS_SPLINE2):
        max_dist_s2 = max(max_dist_s2, np.linalg.norm(spline2_eval_points[i] - spline2_eval_points[k]))
MAX_LEG_LENGTH_WORLD = max(max_dist_s1, max_dist_s2, 1.0) # Ensure not zero

# --- 2. Animation Setup ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Spline Point Distances and Derived Triangles", fontsize=16)

# --- Left Plot: Splines & Distances ---
ax_spline = axes[0]
ax_spline.plot(x1_curve, y1_curve, 'b-', alpha=0.5, label=f'Spline 1')
ax_spline.plot(x2_curve, y2_curve, 'g-', alpha=0.5, label=f'Spline 2')
s1_pts_plot = ax_spline.plot(spline1_eval_points[:, 0], spline1_eval_points[:, 1], 'bo', markersize=7, label=f'S1 Points ({N_POINTS_SPLINE1})')[0]
s2_pts_plot = ax_spline.plot(spline2_eval_points[:, 0], spline2_eval_points[:, 1], 'go', markersize=7, label=f'S2 Points ({M_POINTS_SPLINE2})')[0]
connection_line, = ax_spline.plot([], [], 'k--', lw=1.5, alpha=0.8, label='Ref Connection')

# Text for distances on spline 1
s1_dist_texts = [ax_spline.text(0,0,'', fontsize=8, color='darkblue', ha='right', va='bottom') for _ in range(N_POINTS_SPLINE1)]
# Text for distances on spline 2
s2_dist_texts = [ax_spline.text(0,0,'', fontsize=8, color='darkgreen', ha='left', va='top') for _ in range(M_POINTS_SPLINE2)]

# Highlights for reference points and triangle base points
ref_pt1_highlight = ax_spline.plot([], [], 'o', mec='black', mfc='none', mew=2, markersize=12, label='Ref Pt S1')[0]
ref_pt2_highlight = ax_spline.plot([], [], 'o', mec='black', mfc='none', mew=2, markersize=12, label='Ref Pt S2')[0]
tri_base_pt1_highlight = ax_spline.plot([], [], 'o', mec='red', mfc='none', mew=2, markersize=14, label='Triangle Base S1')[0]
tri_base_pt2_highlight = ax_spline.plot([], [], 'o', mec='red', mfc='none', mew=2, markersize=14, label='Triangle Base S2')[0]

ax_spline.set_xlabel("X")
ax_spline.set_ylabel("Y")
ax_spline.legend(fontsize='small', loc='upper right')
ax_spline.axis('equal')
ax_spline.grid(True, linestyle=':', alpha=0.6)

# --- Right Plot: Right Triangle ---
ax_triangle = axes[1]
ax_triangle.set_xlim(-0.1 * MAX_LEG_LENGTH_WORLD, 1.1 * MAX_LEG_LENGTH_WORLD)
ax_triangle.set_ylim(-0.1 * MAX_LEG_LENGTH_WORLD, 1.1 * MAX_LEG_LENGTH_WORLD)
ax_triangle.set_xlabel("Leg A (from Spline 1 distances)")
ax_triangle.set_ylabel("Leg B (from Spline 2 distances)")
ax_triangle.set_title("Derived Right Triangle")
ax_triangle.set_aspect('equal')
ax_triangle.grid(True, linestyle=':', alpha=0.6)

triangle_patch = patches.Polygon([[0,0], [0,0], [0,0]], closed=True, edgecolor='purple', facecolor='lavender', alpha=0.7)
ax_triangle.add_patch(triangle_patch)

leg_a_text = ax_triangle.text(0.5, -0.08, '', transform=ax_triangle.transAxes, ha='center', va='top', fontsize=9)
leg_b_text = ax_triangle.text(-0.08, 0.5, '', transform=ax_triangle.transAxes, ha='right', va='center', fontsize=9, rotation=90)
hyp_text = ax_triangle.text(0.5, 1.02, '', transform=ax_triangle.transAxes, ha='center', va='bottom', fontsize=10, color='darkred', weight='bold')
info_text = fig.text(0.5, 0.01, '', ha='center', va='bottom', fontsize=9)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# --- 3. Animation Logic ---
total_ref_pairs = N_POINTS_SPLINE1 * M_POINTS_SPLINE2
total_animation_frames = total_ref_pairs * P_TRIANGLES_PER_PAIR

def animate(frame):
    # Determine current reference pair and triangle iteration
    ref_pair_idx = frame // P_TRIANGLES_PER_PAIR
    triangle_iter_p = frame % P_TRIANGLES_PER_PAIR

    ref_idx1 = ref_pair_idx // M_POINTS_SPLINE2
    ref_idx2 = ref_pair_idx % M_POINTS_SPLINE2

    ref_pt1 = spline1_eval_points[ref_idx1]
    ref_pt2 = spline2_eval_points[ref_idx2]

    # --- Update Left Plot (Splines & Distances) ---
    connection_line.set_data([ref_pt1[0], ref_pt2[0]], [ref_pt1[1], ref_pt2[1]])
    ref_pt1_highlight.set_data([ref_pt1[0]], [ref_pt1[1]])
    ref_pt2_highlight.set_data([ref_pt2[0]], [ref_pt2[1]])

    # Update distance texts for spline 1 points relative to ref_pt1
    for i, pt in enumerate(spline1_eval_points):
        dist = np.linalg.norm(pt - ref_pt1)
        s1_dist_texts[i].set_position((pt[0] - 0.1, pt[1] - 0.1)) # Offset for clarity
        s1_dist_texts[i].set_text(f'{dist:.1f}')
        s1_dist_texts[i].set_visible(True)

    # Update distance texts for spline 2 points relative to ref_pt2
    for i, pt in enumerate(spline2_eval_points):
        dist = np.linalg.norm(pt - ref_pt2)
        s2_dist_texts[i].set_position((pt[0] + 0.1, pt[1] + 0.1)) # Offset for clarity
        s2_dist_texts[i].set_text(f'{dist:.1f}')
        s2_dist_texts[i].set_visible(True)

    # --- Update Right Plot (Triangle) & Triangle Base Highlights ---
    # Randomly select points for triangle legs
    # Seed with frame to make random choices deterministic per frame for consistency if animation is re-run
    # but different for each P_TRIANGLES_PER_PAIR iteration
    np.random.seed(frame)
    tri_base_idx1 = np.random.randint(0, N_POINTS_SPLINE1)
    tri_base_idx2 = np.random.randint(0, M_POINTS_SPLINE2)

    tri_base_pt1 = spline1_eval_points[tri_base_idx1]
    tri_base_pt2 = spline2_eval_points[tri_base_idx2]

    # Highlight these chosen base points on the left plot
    tri_base_pt1_highlight.set_data([tri_base_pt1[0]], [tri_base_pt1[1]])
    tri_base_pt2_highlight.set_data([tri_base_pt2[0]], [tri_base_pt2[1]])
    tri_base_pt1_highlight.set_visible(True)
    tri_base_pt2_highlight.set_visible(True)

    leg_A_val = np.linalg.norm(tri_base_pt1 - ref_pt1)
    leg_B_val = np.linalg.norm(tri_base_pt2 - ref_pt2)
    hyp_val = np.sqrt(leg_A_val**2 + leg_B_val**2)

    # Update triangle patch vertices (scaled to fit MAX_LEG_LENGTH_WORLD)
    # Scale for display, but labels show true values
    scale_factor = 0.9 * MAX_LEG_LENGTH_WORLD / max(MAX_LEG_LENGTH_WORLD, leg_A_val, leg_B_val, 1e-6) # Avoid div by zero
    
    # Ensure scaled legs are not excessively tiny if actual values are small compared to max
    display_leg_A = max(leg_A_val * scale_factor, 0.01 * MAX_LEG_LENGTH_WORLD if leg_A_val > 1e-6 else 0)
    display_leg_B = max(leg_B_val * scale_factor, 0.01 * MAX_LEG_LENGTH_WORLD if leg_B_val > 1e-6 else 0)


    triangle_patch.set_xy([
        [0, 0],
        [display_leg_A, 0],
        [0, display_leg_B]
    ])

    leg_a_text.set_text(f'Leg A (S1 Pt{tri_base_idx1} to Ref{ref_idx1}): {leg_A_val:.2f}')
    leg_a_text.set_position((display_leg_A / 2, -0.08 * MAX_LEG_LENGTH_WORLD / (ax_triangle.get_ylim()[1]-ax_triangle.get_ylim()[0]))) # Adjust y pos based on scale

    leg_b_text.set_text(f'Leg B (S2 Pt{tri_base_idx2} to Ref{ref_idx2}): {leg_B_val:.2f}')
    leg_b_text.set_position((-0.08 * MAX_LEG_LENGTH_WORLD / (ax_triangle.get_xlim()[1]-ax_triangle.get_xlim()[0]), display_leg_B / 2)) # Adjust x pos

    hyp_text.set_text(f'Hypotenuse: {hyp_val:.2f}')
    
    current_time_cd = (current_time + np.timedelta64(frame * INTERVAL_MS, 'ms')).item().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + " CDT"
    info_text.set_text(f"Ref Pair: (S1 Pt {ref_idx1}, S2 Pt {ref_idx2}) - Triangle Iter: {triangle_iter_p+1}/{P_TRIANGLES_PER_PAIR} | {current_time_cd}")


    return (connection_line, ref_pt1_highlight, ref_pt2_highlight,
            tri_base_pt1_highlight, tri_base_pt2_highlight, triangle_patch,
            leg_a_text, leg_b_text, hyp_text, info_text, *s1_dist_texts, *s2_dist_texts)

# --- 4. Create and Save Animation ---
FPS_GIF = max(1, int(1000 / INTERVAL_MS))
print(f"Total animation frames: {total_animation_frames}")
print(f"Interval: {INTERVAL_MS}ms, GIF FPS: {FPS_GIF}")

# Get current time for info text
current_time = np.datetime64('now') # Using system's current time. For specific CDT, would need timezone conversion.
# This is a simplification; for true CDT display, pytz or dateutil would be better.
# Using it just to show time progression.

ani = animation.FuncAnimation(fig, animate, frames=total_animation_frames,
                              interval=INTERVAL_MS, blit=False, repeat=False)
# blit=False is safer with many artists and text objects

gif_filename = 'spline_triangle_distances.gif'
print(f"Attempting to save animation to {gif_filename}...")
try:
    ani.save(gif_filename, writer='pillow', fps=FPS_GIF)
    print(f"Successfully saved animation to {gif_filename}")
except Exception as e:
    print(f"Error saving animation: {e}")
    print("Ensure 'pillow' is installed (pip install Pillow).")

plt.close(fig)