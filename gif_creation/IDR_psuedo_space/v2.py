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
INTERVAL_MS = 1000       # Milliseconds between frames

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

max_dist_s1 = 0
for i in range(N_POINTS_SPLINE1):
    for k in range(N_POINTS_SPLINE1):
        max_dist_s1 = max(max_dist_s1, np.linalg.norm(spline1_eval_points[i] - spline1_eval_points[k]))
max_dist_s2 = 0
for i in range(M_POINTS_SPLINE2):
    for k in range(M_POINTS_SPLINE2):
        max_dist_s2 = max(max_dist_s2, np.linalg.norm(spline2_eval_points[i] - spline2_eval_points[k]))
MAX_LEG_LENGTH_WORLD = max(max_dist_s1, max_dist_s2, 1.0)

# --- 2. Animation Setup ---
fig, axes = plt.subplots(1, 2, figsize=(15, 6)) # Adjusted figsize slightly
# fig_title_artist will hold the Text object for the suptitle
fig_title_artist = fig.suptitle("Hypotenuse: Calculating...", fontsize=16)

# --- Left Plot: Splines & Distances ---
ax_spline = axes[0]
ax_spline.plot(x1_curve, y1_curve, 'b-', alpha=0.5)
ax_spline.plot(x2_curve, y2_curve, 'g-', alpha=0.5)
s1_pts_plot = ax_spline.plot(spline1_eval_points[:, 0], spline1_eval_points[:, 1], 'bo', markersize=7)[0]
s2_pts_plot = ax_spline.plot(spline2_eval_points[:, 0], spline2_eval_points[:, 1], 'go', markersize=7)[0]
connection_line, = ax_spline.plot([], [], 'k--', lw=1.5, alpha=0.8)

# Removed s1_dist_texts and s2_dist_texts

ref_pt1_highlight = ax_spline.plot([], [], 'o', mec='black', mfc='none', mew=2, markersize=12)[0]
ref_pt2_highlight = ax_spline.plot([], [], 'o', mec='black', mfc='none', mew=2, markersize=12)[0]
tri_base_pt1_highlight = ax_spline.plot([], [], 'o', mec='red', mfc='none', mew=2, markersize=14)[0]
tri_base_pt2_highlight = ax_spline.plot([], [], 'o', mec='red', mfc='none', mew=2, markersize=14)[0]

ax_spline.set_xlabel("X")
ax_spline.set_ylabel("Y")
# ax_spline.legend(fontsize='small', loc='upper right') # Legend removed
ax_spline.axis('equal')
ax_spline.grid(True, linestyle=':', alpha=0.6)
ax_spline.set_title("Spline Connections")


# --- Right Plot: Right Triangle ---
ax_triangle = axes[1]
ax_triangle.set_xlim(-0.1 * MAX_LEG_LENGTH_WORLD, 1.1 * MAX_LEG_LENGTH_WORLD)
ax_triangle.set_ylim(-0.1 * MAX_LEG_LENGTH_WORLD, 1.1 * MAX_LEG_LENGTH_WORLD)
ax_triangle.set_xlabel("Scaled Leg A value") # Clarified scaled
ax_triangle.set_ylabel("Scaled Leg B value") # Clarified scaled
ax_triangle.set_title("Derived Right Triangle")
ax_triangle.set_aspect('equal')
ax_triangle.grid(True, linestyle=':', alpha=0.6)

triangle_patch = patches.Polygon([[0,0], [0,0], [0,0]], closed=True, edgecolor='purple', facecolor='lavender', alpha=0.7)
ax_triangle.add_patch(triangle_patch)

# Removed leg_a_text, leg_b_text, hyp_text from here (hypotenuse is now suptitle)
# Removed info_text from figure

plt.tight_layout(rect=[0, 0.03, 1, 0.92]) # Adjusted rect for suptitle space

# --- 3. Animation Logic ---
total_ref_pairs = N_POINTS_SPLINE1 * M_POINTS_SPLINE2
total_animation_frames = total_ref_pairs * P_TRIANGLES_PER_PAIR

def animate(frame):
    ref_pair_idx = frame // P_TRIANGLES_PER_PAIR
    # triangle_iter_p = frame % P_TRIANGLES_PER_PAIR # Not directly used in display now

    ref_idx1 = ref_pair_idx // M_POINTS_SPLINE2
    ref_idx2 = ref_pair_idx % M_POINTS_SPLINE2

    ref_pt1 = spline1_eval_points[ref_idx1]
    ref_pt2 = spline2_eval_points[ref_idx2]

    # --- Update Left Plot (Splines & Distances) ---
    connection_line.set_data([ref_pt1[0], ref_pt2[0]], [ref_pt1[1], ref_pt2[1]])
    ref_pt1_highlight.set_data([ref_pt1[0]], [ref_pt1[1]])
    ref_pt2_highlight.set_data([ref_pt2[0]], [ref_pt2[1]])

    # Removed distance text updates

    # --- Update Right Plot (Triangle) & Triangle Base Highlights ---
    np.random.seed(frame) # Keep for deterministic random choices per frame
    tri_base_idx1 = np.random.randint(0, N_POINTS_SPLINE1)
    tri_base_idx2 = np.random.randint(0, M_POINTS_SPLINE2)

    tri_base_pt1 = spline1_eval_points[tri_base_idx1]
    tri_base_pt2 = spline2_eval_points[tri_base_idx2]

    tri_base_pt1_highlight.set_data([tri_base_pt1[0]], [tri_base_pt1[1]])
    tri_base_pt2_highlight.set_data([tri_base_pt2[0]], [tri_base_pt2[1]])
    tri_base_pt1_highlight.set_visible(True)
    tri_base_pt2_highlight.set_visible(True)

    leg_A_val = np.linalg.norm(tri_base_pt1 - ref_pt1)
    leg_B_val = np.linalg.norm(tri_base_pt2 - ref_pt2)
    hyp_val = np.sqrt(leg_A_val**2 + leg_B_val**2)

    # Update triangle patch vertices (scaled for display)
    scale_factor = 0.9 * MAX_LEG_LENGTH_WORLD / max(MAX_LEG_LENGTH_WORLD, leg_A_val, leg_B_val, 1e-6)
    display_leg_A = max(leg_A_val * scale_factor, 0.01 * MAX_LEG_LENGTH_WORLD if leg_A_val > 1e-6 else 0)
    display_leg_B = max(leg_B_val * scale_factor, 0.01 * MAX_LEG_LENGTH_WORLD if leg_B_val > 1e-6 else 0)

    triangle_patch.set_xy([
        [0, 0],
        [display_leg_A, 0],
        [0, display_leg_B]
    ])

    # Removed leg_a_text and leg_b_text updates

    # Update the figure's suptitle with the hypotenuse value
    fig_title_artist.set_text(f'Current Hypotenuse: {hyp_val:.2f}')
    
    # Artists to return (fewer now)
    return (connection_line, ref_pt1_highlight, ref_pt2_highlight,
            tri_base_pt1_highlight, tri_base_pt2_highlight, triangle_patch,
            fig_title_artist) # Include fig_title_artist if blit=True was used

# --- 4. Create and Save Animation ---
FPS_GIF = max(1, int(1000 / INTERVAL_MS))
print(f"Total animation frames: {total_animation_frames}")
print(f"Interval: {INTERVAL_MS}ms, GIF FPS: {FPS_GIF}")

ani = animation.FuncAnimation(fig, animate, frames=total_animation_frames,
                              interval=INTERVAL_MS, blit=False, repeat=False)
# Using blit=False for robustness with suptitle updates and patch changes.

gif_filename = 'spline_triangle_simple_hypotenuse.gif'
print(f"Attempting to save animation to {gif_filename}...")
try:
    ani.save(gif_filename, writer='pillow', fps=FPS_GIF)
    print(f"Successfully saved animation to {gif_filename}")
except Exception as e:
    print(f"Error saving animation: {e}")
    print("Ensure 'pillow' is installed (pip install Pillow).")

plt.close(fig)