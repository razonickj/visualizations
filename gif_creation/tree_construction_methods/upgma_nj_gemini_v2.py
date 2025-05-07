# Previous code... (Imports, Config, Helper funcs, UPGMA, NJ)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.distance import pdist, squareform
import copy
import itertools
import sys # For flushing output

# --- Configuration ---
NUM_POINTS = 8
SAVE_ANIMATION = True # Set to True to save as gif/mp4, False to just show
FILENAME = "upgma_vs_nj_rooted_v2.gif" # Or .mp4
ANIMATION_INTERVAL = 1000 # Milliseconds between frames
DEBUG_PRINT = False # Set to True for detailed console output

# --- Helper Functions ---
# (calculate_initial_distance_matrix remains the same)
def calculate_initial_distance_matrix(points):
    """Calculates the Euclidean distance matrix."""
    return squareform(pdist(points.astype(float), metric='euclidean'))

# --- UPGMA Implementation ---
# (run_upgma remains largely the same, ensure 'parent' link is set)
# Key is that the final node dictionary in the history contains the full tree structure
def run_upgma(points):
    n = len(points)
    points = np.array(points, dtype=float)
    dist_matrix = calculate_initial_distance_matrix(points)

    active_clusters = {i: (i,) for i in range(n)}
    cluster_sizes = {i: 1 for i in range(n)}
    cluster_nodes = {i: {'id': i, 'pos': points[i].copy(), 'children': [], 'height': 0, 'original': True, 'parent': None} for i in range(n)} # Add parent: None initially

    history = [{'clusters': copy.deepcopy(active_clusters),
                'nodes': copy.deepcopy(cluster_nodes),
                'matrix_indices': list(active_clusters.keys()),
                'merged_pair': None}]

    next_cluster_id = n
    current_dist_matrix = dist_matrix.copy()
    current_indices = list(range(n))

    step_count = 0
    while len(active_clusters) > 1:
        step_count += 1
        num_active = len(current_indices)
        if DEBUG_PRINT: print(f"\nUPGMA Step {step_count}: {num_active} clusters.")

        min_dist = np.inf
        idx1, idx2 = -1, -1

        if num_active == 2:
             idx1, idx2 = 0, 1
             if current_dist_matrix.size > 0: min_dist = current_dist_matrix[0, 1]
             else: min_dist = 0
        elif num_active > 2:
            temp_matrix = current_dist_matrix.copy()
            np.fill_diagonal(temp_matrix, np.inf)
            if np.all(np.isinf(temp_matrix) | np.isnan(temp_matrix)):
                 print(" UPGMA WARNING: Distance matrix contains only Infs/NaNs.")
                 break
            min_dist = np.nanmin(temp_matrix)
            if np.isnan(min_dist) or np.isinf(min_dist):
                 print(f" UPGMA ERROR: Invalid minimum distance ({min_dist}).")
                 break
            coords = np.where(temp_matrix == min_dist)
            if len(coords[0]) == 0:
                print(" UPGMA ERROR: Could not find minimum distance coords.")
                break
            idx1, idx2 = coords[0][0], coords[1][0]
            if idx1 > idx2: idx1, idx2 = idx2, idx1
        else: break

        cluster_id1 = current_indices[idx1]
        cluster_id2 = current_indices[idx2]
        if DEBUG_PRINT: print(f"  Merging UPGMA {cluster_id1} & {cluster_id2} (dist={min_dist:.3f})")

        new_cluster_id = next_cluster_id
        next_cluster_id += 1

        merged_indices_tuple = active_clusters[cluster_id1] + active_clusters[cluster_id2]
        new_size = cluster_sizes[cluster_id1] + cluster_sizes[cluster_id2]

        height = min_dist / 2.0
        pos1 = cluster_nodes[cluster_id1]['pos']
        pos2 = cluster_nodes[cluster_id2]['pos']

        new_pos = np.array([np.nan, np.nan])
        if not np.isnan(pos1).any() and not np.isnan(pos2).any():
             new_pos = (pos1 + pos2) / 2.0
        elif DEBUG_PRINT: print(f"  WARNING: NaN parent pos for UPGMA node {new_cluster_id}.")

        if DEBUG_PRINT: print(f"  New UPGMA node {new_cluster_id} pos: {new_pos}, height: {height:.3f}")

        new_node = {'id': new_cluster_id, 'pos': new_pos, 'children': [cluster_id1, cluster_id2],
                    'parent': None, 'height': height, 'original': False}
        cluster_nodes[new_cluster_id] = new_node

        if cluster_id1 in cluster_nodes: cluster_nodes[cluster_id1]['parent'] = new_cluster_id
        if cluster_id2 in cluster_nodes: cluster_nodes[cluster_id2]['parent'] = new_cluster_id

        # --- Update Distance Matrix (simplified logic for brevity, full logic from previous step preferred) ---
        if num_active > 2:
            new_matrix_size = num_active - 1
            new_dist_matrix = np.full((new_matrix_size, new_matrix_size), np.nan)
            new_indices_map = {}
            old_to_new_map = {}
            temp_new_idx = 0
            new_distances = []

            for i in range(num_active):
                if i == idx1 or i == idx2: continue
                other_cluster_id = current_indices[i]
                dist = (current_dist_matrix[idx1, i] * cluster_sizes[cluster_id1] +
                        current_dist_matrix[idx2, i] * cluster_sizes[cluster_id2]) / new_size
                new_distances.append(dist)
                new_indices_map[temp_new_idx] = other_cluster_id
                old_to_new_map[i] = temp_new_idx
                temp_new_idx += 1

            # Copy old distances (ensure correct mapping logic is used here)
            for r_new in range(new_matrix_size - 1):
                 for c_new in range(r_new + 1, new_matrix_size - 1):
                      # Find corresponding old indices for r_new, c_new via old_to_new_map's inverse
                      old_r = [old for old, new in old_to_new_map.items() if new == r_new][0]
                      old_c = [old for old, new in old_to_new_map.items() if new == c_new][0]
                      new_dist_matrix[r_new, c_new] = new_dist_matrix[c_new, r_new] = current_dist_matrix[old_r, old_c]

            # Add new distances
            for i, dist in enumerate(new_distances):
                new_dist_matrix[i, new_matrix_size - 1] = new_dist_matrix[new_matrix_size - 1, i] = dist

            current_dist_matrix = new_dist_matrix
            new_indices = [new_indices_map[i] for i in range(new_matrix_size - 1)] + [new_cluster_id]
        else:
             current_dist_matrix = np.array([[]])
             new_indices = [new_cluster_id]

        current_indices = new_indices

        del active_clusters[cluster_id1]; del active_clusters[cluster_id2]
        del cluster_sizes[cluster_id1]; del cluster_sizes[cluster_id2]
        active_clusters[new_cluster_id] = merged_indices_tuple
        cluster_sizes[new_cluster_id] = new_size

        history.append({'clusters': copy.deepcopy(active_clusters),
                        'nodes': copy.deepcopy(cluster_nodes),
                        'matrix_indices': list(active_clusters.keys()),
                        'merged_pair': (cluster_id1, cluster_id2)})
    return history

# --- Neighbor-Joining Implementation ---
# (calculate_q_matrix remains the same)
def calculate_q_matrix(dist_matrix):
    n_active = dist_matrix.shape[0]
    if n_active <= 2: return None
    q_matrix = np.full_like(dist_matrix, np.nan, dtype=float)
    total_distances = np.nansum(dist_matrix, axis=1)
    for i in range(n_active):
        for j in range(i + 1, n_active):
             d_ij = dist_matrix[i, j]
             sum_i = total_distances[i]; sum_j = total_distances[j]
             if np.isnan(d_ij) or np.isnan(sum_i) or np.isnan(sum_j): q_matrix[i, j] = np.nan
             else: q_matrix[i, j] = (n_active - 2) * d_ij - sum_i - sum_j
             q_matrix[j, i] = q_matrix[i, j]
    return q_matrix

def run_nj(points):
    n = len(points)
    points = np.array(points, dtype=float)
    dist_matrix = calculate_initial_distance_matrix(points)

    active_clusters = {i: (i,) for i in range(n)}
    cluster_nodes = {i: {'id': i, 'pos': points[i].copy(), 'children': [], 'branch_length': 0, 'original': True, 'parent': None} for i in range(n)}

    history = [{'clusters': copy.deepcopy(active_clusters),
                'nodes': copy.deepcopy(cluster_nodes),
                'matrix_indices': list(active_clusters.keys()),
                'merged_pair': None,
                'nj_final_connector_pos': None}] # Add field for special final step

    next_cluster_id = n
    current_dist_matrix = dist_matrix.copy()
    current_indices = list(range(n))

    step_count = 0
    final_connector_pos = None # Store position of the NJ visual center point

    while len(active_clusters) > 3:
        step_count += 1
        n_active = len(current_indices)
        if DEBUG_PRINT: print(f"\nNJ Step {step_count}: {n_active} clusters.")
        if n_active <= 3: break

        if np.isnan(current_dist_matrix).all():
            print(" NJ ERROR: Distance matrix is all NaN. Stopping.")
            break
        q_matrix = calculate_q_matrix(current_dist_matrix)
        if q_matrix is None:
             if DEBUG_PRINT: print("  NJ stopping: Q-matrix calculation failed."); break
        if np.isnan(q_matrix).all():
             print(" NJ ERROR: Q-matrix is all NaN. Stopping."); break

        min_q = np.nanmin(q_matrix)
        if np.isinf(min_q) or np.isnan(min_q):
             print(f" NJ ERROR: Invalid minimum Q value ({min_q}). Stopping."); break
        coords = np.where(q_matrix == min_q)
        if len(coords[0]) == 0: print(" NJ ERROR: Could not find coords for min Q."); break
        idx_f, idx_g = coords[0][0], coords[1][0]
        if idx_f > idx_g: idx_f, idx_g = idx_g, idx_f

        cluster_id_f = current_indices[idx_f]; cluster_id_g = current_indices[idx_g]
        if DEBUG_PRINT: print(f"  Merging NJ {cluster_id_f} & {cluster_id_g} (Q={min_q:.3f})")

        dist_fg = current_dist_matrix[idx_f, idx_g]
        total_dist_f = np.nansum(current_dist_matrix[idx_f, :])
        total_dist_g = np.nansum(current_dist_matrix[idx_g, :])
        len_fu, len_gu = np.nan, np.nan

        if np.isnan(dist_fg): print(f"  NJ WARNING: Dist between {cluster_id_f}, {cluster_id_g} is NaN.")
        elif n_active <= 2: len_fu = len_gu = dist_fg / 2.0 if not np.isnan(dist_fg) else np.nan
        else:
             denom = (n_active - 2); denom = 1 if denom == 0 else denom
             raw_len_fu = 0.5 * dist_fg + (total_dist_f - total_dist_g) / (2 * denom)
             raw_len_gu = dist_fg - raw_len_fu
             len_fu = max(0, raw_len_fu) if not np.isnan(raw_len_fu) else np.nan
             len_gu = max(0, raw_len_gu) if not np.isnan(raw_len_gu) else np.nan
        if DEBUG_PRINT: print(f"  Branch lengths: {cluster_id_f}->u = {len_fu:.3f}, {cluster_id_g}->u = {len_gu:.3f}")

        new_cluster_id = next_cluster_id; next_cluster_id += 1
        merged_indices_tuple = active_clusters[cluster_id_f] + active_clusters[cluster_id_g]
        pos_f = cluster_nodes[cluster_id_f]['pos']; pos_g = cluster_nodes[cluster_id_g]['pos']
        new_pos = np.array([np.nan, np.nan])
        if not np.isnan(len_fu) and not np.isnan(len_gu) and not np.isnan(pos_f).any() and not np.isnan(pos_g).any():
            denominator = len_fu + len_gu
            if denominator > 1e-9: new_pos = pos_f * (len_gu / denominator) + pos_g * (len_fu / denominator)
            else: new_pos = (pos_f + pos_g) / 2.0
        elif DEBUG_PRINT: print(f"  NJ WARNING: Cannot calculate position for node {new_cluster_id}.")
        if DEBUG_PRINT: print(f"  New NJ node {new_cluster_id} pos: {new_pos}")

        new_node = {'id': new_cluster_id, 'pos': new_pos, 'children': [cluster_id_f, cluster_id_g],
                    'parent': None, 'branch_length': 0, 'original': False}
        cluster_nodes[new_cluster_id] = new_node
        if cluster_id_f in cluster_nodes:
            cluster_nodes[cluster_id_f]['parent'] = new_cluster_id; cluster_nodes[cluster_id_f]['branch_length'] = len_fu
        if cluster_id_g in cluster_nodes:
            cluster_nodes[cluster_id_g]['parent'] = new_cluster_id; cluster_nodes[cluster_id_g]['branch_length'] = len_gu

        # --- Update Distance Matrix (simplified logic for brevity) ---
        if n_active > 3:
            new_matrix_size = n_active - 1
            new_dist_matrix = np.full((new_matrix_size, new_matrix_size), np.nan)
            new_indices_map = {}; old_to_new_map = {}; temp_new_idx = 0; new_distances = []
            for i in range(n_active):
                if i == idx_f or i == idx_g: continue
                other_cluster_id = current_indices[i]
                dist_fk = current_dist_matrix[idx_f, i]; dist_gk = current_dist_matrix[idx_g, i]
                dist_uk = np.nan
                if not (np.isnan(dist_fk) or np.isnan(dist_gk) or np.isnan(dist_fg)):
                     dist_uk = 0.5 * (dist_fk + dist_gk - dist_fg)
                new_distances.append(max(0, dist_uk) if not np.isnan(dist_uk) else np.nan)
                new_indices_map[temp_new_idx] = other_cluster_id; old_to_new_map[i] = temp_new_idx; temp_new_idx += 1

            # Copy old distances
            for r_new in range(new_matrix_size - 1):
                for c_new in range(r_new + 1, new_matrix_size - 1):
                      old_r = [old for old, new in old_to_new_map.items() if new == r_new][0]
                      old_c = [old for old, new in old_to_new_map.items() if new == c_new][0]
                      new_dist_matrix[r_new, c_new] = new_dist_matrix[c_new, r_new] = current_dist_matrix[old_r, old_c]
            # Add new distances
            for i, dist in enumerate(new_distances):
                new_dist_matrix[i, new_matrix_size - 1] = new_dist_matrix[new_matrix_size - 1, i] = dist

            current_dist_matrix = new_dist_matrix
            new_indices = [new_indices_map[i] for i in range(new_matrix_size - 1)] + [new_cluster_id]
        else:
            new_indices = [cid for cid in current_indices if cid != cluster_id_f and cid != cluster_id_g] + [new_cluster_id]
            current_dist_matrix = np.array([[]])

        current_indices = new_indices
        del active_clusters[cluster_id_f]; del active_clusters[cluster_id_g]
        active_clusters[new_cluster_id] = merged_indices_tuple

        history.append({'clusters': copy.deepcopy(active_clusters),
                        'nodes': copy.deepcopy(cluster_nodes),
                        'matrix_indices': list(active_clusters.keys()),
                        'merged_pair': (cluster_id_f, cluster_id_g),
                        'nj_final_connector_pos': None})

    # --- Handle the final 3 nodes for NJ (after loop finishes) ---
    if len(current_indices) == 3:
         if DEBUG_PRINT: print("\nNJ: Handling final 3 nodes.")
         id_f, id_g, id_h = current_indices[0], current_indices[1], current_indices[2]

         # Calculate the visual center point - centroid (NOT a real node)
         pos_f = cluster_nodes[id_f]['pos']; pos_g = cluster_nodes[id_g]['pos']; pos_h = cluster_nodes[id_h]['pos']
         final_connector_pos = np.array([np.nan, np.nan]) # Store this special position
         if not np.isnan(pos_f).any() and not np.isnan(pos_g).any() and not np.isnan(pos_h).any():
              final_connector_pos = (pos_f + pos_g + pos_h) / 3.0
         elif DEBUG_PRINT: print("  NJ WARNING: Cannot calculate final connector position.")

         # We don't create a new node in cluster_nodes for this.
         # We just need to store the state representing the end.
         # The drawing function will use final_connector_pos if present.

         # Update history for the *final* frame - show the connection point info
         # Take the last state and add the connector info
         final_state = copy.deepcopy(history[-1]) # Modify the last recorded state
         final_state['nj_final_connector_pos'] = final_connector_pos
         final_state['merged_pair'] = (id_f, id_g, id_h) # Note the final connection
         # The active clusters/matrix indices might not be meaningful here, keep as is?
         # Or represent as a single "final tree" cluster? Let's keep it simple.
         # We need a *new* history entry to signify this final connection step visually
         history.append(final_state)

    elif DEBUG_PRINT: print(f"NJ finished with {len(current_indices)} clusters.")

    return history

# --- Animation Setup ---
np.random.seed(42)
points = np.random.rand(NUM_POINTS, 2) * 10

print("Running UPGMA...")
upgma_history = run_upgma(points)
print("Running Neighbor-Joining...")
nj_history = run_nj(points)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
fig.subplots_adjust(bottom=0.15) # Add space at the bottom for text labels

# Find overall bounds
all_x = points[:, 0]; all_y = points[:, 1]
x_min, x_max = np.min(all_x), np.max(all_x)
y_min, y_max = np.min(all_y), np.max(all_y)
x_range = max(1.0, x_max - x_min); y_range = max(1.0, y_max - y_min)
padding = max(x_range, y_range) * 0.20 # Slightly more padding

# Add descriptive text below axes
fig.text(0.25, 0.05, 'Rooted Tree (UPGMA)', ha='center', va='center', fontsize=12)
fig.text(0.75, 0.05, 'Unrooted Tree (Neighbor-Joining)', ha='center', va='center', fontsize=12)


# --- draw_tree Function ---
# Add parameter final_upgma_root_id
# Add parameter nj_connector_pos
def draw_tree(ax, nodes_state, original_points, step_num, title, frame_num,
              final_upgma_root_id=None, nj_connector_pos=None, is_nj_plot=False):
    ax.clear()
    leaf_nodes = {nid: ninfo for nid, ninfo in nodes_state.items() if ninfo.get('original', False)}
    leaf_coords = np.array([ninfo['pos'] for nid, ninfo in leaf_nodes.items() if not np.isnan(ninfo['pos']).any()])
    if leaf_coords.size > 0:
        ax.scatter(leaf_coords[:, 0], leaf_coords[:, 1], c='blue', zorder=5, s=40)

    drawn_connections = set()
    internal_node_coords = {} # Store coords keyed by node_id for special styling

    if DEBUG_PRINT: print(f"\n--- Drawing Frame {frame_num} ({title} Step {step_num}) ---")

    # Draw lines first
    for node_id, node_info in nodes_state.items():
        parent_id = node_info.get('parent')

        # Add valid internal node positions for later plotting
        if not node_info.get('original', False):
             node_pos = node_info['pos']
             if not np.isnan(node_pos).any():
                 internal_node_coords[node_id] = node_pos

        # --- Draw line to parent ---
        target_pos = None
        is_final_nj_connection = False

        if parent_id is not None and parent_id in nodes_state:
            parent_info = nodes_state[parent_id]
            target_pos = parent_info['pos']
            connection_id = parent_id # ID of the parent node
        elif is_nj_plot and nj_connector_pos is not None and not np.isnan(nj_connector_pos).any():
             # Special case: If it's the NJ plot, and this node's parent should be the visual connector
             # We identify the nodes that connect TO the center point in the final step.
             # This logic needs refinement: how do we know node_id should connect to nj_connector_pos?
             # Let's assume if parent is None AND nj_connector_pos is valid, it *might* be one of the final 3.
             # A better way: The final state in history explicitly lists the 3 nodes connected at the end.
             # Let's simplify: just use parent link. The final connector isn't a 'parent' in nodes_state.
             pass # Handled below - special check for nj_connector_pos added


        # If we have a valid parent node in nodes_state
        if parent_id is not None and parent_id in nodes_state:
             parent_info = nodes_state[parent_id]
             target_pos = parent_info['pos']
             child_pos = node_info['pos']
             connection = tuple(sorted((node_id, parent_id)))

             if connection not in drawn_connections:
                 if np.isnan(child_pos).any() or np.isnan(target_pos).any():
                      if DEBUG_PRINT: print(f"  Skip line: NaN pos {connection}. Child:{child_pos}, Parent:{target_pos}")
                 else:
                      if DEBUG_PRINT: print(f"  Draw line: {connection} to parent {parent_id}")
                      ax.plot([target_pos[0], child_pos[0]], [target_pos[1], child_pos[1]],
                              color='grey', linestyle='-', linewidth=1.5, zorder=1)
                      drawn_connections.add(connection)

    # --- Special final NJ connection lines ---
    if is_nj_plot and nj_connector_pos is not None and not np.isnan(nj_connector_pos).any():
         # Find the last 3 nodes that were connected (assuming they are the only ones without parents now)
         final_nodes_ids = [nid for nid, ninfo in nodes_state.items() if ninfo.get('parent') is None and nid in internal_node_coords] # Find nodes without parent
         # Correction: Need the IDs from the *last step* merge info. Pass this maybe?
         # Simpler: Assume the 3 nodes connected are those whose parent *would be* the connector.
         # This is tricky. Let's use the merged_pair from the last state if it's length 3
         last_merged = history[step_num].get('merged_pair') # Get merged pair info from history
         if isinstance(last_merged, tuple) and len(last_merged) == 3:
             final_nodes_ids = last_merged
             for node_id in final_nodes_ids:
                 if node_id in nodes_state:
                     child_pos = nodes_state[node_id]['pos']
                     if not np.isnan(child_pos).any():
                          # Check if connection already drawn (e.g. if parent link was somehow set)
                          # This check is complex, let's just draw these final lines
                          if DEBUG_PRINT: print(f"  Draw line: final NJ node {node_id} to connector")
                          ax.plot([nj_connector_pos[0], child_pos[0]], [nj_connector_pos[1], child_pos[1]],
                                  color='grey', linestyle='-', linewidth=1.5, zorder=1)
                          # drawn_connections.add(???) # How to represent connection to virtual node? Skip for now.


    # --- Plot internal node markers ---
    for node_id, node_pos in internal_node_coords.items():
        marker = 's' # Default square
        size = 35
        color = 'red'
        order = 3

        # Special style for final UPGMA root
        if not is_nj_plot and node_id == final_upgma_root_id:
            marker = 'D' # Diamond
            size = 60
            color = 'purple'
            order = 4

        # Skip the visual connector point in NJ plot (it has no node_id in internal_node_coords)
        # The check `is_nj_plot and nj_connector_pos is not None` is handled by not having a node for it.

        ax.scatter(node_pos[0], node_pos[1], c=color, marker=marker, s=size, zorder=order, alpha=0.8)


    ax.set_title(f"{title} - Step {step_num}")
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')


# --- Animation Update Function ---
num_frames = max(len(upgma_history), len(nj_history))

# Identify the final UPGMA root ID (if history is not empty)
final_upgma_root_id = None
if upgma_history:
    last_upgma_state = upgma_history[-1]
    if last_upgma_state['matrix_indices']:
         final_upgma_root_id = last_upgma_state['matrix_indices'][0]


def update(frame):
    if DEBUG_PRINT: print(f"\n>>> Generating Frame {frame+1}/{num_frames}", flush=True)

    # UPGMA Update
    upgma_idx = min(frame, len(upgma_history) - 1)
    is_final_upgma_frame = (frame >= len(upgma_history) - 1)
    if upgma_idx < len(upgma_history):
        upgma_state = upgma_history[upgma_idx]
        draw_tree(ax1, upgma_state['nodes'], points, upgma_idx, "UPGMA", frame+1,
                  final_upgma_root_id=(final_upgma_root_id if is_final_upgma_frame else None), # Only mark root on final frame(s)
                  is_nj_plot=False)
    else:
        ax1.clear(); ax1.set_title(f"UPGMA - End")

    # NJ Update
    nj_idx = min(frame, len(nj_history) - 1)
    is_final_nj_frame = (frame >= len(nj_history) - 1)
    if nj_idx < len(nj_history):
        nj_state = nj_history[nj_idx]
        # Get the special connector position for this frame (only present in the very last state)
        nj_connector_pos = nj_state.get('nj_final_connector_pos')
        draw_tree(ax2, nj_state['nodes'], points, nj_idx, "Neighbor-Joining", frame+1,
                  nj_connector_pos=nj_connector_pos, # Pass connector pos
                  is_nj_plot=True)
    else:
        ax2.clear(); ax2.set_title(f"NJ - End")

    # fig.tight_layout(pad=2.0) # REMOVED FROM HERE


# --- Create and Save/Show Animation ---
# Apply tight_layout ONCE before animation
try:
    fig.tight_layout(pad=2.0, rect=[0, 0.08, 1, 1]) # Adjust rect to accommodate bottom text
except Exception as e:
     print(f"Warning: tight_layout failed: {e}") # Sometimes fails, proceed anyway

ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=ANIMATION_INTERVAL, repeat=False)

# (Saving/Showing logic remains the same as previous step)
if SAVE_ANIMATION:
    print(f"Saving animation to {FILENAME}...")
    # ... (rest of saving logic) ...
    try:
        writer = None
        fps = max(1, int(1000/ANIMATION_INTERVAL))
        if FILENAME.endswith(".gif"):
             writer = animation.PillowWriter(fps=fps)
        elif FILENAME.endswith(".mp4"):
             writer = animation.FFMpegWriter(fps=fps)
        else:
             raise ValueError("Unsupported file extension. Use .gif or .mp4")

        if writer:
            ani.save(FILENAME, writer=writer)
            print(f"Animation saved successfully to {FILENAME}")
        else:
             print("Could not create animation writer.")
             if not DEBUG_PRINT: plt.show()
    except Exception as e:
        print(f"\n--- Error saving animation ---\n{type(e).__name__}: {e}\n------------------------------")
        print("Trying to show static plot instead...")
        try:
            update(num_frames - 1) # Draw last frame
            plt.show()
        except Exception as show_e:
             print(f"Also failed to show plot: {show_e}")

else:
    print("Showing plot...")
    plt.show()

print("Script finished.")