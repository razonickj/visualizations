# Previous code... (Imports, Config, Helper funcs)
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
FILENAME = "upgma_vs_nj_2d_revised.gif" # Or .mp4
ANIMATION_INTERVAL = 1000 # Slower for debugging Milliseconds between frames
DEBUG_PRINT = False # Set to True for detailed console output

# --- Helper Functions ---
def calculate_initial_distance_matrix(points):
    """Calculates the Euclidean distance matrix."""
    # Ensure input is float to avoid potential integer distance issues if using certain metrics
    return squareform(pdist(points.astype(float), metric='euclidean'))

# --- UPGMA Implementation ---
def run_upgma(points):
    n = len(points)
    # Make sure points are float
    points = np.array(points, dtype=float)
    dist_matrix = calculate_initial_distance_matrix(points)

    active_clusters = {i: (i,) for i in range(n)}
    cluster_sizes = {i: 1 for i in range(n)}
    # Initialize nodes correctly, store original status
    cluster_nodes = {i: {'id': i, 'pos': points[i].copy(), 'children': [], 'height': 0, 'original': True} for i in range(n)}

    history = [{'clusters': copy.deepcopy(active_clusters),
                'nodes': copy.deepcopy(cluster_nodes),
                'matrix_indices': list(active_clusters.keys()),
                'merged_pair': None}] # Add info about what merged

    next_cluster_id = n
    current_dist_matrix = dist_matrix.copy()
    current_indices = list(range(n)) # Maps matrix row/col index to cluster_id

    step_count = 0
    while len(active_clusters) > 1:
        step_count += 1
        num_active = len(current_indices)
        if DEBUG_PRINT: print(f"\nUPGMA Step {step_count}: {num_active} clusters.")

        min_dist = np.inf
        idx1, idx2 = -1, -1
        
        if num_active == 2:
             idx1, idx2 = 0, 1
             # Ensure matrix access is valid if matrix became 1x1 then 0x0 before loop check
             if current_dist_matrix.size > 0:
                 min_dist = current_dist_matrix[idx1, idx2]
             else:
                 if DEBUG_PRINT: print(" UPGMA: Matrix empty at final step?")
                 min_dist = 0 # Or handle appropriately
        elif num_active > 2:
            # Find the pair with the minimum distance - ignoring diagonal NaNs
            temp_matrix = current_dist_matrix.copy()
            np.fill_diagonal(temp_matrix, np.inf) # Avoid self-comparison
            
            if np.all(np.isinf(temp_matrix)): # Check if only Infs left
                 print(" UPGMA WARNING: Distance matrix contains only Infs.")
                 break # Cannot find minimum finite distance

            min_dist = np.nanmin(temp_matrix) # Use nanmin
            coords = np.where(temp_matrix == min_dist)
            
            if len(coords[0]) == 0: # Handle case where min_dist might be NaN itself if all entries were NaN
                print(" UPGMA ERROR: Could not find minimum distance (all NaN/Inf?).")
                break

            # Take the first occurrence
            idx1, idx2 = coords[0][0], coords[1][0]
            # Ensure idx1 < idx2 for consistent removal later
            if idx1 > idx2: idx1, idx2 = idx2, idx1
        else: # Should not happen if loop condition is len > 1
            break

        cluster_id1 = current_indices[idx1]
        cluster_id2 = current_indices[idx2]
        if DEBUG_PRINT: print(f"  Merging UPGMA {cluster_id1} & {cluster_id2} (dist={min_dist:.3f})")

        # Merge clusters
        new_cluster_id = next_cluster_id
        next_cluster_id += 1

        merged_indices_tuple = active_clusters[cluster_id1] + active_clusters[cluster_id2]
        new_size = cluster_sizes[cluster_id1] + cluster_sizes[cluster_id2]

        height = min_dist / 2.0
        pos1 = cluster_nodes[cluster_id1]['pos']
        pos2 = cluster_nodes[cluster_id2]['pos']
        
        # Ensure positions are valid before averaging
        new_pos = np.array([np.nan, np.nan]) # Default to NaN
        if not np.isnan(pos1).any() and not np.isnan(pos2).any():
             new_pos = (pos1 + pos2) / 2.0 # Simple midpoint for visualization
        elif DEBUG_PRINT:
             print(f"  WARNING: NaN position encountered for parent {cluster_id1} or {cluster_id2}. New node {new_cluster_id} pos is NaN.")

        if DEBUG_PRINT: print(f"  New UPGMA node {new_cluster_id} pos: {new_pos}, height: {height:.3f}")

        new_node = {'id': new_cluster_id,
                    'pos': new_pos,
                    'children': [cluster_id1, cluster_id2],
                    'parent': None, # Parent link set later if this node merges
                    'height': height,
                    'original': False}
        cluster_nodes[new_cluster_id] = new_node
        
        # Set parent link in children
        if cluster_id1 in cluster_nodes: cluster_nodes[cluster_id1]['parent'] = new_cluster_id
        if cluster_id2 in cluster_nodes: cluster_nodes[cluster_id2]['parent'] = new_cluster_id


        # --- Update Distance Matrix ---
        if num_active > 2: # Only need to update matrix if >2 clusters were present
            new_matrix_size = num_active - 1
            new_dist_matrix = np.full((new_matrix_size, new_matrix_size), np.nan) # Init with NaN

            # Calculate distances from new cluster to others
            new_distances = np.full(new_matrix_size -1, np.nan) # Distances to non-merged nodes
            
            temp_new_idx = 0
            new_indices_map = {} # Maps new matrix index to old cluster ID
            old_to_new_map = {}  # Maps old matrix index to new matrix index

            for i in range(num_active):
                if i == idx1 or i == idx2:
                    continue
                other_cluster_id = current_indices[i]
                
                # Weighted average distance calculation
                dist = (current_dist_matrix[idx1, i] * cluster_sizes[cluster_id1] +
                        current_dist_matrix[idx2, i] * cluster_sizes[cluster_id2]) / new_size
                new_distances[temp_new_idx] = dist
                
                new_indices_map[temp_new_idx] = other_cluster_id
                old_to_new_map[i] = temp_new_idx
                temp_new_idx += 1

            # Copy over old distances among remaining nodes
            for i in range(new_matrix_size - 1):
                for j in range(i + 1, new_matrix_size - 1):
                     old_i_idx = -1
                     old_j_idx = -1
                     # Find original indices corresponding to new indices i, j
                     for old_idx, new_idx in old_to_new_map.items():
                         if new_idx == i: old_i_idx = old_idx
                         if new_idx == j: old_j_idx = old_idx
                     
                     if old_i_idx != -1 and old_j_idx != -1:
                          new_dist_matrix[i, j] = new_dist_matrix[j, i] = current_dist_matrix[old_i_idx, old_j_idx]
                     elif DEBUG_PRINT:
                          print(f"  UPGMA WARN: Could not map new indices {i},{j} back to old.")


            # Add new distances for the merged cluster (last row/col)
            for i, dist in enumerate(new_distances):
                new_dist_matrix[i, new_matrix_size - 1] = new_dist_matrix[new_matrix_size - 1, i] = dist

            current_dist_matrix = new_dist_matrix
            # Update active clusters and indices
            new_indices = [new_indices_map[i] for i in range(new_matrix_size - 1)] + [new_cluster_id]
        else: # Last merge, only the new cluster remains
             current_dist_matrix = np.array([[]]) # Empty matrix
             new_indices = [new_cluster_id]


        current_indices = new_indices

        del active_clusters[cluster_id1]
        del active_clusters[cluster_id2]
        del cluster_sizes[cluster_id1]
        del cluster_sizes[cluster_id2]
        active_clusters[new_cluster_id] = merged_indices_tuple
        cluster_sizes[new_cluster_id] = new_size

        # Store state
        history.append({'clusters': copy.deepcopy(active_clusters),
                        'nodes': copy.deepcopy(cluster_nodes), # Ensure deep copy of node states
                        'matrix_indices': list(active_clusters.keys()),
                        'merged_pair': (cluster_id1, cluster_id2)})

    return history


# --- Neighbor-Joining Implementation ---
def calculate_q_matrix(dist_matrix):
    n_active = dist_matrix.shape[0]
    if n_active <= 2: # Q-matrix not defined for n=2
        return None
        
    q_matrix = np.full_like(dist_matrix, np.nan, dtype=float) # Initialize with NaN
    
    # Use nan-aware sum
    total_distances = np.nansum(dist_matrix, axis=1) 
    
    # If a row/col in dist_matrix is all NaN, its sum will be 0. Handle this?
    # If dist_matrix has NaNs, Q calculation might produce NaNs

    for i in range(n_active):
        for j in range(i + 1, n_active):
             d_ij = dist_matrix[i, j]
             sum_i = total_distances[i]
             sum_j = total_distances[j]
             
             # If any component is NaN, result is NaN
             if np.isnan(d_ij) or np.isnan(sum_i) or np.isnan(sum_j):
                 q_matrix[i, j] = np.nan
             else:
                 # Standard Q formula
                 q_value = (n_active - 2) * d_ij - sum_i - sum_j
                 q_matrix[i, j] = q_value

             q_matrix[j, i] = q_matrix[i, j] # Symmetric
             
    return q_matrix

def run_nj(points):
    n = len(points)
    points = np.array(points, dtype=float) # Ensure float
    dist_matrix = calculate_initial_distance_matrix(points)

    active_clusters = {i: (i,) for i in range(n)}
    cluster_nodes = {i: {'id': i, 'pos': points[i].copy(), 'children': [], 'branch_length': 0, 'original': True} for i in range(n)}

    history = [{'clusters': copy.deepcopy(active_clusters),
                'nodes': copy.deepcopy(cluster_nodes),
                'matrix_indices': list(active_clusters.keys()),
                'merged_pair': None}]

    next_cluster_id = n
    current_dist_matrix = dist_matrix.copy()
    current_indices = list(range(n)) # Maps matrix row/col index to cluster_id

    step_count = 0
    while len(active_clusters) > 3: # Standard NJ stops when 3 clusters remain
        step_count += 1
        n_active = len(current_indices)
        if DEBUG_PRINT: print(f"\nNJ Step {step_count}: {n_active} clusters.")
        if n_active <= 3: break # Exit if somehow we reach <=3

        # Check distance matrix for validity
        if np.isnan(current_dist_matrix).all():
            print(" NJ ERROR: Distance matrix is all NaN. Stopping.")
            break

        # Calculate Q-matrix
        q_matrix = calculate_q_matrix(current_dist_matrix)
        if q_matrix is None:
             if DEBUG_PRINT: print("  NJ stopping: Q-matrix calculation failed (<=2 clusters?).")
             break

        # Check Q-matrix for validity
        if np.isnan(q_matrix).all():
             print(" NJ ERROR: Q-matrix is all NaN. Cannot find minimum. Stopping.")
             # Add a placeholder error state?
             # history.append({'clusters': ..., 'nodes': ..., 'merged_pair': 'ERROR_Q_NAN'})
             break

        # Find pair (f, g) with minimum Q value, ignoring NaNs
        min_q = np.nanmin(q_matrix)

        if np.isinf(min_q) or np.isnan(min_q):
             print(f" NJ ERROR: Invalid minimum Q value ({min_q}). Check Q matrix calculation. Stopping.")
             # Check contents of q_matrix
             if DEBUG_PRINT: print(" Q-Matrix:\n", q_matrix)
             break

        coords = np.where(q_matrix == min_q)
        if len(coords[0]) == 0:
             print(" NJ ERROR: Could not find coordinates for minimum Q value. Stopping.")
             break
             
        # Take the first occurrence
        idx_f, idx_g = coords[0][0], coords[1][0]

        # Ensure idx_f < idx_g for consistent processing
        if idx_f > idx_g: idx_f, idx_g = idx_g, idx_f

        cluster_id_f = current_indices[idx_f]
        cluster_id_g = current_indices[idx_g]
        if DEBUG_PRINT: print(f"  Merging NJ {cluster_id_f} & {cluster_id_g} (Q={min_q:.3f})")

        # --- Calculate branch lengths to new node u ---
        dist_fg = current_dist_matrix[idx_f, idx_g]

        # Use nan-aware sums for total distances
        total_dist_f = np.nansum(current_dist_matrix[idx_f, :])
        total_dist_g = np.nansum(current_dist_matrix[idx_g, :])
        
        len_fu, len_gu = np.nan, np.nan # Default to NaN

        if np.isnan(dist_fg):
             print(f"  NJ WARNING: Distance between chosen pair ({cluster_id_f}, {cluster_id_g}) is NaN. Branch lengths will be NaN.")
        elif n_active <= 2:
             # Should not happen with loop condition > 3, but as fallback
             len_fu = len_gu = dist_fg / 2.0 if not np.isnan(dist_fg) else np.nan
        else:
             denom = (n_active - 2) # Denominator for NJ branch length formula
             if denom == 0: denom = 1 # Safety, avoid division by zero
                 
             # Calculate raw lengths
             raw_len_fu = 0.5 * dist_fg + (total_dist_f - total_dist_g) / (2 * denom)
             raw_len_gu = dist_fg - raw_len_fu # Use relationship d(f,g) = L(f,u) + L(g,u)

             # Clamp branch lengths to be non-negative, propagate NaN
             len_fu = max(0, raw_len_fu) if not np.isnan(raw_len_fu) else np.nan
             len_gu = max(0, raw_len_gu) if not np.isnan(raw_len_gu) else np.nan

        if DEBUG_PRINT: print(f"  Branch lengths: {cluster_id_f}->u = {len_fu:.3f}, {cluster_id_g}->u = {len_gu:.3f}")

        # --- Create new node u ---
        new_cluster_id = next_cluster_id
        next_cluster_id += 1
        merged_indices_tuple = active_clusters[cluster_id_f] + active_clusters[cluster_id_g]

        pos_f = cluster_nodes[cluster_id_f]['pos']
        pos_g = cluster_nodes[cluster_id_g]['pos']

        # Calculate new position only if lengths and positions are valid
        new_pos = np.array([np.nan, np.nan]) # Default to NaN
        if not np.isnan(len_fu) and not np.isnan(len_gu) and \
           not np.isnan(pos_f).any() and not np.isnan(pos_g).any():
            denominator = len_fu + len_gu
            if denominator > 1e-9: # Avoid division by zero for coincident points/zero lengths
                 # Place node along line segment weighted by branch lengths
                 new_pos = pos_f * (len_gu / denominator) + pos_g * (len_fu / denominator)
            else:
                 new_pos = (pos_f + pos_g) / 2.0 # Fallback: midpoint if lengths are zero
        elif DEBUG_PRINT:
             print(f"  NJ WARNING: Cannot calculate position for node {new_cluster_id} due to invalid inputs (lengths or parent positions).")

        if DEBUG_PRINT: print(f"  New NJ node {new_cluster_id} pos: {new_pos}")

        new_node = {'id': new_cluster_id,
                    'pos': new_pos,
                    'children': [cluster_id_f, cluster_id_g],
                    'parent': None, # Set later
                    'branch_length': 0, # Branch length is for connection *to* parent
                    'original': False}
        cluster_nodes[new_cluster_id] = new_node
        
        # Set parent links and branch lengths for children connecting TO the new node u
        if cluster_id_f in cluster_nodes:
            cluster_nodes[cluster_id_f]['parent'] = new_cluster_id
            cluster_nodes[cluster_id_f]['branch_length'] = len_fu
        if cluster_id_g in cluster_nodes:
            cluster_nodes[cluster_id_g]['parent'] = new_cluster_id
            cluster_nodes[cluster_id_g]['branch_length'] = len_gu

        # --- Update distance matrix for remaining nodes ---
        if n_active > 3: # Only need to update if more than 3 nodes existed before merge
            new_matrix_size = n_active - 1
            new_dist_matrix = np.full((new_matrix_size, new_matrix_size), np.nan) # Init with NaN

            new_distances = np.full(new_matrix_size - 1, np.nan) # Distances from new node 'u' to others 'k'
            
            temp_new_idx = 0
            new_indices_map = {} # Maps new matrix index to old cluster ID
            old_to_new_map = {}  # Maps old matrix index to new matrix index
            
            # Calculate distances d(u,k)
            for i in range(n_active):
                if i == idx_f or i == idx_g:
                    continue
                other_cluster_id = current_indices[i]
                
                dist_fk = current_dist_matrix[idx_f, i]
                dist_gk = current_dist_matrix[idx_g, i]

                # Propagate NaNs
                if np.isnan(dist_fk) or np.isnan(dist_gk) or np.isnan(dist_fg):
                     dist_uk = np.nan
                else:
                     dist_uk = 0.5 * (dist_fk + dist_gk - dist_fg)
                     
                # Store non-negative distance, keep NaN if calculated as NaN
                new_distances[temp_new_idx] = max(0, dist_uk) if not np.isnan(dist_uk) else np.nan
                
                new_indices_map[temp_new_idx] = other_cluster_id
                old_to_new_map[i] = temp_new_idx
                temp_new_idx += 1

            # Copy over old distances d(k,l) among remaining nodes
            for i in range(new_matrix_size - 1):
                for j in range(i + 1, new_matrix_size - 1):
                     old_i_idx, old_j_idx = -1, -1
                     for old_idx, new_idx in old_to_new_map.items():
                          if new_idx == i: old_i_idx = old_idx
                          if new_idx == j: old_j_idx = old_idx
                          
                     if old_i_idx != -1 and old_j_idx != -1:
                          new_dist_matrix[i, j] = new_dist_matrix[j, i] = current_dist_matrix[old_i_idx, old_j_idx]
                     elif DEBUG_PRINT:
                           print(f"  NJ WARN: Could not map new indices {i},{j} back to old for dist copy.")

            # Add new distances d(u,k) involving the merged cluster (last row/col)
            for i, dist in enumerate(new_distances):
                new_dist_matrix[i, new_matrix_size - 1] = new_dist_matrix[new_matrix_size - 1, i] = dist

            current_dist_matrix = new_dist_matrix
            # Update active cluster IDs mapping
            new_indices = [new_indices_map[i] for i in range(new_matrix_size - 1)] + [new_cluster_id]
        
        else: # We just merged down TO 3 nodes, no further matrix update needed for loop
            # Determine the final 3 indices correctly
            new_indices = [cid for cid in current_indices if cid != cluster_id_f and cid != cluster_id_g] + [new_cluster_id]
            current_dist_matrix = np.array([[]]) # Matrix no longer needed for loop logic

        current_indices = new_indices

        del active_clusters[cluster_id_f]
        del active_clusters[cluster_id_g]
        active_clusters[new_cluster_id] = merged_indices_tuple

        # Store state for this step
        history.append({'clusters': copy.deepcopy(active_clusters),
                        'nodes': copy.deepcopy(cluster_nodes),
                        'matrix_indices': list(active_clusters.keys()),
                        'merged_pair': (cluster_id_f, cluster_id_g)})

    # --- Handle the final 3 nodes for NJ (after loop finishes) ---
    if len(current_indices) == 3:
         if DEBUG_PRINT: print("\nNJ: Handling final 3 nodes.")
         id_f, id_g, id_h = current_indices[0], current_indices[1], current_indices[2]

         # Final step: connect these last 3 nodes. The tree is unrooted.
         # We calculate the final branch lengths connecting them to the central 'inferred' node.
         # Need the distances between f, g, h. These should be the last calculated distances
         # before they were potentially merged into larger clusters OR use original distances
         # if they are original leaves. Let's try recalculating from original points they contain.

         # For simplicity in this VISUALIZATION, we connect them to a central point.
         # The branch lengths used SHOULD be those calculated when f, g, h were LAST merged
         # into their parent nodes (which are now children of this final trifurcation).

         len_f = cluster_nodes[id_f].get('branch_length', 0.1) # Length connecting f to the center
         len_g = cluster_nodes[id_g].get('branch_length', 0.1) # Length connecting g to the center
         len_h = cluster_nodes[id_h].get('branch_length', 0.1) # Length connecting h to the center
         
         if DEBUG_PRINT: print(f"  Final lengths: f={len_f:.3f}, g={len_g:.3f}, h={len_h:.3f}")

         # Create a final central node (visual proxy for the unrooted center)
         final_node_id = next_cluster_id
         pos_f = cluster_nodes[id_f]['pos']
         pos_g = cluster_nodes[id_g]['pos']
         pos_h = cluster_nodes[id_h]['pos']

         # Calculate position only if valid inputs
         final_pos = np.array([np.nan, np.nan])
         if not np.isnan(pos_f).any() and not np.isnan(pos_g).any() and not np.isnan(pos_h).any():
              # Simple centroid for visualization - NOT phylogenetically meaningful position
              final_pos = (pos_f + pos_g + pos_h) / 3.0
         elif DEBUG_PRINT:
              print("  NJ WARNING: Cannot calculate final node position due to invalid parent positions.")

         if DEBUG_PRINT: print(f"  Final NJ node {final_node_id} pos: {final_pos}")

         cluster_nodes[final_node_id] = {'id': final_node_id, 'pos': final_pos,
                                         'children': [id_f, id_g, id_h],
                                         'parent': None, 'branch_length': 0, 'original': False}
         # Update parent links for the last 3 nodes
         if id_f in cluster_nodes: cluster_nodes[id_f]['parent'] = final_node_id
         if id_g in cluster_nodes: cluster_nodes[id_g]['parent'] = final_node_id
         if id_h in cluster_nodes: cluster_nodes[id_h]['parent'] = final_node_id

         # Update active clusters to contain only the root
         merged_final_tuple = active_clusters[id_f] + active_clusters[id_g] + active_clusters[id_h]
         active_clusters = {final_node_id: merged_final_tuple}
         
         # Store this final state
         history.append({'clusters': copy.deepcopy(active_clusters),
                        'nodes': copy.deepcopy(cluster_nodes),
                        'matrix_indices': list(active_clusters.keys()),
                        'merged_pair': (id_f, id_g, id_h)}) # Indicate final trifurcation

    elif DEBUG_PRINT:
         print(f"NJ finished with {len(current_indices)} clusters (expected 3).")


    return history


# --- Animation Setup ---
np.random.seed(42) # for reproducibility
points = np.random.rand(NUM_POINTS, 2) * 10 # Random points in a 10x10 area

print("Running UPGMA...")
upgma_history = run_upgma(points)
# Add summary printouts if DEBUG_PRINT is True

print("\nRunning Neighbor-Joining...")
nj_history = run_nj(points)
# Add summary printouts if DEBUG_PRINT is True


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7)) # Wider figure

# Find overall bounds for consistent axes
all_x = [p[0] for p in points]
all_y = [p[1] for p in points]
x_min, x_max = min(all_x), max(all_x)
y_min, y_max = min(all_y), max(all_y)
# Add padding based on point range, minimum 1 unit
x_range = max(1.0, x_max - x_min)
y_range = max(1.0, y_max - y_min)
padding = max(x_range, y_range) * 0.15 # Increased padding


def draw_tree(ax, nodes_state, original_points, step_num, title, frame_num):
    """Helper function to draw the current state of the tree."""
    ax.clear()
    # Use node info to plot original points (leaves)
    leaf_nodes = {nid: ninfo for nid, ninfo in nodes_state.items() if ninfo.get('original', False)}
    leaf_coords = np.array([ninfo['pos'] for nid, ninfo in leaf_nodes.items() if not np.isnan(ninfo['pos']).any()])
    if leaf_coords.size > 0:
        ax.scatter(leaf_coords[:, 0], leaf_coords[:, 1], c='blue', label='Original Points', zorder=5, s=40)
    elif DEBUG_PRINT:
        print(f" Frame {frame_num} ({title}): No valid leaf coordinates to plot.")


    # Draw connections by iterating through all nodes and drawing line to parent if exists
    drawn_connections = set()
    internal_node_coords = []

    if DEBUG_PRINT: print(f"\n--- Drawing Frame {frame_num} ({title} Step {step_num}) ---")
    if DEBUG_PRINT: print(f"  Nodes available: {list(nodes_state.keys())}")


    for node_id, node_info in nodes_state.items():
        parent_id = node_info.get('parent')

        # Plot internal nodes
        if not node_info.get('original', False):
             node_pos = node_info['pos']
             if not np.isnan(node_pos).any():
                 internal_node_coords.append(node_pos)
             elif DEBUG_PRINT:
                  print(f"  Skipping internal node marker {node_id}: NaN pos {node_pos}")


        # Draw line to parent if parent exists and is valid
        if parent_id is not None and parent_id in nodes_state:
            parent_info = nodes_state[parent_id]
            
            child_pos = node_info['pos']
            parent_pos = parent_info['pos']
            
            connection = tuple(sorted((node_id, parent_id)))

            if connection in drawn_connections:
                continue # Already drawn

            # Check if positions are valid before drawing
            if np.isnan(child_pos).any() or np.isnan(parent_pos).any():
                 if DEBUG_PRINT: print(f"  Skipping line: NaN pos for {connection}. ChildPos:{child_pos}, ParentPos:{parent_pos}")
            else:
                 if DEBUG_PRINT: print(f"  Drawing line: {connection} from {child_pos} to {parent_pos}")
                 ax.plot([parent_pos[0], child_pos[0]],
                         [parent_pos[1], child_pos[1]],
                         color='grey', linestyle='-', linewidth=1.5, zorder=1)
                 drawn_connections.add(connection)

    # Plot internal nodes after lines (so they are on top)
    if internal_node_coords:
        internal_node_coords = np.array(internal_node_coords)
        ax.scatter(internal_node_coords[:, 0], internal_node_coords[:, 1], c='red', marker='s', s=35, zorder=3, alpha=0.8, label='Internal Nodes')


    ax.set_title(f"{title} - Step {step_num}")
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    # ax.set_xlabel("X coordinate") # Keep plots cleaner
    # ax.set_ylabel("Y coordinate")
    ax.set_xticks([]) # Hide ticks
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    # ax.legend() # Optional legend


# Determine the maximum number of steps/frames needed
num_frames = max(len(upgma_history), len(nj_history))

def update(frame):
    # Use print with flush=True for better real-time feedback in some environments
    if DEBUG_PRINT: print(f"\n>>> Generating Frame {frame+1}/{num_frames}", flush=True)

    # Get UPGMA state for this frame (use last state if frame exceeds history)
    upgma_idx = min(frame, len(upgma_history) - 1)
    if upgma_idx < len(upgma_history):
        upgma_state = upgma_history[upgma_idx]
        draw_tree(ax1, upgma_state['nodes'], points, upgma_idx, "UPGMA", frame+1)
    else:
        if DEBUG_PRINT: print(f"Warning: UPGMA index {upgma_idx} out of bounds (len={len(upgma_history)})")
        ax1.clear(); ax1.set_title(f"UPGMA - Step {upgma_idx} (End)")


    # Get NJ state for this frame (use last state if frame exceeds history)
    nj_idx = min(frame, len(nj_history) - 1)
    if nj_idx < len(nj_history):
        nj_state = nj_history[nj_idx]
        draw_tree(ax2, nj_state['nodes'], points, nj_idx, "Neighbor-Joining", frame+1)
    else:
        if DEBUG_PRINT: print(f"Warning: NJ index {nj_idx} out of bounds (len={len(nj_history)})")
        ax2.clear(); ax2.set_title(f"NJ - Step {nj_idx} (End)")


    fig.tight_layout(pad=2.0) # Add padding between subplots


# --- Create and Save/Show Animation ---
# Create animation
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=ANIMATION_INTERVAL, repeat=False)

# Save or show
if SAVE_ANIMATION:
    print(f"Saving animation to {FILENAME}...")
    try:
        writer = None
        fps = max(1, int(1000/ANIMATION_INTERVAL))
        if FILENAME.endswith(".gif"):
             # Make sure Pillow is installed: pip install Pillow
             writer = animation.PillowWriter(fps=fps)
        elif FILENAME.endswith(".mp4"):
             # Ensure ffmpeg is installed and accessible by matplotlib
             # May need: plt.rcParams['animation.ffmpeg_path'] = '/path/to/ffmpeg'
             writer = animation.FFMpegWriter(fps=fps)
        else:
             raise ValueError("Unsupported file extension. Use .gif or .mp4")

        if writer:
            ani.save(FILENAME, writer=writer)
            print(f"Animation saved successfully to {FILENAME}")
        else:
             print("Could not create animation writer.")
             if not DEBUG_PRINT: # Only show plot if not debugging heavily
                  plt.show()

    except Exception as e:
        print(f"\n--- Error saving animation ---")
        print(f"{type(e).__name__}: {e}")
        # print traceback maybe?
        # import traceback
        # traceback.print_exc()
        print("------------------------------")
        print("Trying to show plot instead...")
        # Fallback to showing the plot if saving fails
        # Note: Showing might not work well in all environments after a save attempt
        #       and the animation might need to be regenerated.
        # If saving failed, just showing the static plot might be more reliable.
        try:
            # Attempt to show the first frame as a static plot for feedback
             update(0) # Draw first frame
             plt.show()
        except Exception as show_e:
             print(f"Also failed to show plot: {show_e}")

else:
    print("Showing plot...")
    plt.show()

print("Script finished.")