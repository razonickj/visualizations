import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Example data points in a 2D plane
data_points = np.array([
    [1, 1],
    [2, 1],
    [4, 3],
    [5, 4],
    [3, 5]
])

# Placeholder functions for UPGMA and NJ tree construction
def upgma_tree_construction(data):
    """
    Perform UPGMA tree construction.
    :param data: A 2D array of points (used to calculate pairwise distances).
    :return: A list of steps showing the clustering process.
    """
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage

    # Calculate pairwise distances (condensed distance matrix)
    distance_matrix = pdist(data, metric='euclidean')

    # Perform UPGMA clustering
    clusters = linkage(distance_matrix, method='average')

    # Simulate steps for visualization
    steps = []
    current_clusters = [[point] for point in data]
    for i, (c1, c2, dist, _) in enumerate(clusters):
        c1, c2 = int(c1), int(c2)
        new_cluster = np.mean(current_clusters[c1] + current_clusters[c2], axis=0)
        current_clusters.append([new_cluster])
        steps.append(np.array([np.mean(cluster, axis=0) for cluster in current_clusters if len(cluster) > 0]))

    return steps


def nj_tree_construction(data):
    """
    Perform Neighbor Joining tree construction.
    :param data: A 2D array of points (used to calculate pairwise distances).
    :return: A list of steps showing the clustering process.
    """
    from scipy.spatial.distance import pdist, squareform
    import numpy as np

    # Calculate pairwise distances (square distance matrix)
    distance_matrix = squareform(pdist(data, metric='euclidean'))

    # Initialize clusters
    clusters = {i: [i] for i in range(len(data))}
    steps = [data.copy()]

    while len(clusters) > 1:
        # Compute Q-matrix
        n = len(clusters)
        q_matrix = np.zeros((n, n))
        keys = list(clusters.keys())  # Update keys to reflect current cluster indices
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    q_matrix[i, j] = (n - 2) * distance_matrix[i, j] - \
                                     sum(distance_matrix[i, k] for k in range(n)) - \
                                     sum(distance_matrix[j, k] for k in range(n))

        # Find the pair with the smallest Q-value
        i, j = np.unravel_index(np.argmin(q_matrix), q_matrix.shape)
        cluster_i, cluster_j = keys[i], keys[j]

        # Store the clusters to merge before removing them
        merged_clusters = clusters[cluster_i] + clusters[cluster_j]
        
        # Remove the old clusters
        clusters.pop(cluster_i)
        clusters.pop(cluster_j)
        
        # Add the new merged cluster with a new key
        new_key = max(keys) + 1 if keys else 0
        clusters[new_key] = merged_clusters

        # Calculate new cluster centroid
        new_cluster = np.mean([data[idx] for idx in merged_clusters], axis=0)

        # Update distance matrix
        new_distances = []
        for k in range(n):
            if k != i and k != j:
                dist = (distance_matrix[i, k] + distance_matrix[j, k] -
                        distance_matrix[i, j]) / 2
                new_distances.append(dist)
        distance_matrix = np.delete(distance_matrix, [i, j], axis=0)
        distance_matrix = np.delete(distance_matrix, [i, j], axis=1)
        distance_matrix = np.vstack([distance_matrix, new_distances])
        distance_matrix = np.hstack([distance_matrix, np.append(new_distances, 0).reshape(-1, 1)])

        # Record step
        steps.append(np.array([np.mean(data[cluster], axis=0) for cluster in clusters.values()]))

    return steps

# Generate tree construction steps
upgma_steps = upgma_tree_construction(data_points)
nj_steps = nj_tree_construction(data_points)

# Create the animation
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
upgma_ax, nj_ax = axes
upgma_ax.set_title("UPGMA")
nj_ax.set_title("Neighbor Joining")

def update(frame):
    upgma_ax.clear()
    nj_ax.clear()
    upgma_ax.set_title("UPGMA")
    nj_ax.set_title("Neighbor Joining")
    
    # Plot original points
    upgma_ax.scatter(data_points[:, 0], data_points[:, 1], color='gray', alpha=0.5, label='Original Points')
    nj_ax.scatter(data_points[:, 0], data_points[:, 1], color='gray', alpha=0.5, label='Original Points')
    
    # Plot current step points
    upgma_ax.scatter(upgma_steps[frame][:, 0], upgma_steps[frame][:, 1], color='blue', label='Current Nodes')
    nj_ax.scatter(nj_steps[frame][:, 0], nj_steps[frame][:, 1], color='green', label='Current Nodes')
    
    # Draw branches (lines) connecting nodes
    if frame > 0:
        for i in range(len(upgma_steps[frame])):
            upgma_ax.plot(
                [upgma_steps[frame - 1][i, 0], upgma_steps[frame][i, 0]],
                [upgma_steps[frame - 1][i, 1], upgma_steps[frame][i, 1]],
                color='blue', linestyle='--'
            )
        for i in range(len(nj_steps[frame])):
            nj_ax.plot(
                [nj_steps[frame - 1][i, 0], nj_steps[frame][i, 0]],
                [nj_steps[frame - 1][i, 1], nj_steps[frame][i, 1]],
                color='green', linestyle='--'
            )
    
    # Add legends
    upgma_ax.legend()
    nj_ax.legend()

ani = animation.FuncAnimation(fig, update, frames=len(upgma_steps), repeat=False)

# Save the animation as a movie file
ani.save("upgma_vs_nj_comparison.mp4", writer="ffmpeg", fps=1)

plt.show()