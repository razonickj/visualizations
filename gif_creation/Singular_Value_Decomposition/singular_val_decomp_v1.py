import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warnings # To suppress UserWarning about forcing aspect ratio

# --- Configuration ---
# Define the 2x2 transformation matrix A
A = np.array([
    [1, 1.5],
    [0.5, 2]
])

# Animation parameters
N_POINTS = 100 # Number of points on the circle
N_FRAMES_PER_STEP = 50 # Frames for each transformation step (V, Sigma, U)
N_PAUSE_FRAMES = 15    # Frames to pause between steps
FPS = 30               # Frames per second for the output video
OUTPUT_FILENAME = 'svd_visualization.mp4'

# --- SVD Calculation ---
U, s, Vt = np.linalg.svd(A)
Sigma_diag = np.diag(s) # Create the diagonal Sigma matrix

# --- Generate Input Data (Points on a unit circle) ---
thetas = np.linspace(0, 2 * np.pi, N_POINTS)
circle_points = np.vstack([np.cos(thetas), np.sin(thetas)]) # 2xN array

# Standard basis vectors (only need tips, origins are fixed at 0,0)
basis_vector_tips = np.array([[1, 0], [0, 1]]).T # Shape (2, 2), each column is a vector tip

# --- Calculate Transformed Points at Each Stage ---
points_orig = circle_points
vectors_orig_tips = basis_vector_tips

# Stage 1: Apply Vt
points_V = Vt @ points_orig
vectors_V_tips = Vt @ vectors_orig_tips

# Stage 2: Apply Sigma
points_Sigma = Sigma_diag @ points_V
vectors_Sigma_tips = Sigma_diag @ vectors_V_tips

# Stage 3: Apply U (Final = A)
points_U = U @ points_Sigma
vectors_U_tips = U @ vectors_Sigma_tips

# --- Set up Plot ---
fig, ax = plt.subplots(figsize=(8, 8))

# Determine plot limits to encompass all transformed points AND vectors
all_points = np.hstack([points_orig, points_V, points_Sigma, points_U])
all_vectors = np.hstack([vectors_orig_tips, vectors_V_tips, vectors_Sigma_tips, vectors_U_tips])
max_val = np.max(np.abs(np.hstack([all_points, all_vectors]))) * 1.1 # Include vector tips
ax.set_xlim(-max_val, max_val)
ax.set_ylim(-max_val, max_val)

# Setting aspect ratio *after* setting limits can sometimes avoid warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    ax.set_aspect('equal', adjustable='box')

ax.grid(True, linestyle='--', alpha=0.6)
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)


# Artists to update in animation
scatter = ax.scatter([], [], s=10, c='blue', zorder=2) # Plot points above quiver arrows

# Initialize Quiver plot for the two basis vectors
# Origins (X, Y), Components (U, V)
origins_X = [0, 0]
origins_Y = [0, 0]
tips_U = basis_vector_tips[0, :] # Initial dx components [1, 0]
tips_V = basis_vector_tips[1, :] # Initial dy components [0, 1]
quiver = ax.quiver(origins_X, origins_Y, tips_U, tips_V,
                   angles='xy', scale_units='xy', scale=1,
                   color=['red', 'lime'], zorder=3) # lime is brighter green

title = ax.set_title("Initialization")

# --- Animation Update Function ---
total_frames = 3 * N_FRAMES_PER_STEP + 4 * N_PAUSE_FRAMES # V, Sigma, U steps + pauses

def update(frame):
    # Determine current stage and interpolation factor 't'
    stage_idx = frame // (N_FRAMES_PER_STEP + N_PAUSE_FRAMES)
    frame_in_stage = frame % (N_FRAMES_PER_STEP + N_PAUSE_FRAMES)

    stage_title = ""
    points_start, points_end = None, None
    vectors_start_tips, vectors_end_tips = None, None
    t = 0 # Default interpolation factor

    # Determine start/end points/vectors and interpolation factor t for the current frame
    if stage_idx == 0: # Pause Initial -> V
        points_start, points_end = points_orig, points_V
        vectors_start_tips, vectors_end_tips = vectors_orig_tips, vectors_V_tips
        if frame_in_stage < N_PAUSE_FRAMES:
             stage_title = f"Original (Pause {frame_in_stage+1}/{N_PAUSE_FRAMES})"
             t = 0
        else:
             t = (frame_in_stage - N_PAUSE_FRAMES) / N_FRAMES_PER_STEP
             stage_title = f"Applying Vᵀ (Rotation/Reflection)"

    elif stage_idx == 1: # Pause V -> Sigma
        points_start, points_end = points_V, points_Sigma
        vectors_start_tips, vectors_end_tips = vectors_V_tips, vectors_Sigma_tips
        if frame_in_stage < N_PAUSE_FRAMES:
            stage_title = f"After Vᵀ (Pause {frame_in_stage+1}/{N_PAUSE_FRAMES})"
            t = 0 # Stay at end of previous stage (which is start of this stage)
        else:
            t = (frame_in_stage - N_PAUSE_FRAMES) / N_FRAMES_PER_STEP
            stage_title = f"Applying Σ (Scaling σ₁={s[0]:.2f}, σ₂={s[1]:.2f})"

    elif stage_idx == 2: # Pause Sigma -> U
        points_start, points_end = points_Sigma, points_U
        vectors_start_tips, vectors_end_tips = vectors_Sigma_tips, vectors_U_tips
        if frame_in_stage < N_PAUSE_FRAMES:
            stage_title = f"After Σ (Pause {frame_in_stage+1}/{N_PAUSE_FRAMES})"
            t = 0 # Stay at end of previous stage
        else:
            t = (frame_in_stage - N_PAUSE_FRAMES) / N_FRAMES_PER_STEP
            stage_title = f"Applying U (Rotation/Reflection)"

    elif stage_idx == 3: # Pause Final
        # Use the final state from the previous stage
        points_start, points_end = points_U, points_U
        vectors_start_tips, vectors_end_tips = vectors_U_tips, vectors_U_tips
        if frame_in_stage < N_PAUSE_FRAMES:
            stage_title = f"Final Transformation A = UΣVᵀ (Pause {frame_in_stage+1}/{N_PAUSE_FRAMES})"
            t = 0 # Stay at final state
        else: # Hold last frame appearance
            stage_title = f"Final Transformation A = UΣVᵀ"
            t=0 # Ensures we use points_start (which is points_U)

    # Interpolate points and vector *tips*
    # Handle the pause frames correctly where t should be 0
    if stage_idx == 0 and frame_in_stage < N_PAUSE_FRAMES: # Initial state
        current_points = points_start
        current_vector_tips = vectors_start_tips
    elif stage_idx == 1 and frame_in_stage < N_PAUSE_FRAMES: # After V
        current_points = points_start # points_V
        current_vector_tips = vectors_start_tips # vectors_V_tips
    elif stage_idx == 2 and frame_in_stage < N_PAUSE_FRAMES: # After Sigma
        current_points = points_start # points_Sigma
        current_vector_tips = vectors_start_tips # vectors_Sigma_tips
    elif stage_idx == 3: # Final state / Pause
         current_points = points_start # points_U
         current_vector_tips = vectors_start_tips # vectors_U_tips
    else: # Interpolating during transitions
        current_points = (1 - t) * points_start + t * points_end
        current_vector_tips = (1 - t) * vectors_start_tips + t * vectors_end_tips


    # Update plot data
    scatter.set_offsets(current_points.T) # .T because scatter expects Nx2

    # Update Quiver data
    # U, V are the components (dx, dy) of the vectors
    quiver.set_UVC(current_vector_tips[0, :], current_vector_tips[1, :])

    title.set_text(stage_title)

    # Return tuple of artists updated (though not strictly needed for blit=False)
    return scatter, quiver, title


# --- Create and Save Animation ---
print("Creating animation...")
ani = animation.FuncAnimation(fig, update, frames=total_frames,
                              interval=int(1000/FPS), blit=False)

# Save the animation
try:
    print(f"Saving animation to {OUTPUT_FILENAME} (this may take a while)...")
    # Use a specific writer if needed, e.g., Pillow for GIF
    writer = animation.FFMpegWriter(fps=FPS)
    ani.save(OUTPUT_FILENAME, writer=writer)
    print("Animation saved successfully.")
except FileNotFoundError:
    print("\nERROR: FFmpeg not found.")
    print("Please install FFmpeg and ensure it's in your system's PATH.")
    print("Alternatively, try saving as a GIF using:")
    print("  pip install imageio")
    print("  ani.save('svd_visualization.gif', writer='imageio', fps=FPS)")
except Exception as e:
    print(f"\nERROR saving animation: {e}")
    print("Ensure FFmpeg is installed correctly and accessible.")

# plt.show() # Uncomment to display the animation plot window