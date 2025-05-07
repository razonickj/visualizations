import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Constants ---
# From previous script (vector doesn't matter, just its initial magnitude)
INITIAL_VECTOR = np.array([2.5, 1.0])
SCALE_FACTOR = 0.97  # Decay factor per step (< 1)
NUM_ITERATIONS = 75 # Fewer iterations needed to see decay clearly on cobweb
INTERVAL = 100       # Milliseconds between frames
FILENAME = "cobweb_decay.gif"

# --- Setup ---
r0 = np.linalg.norm(INITIAL_VECTOR) # Initial magnitude
s = SCALE_FACTOR

# Pre-calculate magnitude sequence
r_sequence = [r0]
for _ in range(NUM_ITERATIONS):
    r_sequence.append(s * r_sequence[-1])
r_sequence = np.array(r_sequence)

# --- Plot Setup ---
fig, ax = plt.subplots(figsize=(7, 7))
limit = r0 * 1.1 # Axis limit based on initial magnitude
ax.set_xlim(0, limit)
ax.set_ylim(0, limit)
ax.set_aspect('equal', adjustable='box')
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_xlabel('Magnitude r_n')
ax.set_ylabel('Magnitude r_{n+1}')
ax.set_title('Cobweb Plot: Magnitude Decay r_{n+1} = %.2f * r_n' % s)

# Plot y = s*x (the function)
x_func = np.array([0, limit])
y_func = s * x_func
ax.plot(x_func, y_func, 'r-', lw=2, label=f'r_{{n+1}} = {s:.2f} * r_n')

# Plot y = x line
ax.plot([0, limit], [0, limit], 'k--', lw=1, label='r_{n+1} = r_n')

# Initialize cobweb line (empty at first)
cobweb_line, = ax.plot([], [], 'b-', lw=1.5, alpha=0.8)

# Initialize text for current magnitude display
mag_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, va='top')

# --- Animation Function ---
def update(n): # n goes from 0 to NUM_ITERATIONS - 1
    # Build the path points up to iteration n
    x_path = []
    y_path = []
    # Start point conceptually on y=x
    x_path.append(r_sequence[0])
    y_path.append(r_sequence[0])
    for i in range(n + 1): # Iterate up to n to include n->n+1 step
        rn = r_sequence[i]
        rn1 = r_sequence[i+1] # We precalculated one extra point

        # Vertical segment: (rn, rn) -> (rn, rn1)
        x_path.append(rn)
        y_path.append(rn1)

        # Horizontal segment: (rn, rn1) -> (rn1, rn1)
        x_path.append(rn1)
        y_path.append(rn1)

    # Update the cobweb line data
    cobweb_line.set_data(x_path, y_path)

    # Update text display
    current_r = r_sequence[n]
    mag_text.set_text(f'Iteration: {n}\nr_{n} ≈ {current_r:.3f}')

    return cobweb_line, mag_text

# --- Create Animation ---
print("Creating cobweb animation...")
# Frames = number of iterations
ani = animation.FuncAnimation(fig, update, frames=NUM_ITERATIONS,
                                interval=INTERVAL, blit=True, repeat=False)

# --- Save ---
try:
    print(f"Saving animation to {FILENAME}...")
    writer = animation.PillowWriter(fps=1000/INTERVAL)
    ani.save(FILENAME, writer=writer)
    print(f"Animation saved successfully to {FILENAME}")
except Exception as e:
    print(f"\n--- Error saving animation ---\n{type(e).__name__}: {e}\n------------------------------")
    # plt.show()

# plt.show()
print("Script finished.")

