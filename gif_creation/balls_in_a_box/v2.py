import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import time # For seeding random

# --- Simulation Parameters ---
N_SPECIES_1 = 30     # Number of particles of species 1
N_SPECIES_2 = 20     # Number of particles of species 2
N_PARTICLES = N_SPECIES_1 + N_SPECIES_2

# Species properties [radius, mass, color]
species_props = {
    1: [0.03, 1.0, 'blue'],
    2: [0.05, 2.0, 'green']
}

BOX_SIZE = 1.0       # Size of the square box (side length)
# ---- Reduced DT and increased steps per frame ----
DT = 0.001           # Simulation time step (Smaller for accuracy)
SIM_DURATION = 7.0  # Total simulation time (seconds)
STEPS_PER_FRAME = 10 # More simulation steps per animation frame
# Adjust total frames to match duration
TOTAL_STEPS = int(SIM_DURATION / DT)
FRAMES = int(TOTAL_STEPS / STEPS_PER_FRAME)
# ---- End of changes ----

KB = 1.0             # Boltzmann constant (in simulation units)

# Temperature modulation function (example: ramp up, hold, ramp down)
def target_temperature(t, total_duration):
    ramp_time = total_duration / 3.0
    hold_temp = 3.0 # Target max temp
    initial_temp = 0.5 # Starting temp
    if t < ramp_time:
        # Ramp up
        return initial_temp + (hold_temp - initial_temp) * (t / ramp_time)
    elif t < 2 * ramp_time:
        # Hold
        return hold_temp
    else:
        # Ramp down
        return hold_temp + (initial_temp - hold_temp) * ((t - 2 * ramp_time) / ramp_time)

# --- Initialization ---
np.random.seed(int(time.time())) # Seed random number generator

# Arrays to store particle data
pos = np.zeros((N_PARTICLES, 2)) # x, y positions
vel = np.zeros((N_PARTICLES, 2)) # vx, vy velocities
radii = np.zeros(N_PARTICLES)
masses = np.zeros(N_PARTICLES)
colors = []
species_id = np.zeros(N_PARTICLES, dtype=int)

# Assign properties and initial positions/velocities
idx = 0
initial_T = target_temperature(0, SIM_DURATION)

# Species 1
r1, m1, c1 = species_props[1]
radii[idx:idx+N_SPECIES_1] = r1
masses[idx:idx+N_SPECIES_1] = m1
colors.extend([c1] * N_SPECIES_1)
species_id[idx:idx+N_SPECIES_1] = 1
vel[idx:idx+N_SPECIES_1, :] = np.random.randn(N_SPECIES_1, 2) * np.sqrt(2 * KB * initial_T / m1)
idx += N_SPECIES_1

# Species 2
r2, m2, c2 = species_props[2]
radii[idx:idx+N_PARTICLES] = r2
masses[idx:idx+N_PARTICLES] = m2
colors.extend([c2] * N_SPECIES_2)
species_id[idx:idx+N_PARTICLES] = 2
vel[idx:idx+N_PARTICLES, :] = np.random.randn(N_SPECIES_2, 2) * np.sqrt(2 * KB * initial_T / m2)

# Assign initial positions randomly, trying to avoid overlap
max_radius = np.max(radii)
safe_box_min = max_radius
safe_box_max = BOX_SIZE - max_radius
placed = 0
attempts = 0
max_attempts = N_PARTICLES * 2000 # Increased attempts

while placed < N_PARTICLES and attempts < max_attempts:
    attempts += 1
    p_idx = placed
    pos[p_idx] = np.random.rand(2) * (safe_box_max - safe_box_min) + safe_box_min

    overlap = False
    # Check overlap only with previously placed particles
    for j in range(p_idx):
        dist_sq = np.sum((pos[p_idx] - pos[j])**2)
        min_dist = radii[p_idx] + radii[j]
        # Add a small buffer to prevent starting exactly touching
        if dist_sq < (min_dist + 1e-6)**2 :
            overlap = True
            break
    if not overlap:
        placed += 1

if placed < N_PARTICLES:
     print(f"Warning: Could only place {placed}/{N_PARTICLES} without overlap. Some may start overlapped.")
     for p_idx in range(placed, N_PARTICLES):
         pos[p_idx] = np.random.rand(2) * (safe_box_max - safe_box_min) + safe_box_min


# Center momentum
total_momentum = np.sum(vel * masses[:, np.newaxis], axis=0)
vel -= total_momentum / np.sum(masses)

# --- Simulation Step Functions ---

def calculate_temperature(vel, masses):
    ke = 0.5 * np.sum(masses * np.sum(vel**2, axis=1))
    temp = ke / (N_PARTICLES * KB) # d=2
    return temp

def apply_thermostat(vel, masses, target_T):
    current_T = calculate_temperature(vel, masses)
    if current_T > 1e-8: # Increased tolerance slightly
        scaling_factor = np.sqrt(target_T / current_T)
        vel *= scaling_factor

def advance(pos, vel, radii, masses, dt, box_size):
    """Advance simulation by one time step dt."""
    # 1. Drift particles
    pos += vel * dt

    # 2. Handle Particle-Wall Collisions
    for i in range(N_PARTICLES):
        # Check X boundaries
        if pos[i, 0] - radii[i] < 0:
            pos[i, 0] = radii[i] # Correct position
            vel[i, 0] = -vel[i, 0] # Reflect velocity
        elif pos[i, 0] + radii[i] > box_size:
            pos[i, 0] = box_size - radii[i]
            vel[i, 0] = -vel[i, 0]

        # Check Y boundaries
        if pos[i, 1] - radii[i] < 0:
            pos[i, 1] = radii[i]
            vel[i, 1] = -vel[i, 1]
        elif pos[i, 1] + radii[i] > box_size:
            pos[i, 1] = box_size - radii[i]
            vel[i, 1] = -vel[i, 1]

    # 3. Handle Particle-Particle Collisions
    collided_pairs = set()
    for i in range(N_PARTICLES):
        for j in range(i + 1, N_PARTICLES):
            # Skip if already processed (though set check commented out earlier?)
            # if (i, j) in collided_pairs or (j, i) in collided_pairs : continue

            rij = pos[j] - pos[i]
            dist_sq = np.sum(rij**2)
            sum_radii = radii[i] + radii[j]
            sum_radii_sq = sum_radii**2

            if dist_sq < sum_radii_sq:
                dist = np.sqrt(dist_sq)
                if dist == 0: # Handle exact overlap
                   nudge = (np.random.rand(2) - 0.5) * 1e-6
                   pos[i] -= nudge
                   pos[j] += nudge
                   rij = pos[j] - pos[i]
                   dist_sq = np.sum(rij**2)
                   dist = np.sqrt(dist_sq)
                   if dist == 0: continue # Give up if still overlapped

                # Normal vector (unit vector from i to j)
                normal = rij / dist
                # Relative velocity
                vij = vel[i] - vel[j]
                # Relative velocity along the normal
                v_rel_normal = np.dot(vij, normal)

                # Only resolve collision if particles are moving towards each other
                if v_rel_normal < 0:
                    m1, m2 = masses[i], masses[j]
                    # Impulse magnitude calculation (derived from conservation laws)
                    # J = (2 * m1 * m2) / (m1 + m2) * (-v_rel_normal) # Original impulse definition
                    # Or using the change in velocity formulation directly:
                    impulse_factor = (2.0 * v_rel_normal) / (1.0/m1 + 1.0/m2)

                    # Apply velocity changes due to impulse
                    vel[i] -= (impulse_factor / m1) * normal
                    vel[j] += (impulse_factor / m2) * normal

                    # collided_pairs.add(tuple(sorted((i, j)))) # Mark as collided

                    # ---- Overlap Resolution ----
                    # Calculate overlap amount
                    overlap = sum_radii - dist
                    # Ensure we push them apart slightly more than the overlap
                    # Minimum separation added to prevent floating point issues
                    push_amount = overlap / 2.0 + 1e-9
                    separation_vec = push_amount * normal

                    pos[i] -= separation_vec
                    pos[j] += separation_vec
                    # ---- End Overlap Resolution ----


# --- Animation Setup ---
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(0, BOX_SIZE)
ax.set_ylim(0, BOX_SIZE)
ax.set_aspect('equal', adjustable='box')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("2D Gas Simulation (Repulsive Collisions)")

particles = [patches.Circle(pos[i], radii[i], fc=colors[i]) for i in range(N_PARTICLES)]
for p in particles:
    ax.add_patch(p)

temp_text = ax.text(0.02, 1.02, '', transform=ax.transAxes, fontsize=12)
time_text = ax.text(0.75, 1.02, '', transform=ax.transAxes, fontsize=12)

# --- Animation Function ---
sim_step_counter = 0 # Use a counter for simulation time

def animate(frame):
    global sim_step_counter, pos, vel

    # Calculate target temperature for this frame's approximate time
    # Note: using frame number might drift slightly from actual sim time
    approx_time = frame * STEPS_PER_FRAME * DT
    T_target = target_temperature(approx_time, SIM_DURATION)

    # Run multiple simulation steps per frame
    for _ in range(STEPS_PER_FRAME):
        advance(pos, vel, radii, masses, DT, BOX_SIZE)
        sim_step_counter += 1

    # Apply thermostat (e.g., every frame)
    apply_thermostat(vel, masses, T_target)

    # Update particle positions on the plot
    for i, p in enumerate(particles):
        p.center = pos[i]

    # Update displayed text
    actual_sim_time = sim_step_counter * DT
    temp_text.set_text(f'Target T = {T_target:.2f}')
    time_text.set_text(f'Time = {actual_sim_time:.2f}s')

    return particles + [temp_text, time_text]

# --- Create and Save Animation ---
print(f"Creating animation with DT={DT}, Steps/Frame={STEPS_PER_FRAME}, Total Frames={FRAMES}")
print("This will take some time...")

ani = animation.FuncAnimation(fig, animate, frames=FRAMES,
                              interval=30, blit=True, repeat=False) # interval in ms

gif_filename = 'gas_simulation_repulsive.gif'
try:
    ani.save(gif_filename, writer='pillow', fps=30)
    print(f"Successfully saved animation to {gif_filename}")
except ImportError:
    print("Pillow not found. Install with 'pip install Pillow'. Cannot save GIF.")
    # You might see the plot window if saving fails and blit=True wasn't fully compatible
except Exception as e:
    print(f"Error saving animation: {e}")

# Explicitly close plot if it wasn't closed automatically
plt.close(fig)