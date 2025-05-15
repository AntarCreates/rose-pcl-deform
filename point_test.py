import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from matplotlib.animation import FuncAnimation
import matplotlib as mpl

# For better interactive performance
mpl.use('TkAgg')  # Use TkAgg backend for interactive rotation

class VoxelTerrainSimulation:
    def __init__(self, grid_size=(50, 50, 20), voxel_size=.01):
        # Initialize terrain grid
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.terrain = np.zeros(grid_size, dtype=bool)
        
        # Create a simple terrain surface (just an example)
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                height = int(grid_size[2] * 0.3 + 
                          grid_size[2] * 0.1 * np.sin(x/10) * np.cos(y/10))
                for z in range(height):
                    self.terrain[x, y, z] = True
        
        # Rigid body properties
        self.rigid_body_pos = np.array([grid_size[0]//2, grid_size[1]//2, grid_size[2]*0.7])
        self.rigid_body_radius = 5.0
        self.rigid_body_velocity = np.array([0.0, 0.0, -0.5])
        
        # Active voxels for simulation
        self.active_voxels = set()
        self.voxel_velocities = {}
        
        # Simulation parameters
        self.gravity = np.array([0.0, 0.0, -0.2])
        self.restitution = 0.3
        self.active_radius = self.rigid_body_radius * 1.5
        
        # Identify initial active voxels
        self.update_active_voxels()
        
        # Animation properties
        self.frame_count = 0
        
    def update_active_voxels(self):
        """Find voxels that are within the active radius of the rigid body"""
        new_active_voxels = set()
        
        # Define search region (optimization to avoid checking every voxel)
        search_min = np.maximum(np.floor(self.rigid_body_pos - self.active_radius).astype(int), [0, 0, 0])
        search_max = np.minimum(np.ceil(self.rigid_body_pos + self.active_radius).astype(int), 
                                [self.grid_size[0]-1, self.grid_size[1]-1, self.grid_size[2]-1])
        
        # Check voxels within this region
        for x in range(search_min[0], search_max[0] + 1):
            for y in range(search_min[1], search_max[1] + 1):
                for z in range(search_min[2], search_max[2] + 1):
                    if not self.terrain[x, y, z]:
                        continue
                    
                    voxel_pos = np.array([x, y, z])
                    dist = np.linalg.norm(voxel_pos - self.rigid_body_pos)
                    
                    if dist < self.active_radius:
                        new_active_voxels.add((x, y, z))
                        # Initialize velocity for new active voxels
                        if (x, y, z) not in self.voxel_velocities:
                            self.voxel_velocities[(x, y, z)] = np.array([0.0, 0.0, 0.0])
        
        # Remove velocities for voxels that are no longer active
        voxels_to_remove = []
        for voxel in self.voxel_velocities:
            if voxel not in new_active_voxels:
                voxels_to_remove.append(voxel)
                
        for voxel in voxels_to_remove:
            del self.voxel_velocities[voxel]
        
        # Update active voxels set
        self.active_voxels = new_active_voxels
    
    def check_rigid_body_collision(self):
        """Check and resolve collisions between rigid body and voxels"""
        collision_detected = False
        
        # Use a list copy to avoid modification during iteration
        active_voxels_list = list(self.active_voxels)
        
        for voxel in active_voxels_list:
            x, y, z = voxel
            # Skip if voxel is out of bounds
            if (x < 0 or x >= self.grid_size[0] or
                y < 0 or y >= self.grid_size[1] or
                z < 0 or z >= self.grid_size[2]):
                continue
                
            # Skip if voxel is no longer part of terrain
            if not self.terrain[x, y, z]:
                continue
                
            voxel_pos = np.array(voxel)
            dist = np.linalg.norm(voxel_pos - self.rigid_body_pos)
            
            # Check for collision
            if dist < self.rigid_body_radius:
                collision_detected = True
                
                # Calculate collision normal
                collision_normal = (voxel_pos - self.rigid_body_pos) / (dist + 1e-10)  # Avoid division by zero
                
                # Apply impulse to voxel
                self.voxel_velocities[voxel] = self.rigid_body_velocity * self.restitution
                
                # Add some random displacement for more natural behavior
                self.voxel_velocities[voxel] += np.random.uniform(-0.1, 0.1, 3)
                
                # Flag this voxel as no longer part of the terrain
                self.terrain[x, y, z] = False
        
        # Slow down rigid body if collision occurred
        if collision_detected:
            self.rigid_body_velocity *= 0.95
    
    def update_voxel_positions(self):
        """Update positions of active voxels based on their velocities"""
        voxels_to_remove = []
        voxels_to_add = []
        
        for voxel, velocity in list(self.voxel_velocities.items()):
            x, y, z = voxel
            
            # Skip if out of bounds
            if (x < 0 or x >= self.grid_size[0] or
                y < 0 or y >= self.grid_size[1] or
                z < 0 or z >= self.grid_size[2]):
                voxels_to_remove.append(voxel)
                continue
                
            # Check if the voxel is part of the terrain
            if not self.terrain[x, y, z]:  # This voxel has been dislodged
                # Update velocity (apply gravity)
                velocity += self.gravity
                
                # Calculate new position
                new_pos = np.array(voxel) + velocity
                new_pos_int = tuple(np.round(new_pos).astype(int))
                
                # Check if new position is valid
                if (0 <= new_pos_int[0] < self.grid_size[0] and
                    0 <= new_pos_int[1] < self.grid_size[1] and
                    0 <= new_pos_int[2] < self.grid_size[2] and
                    new_pos_int != voxel):
                    
                    # Move to new position if empty
                    if not self.terrain[new_pos_int]:
                        self.terrain[new_pos_int] = True
                        voxels_to_remove.append(voxel)
                        voxels_to_add.append(new_pos_int)
                        self.voxel_velocities[new_pos_int] = velocity * 0.98  # Some damping
                
                # If voxel goes out of bounds or hasn't moved
                if velocity.dot(velocity) < 0.01:  # Almost stopped
                    voxels_to_remove.append(voxel)
        
        # Clean up voxels that have been moved or stopped
        for voxel in voxels_to_remove:
            if voxel in self.voxel_velocities:
                del self.voxel_velocities[voxel]
            if voxel in self.active_voxels:
                self.active_voxels.remove(voxel)
        
        # Add new voxels
        for voxel in voxels_to_add:
            self.active_voxels.add(voxel)
    
    def update_rigid_body(self):
        """Update rigid body position based on velocity"""
        self.rigid_body_pos += self.rigid_body_velocity
        
        # Simple gravity
        self.rigid_body_velocity += self.gravity
        
        # Simple ground collision for the rigid body
        if self.rigid_body_pos[2] - self.rigid_body_radius < 0:
            self.rigid_body_pos[2] = self.rigid_body_radius
            self.rigid_body_velocity[2] = -self.rigid_body_velocity[2] * 0.5  # Bounce
            
        # Add some random horizontal movement to make it more interesting
        if np.random.random() < 0.05:  # 5% chance each frame
            self.rigid_body_velocity[0] += np.random.uniform(-0.1, 0.1)
            self.rigid_body_velocity[1] += np.random.uniform(-0.1, 0.1)
            
        # Boundary conditions to keep the ball in the grid
        margin = 5
        for i in range(2):  # For x and y dimensions
            if self.rigid_body_pos[i] < margin:
                self.rigid_body_pos[i] = margin
                self.rigid_body_velocity[i] *= -0.8  # Bounce off wall
            elif self.rigid_body_pos[i] > self.grid_size[i] - margin:
                self.rigid_body_pos[i] = self.grid_size[i] - margin
                self.rigid_body_velocity[i] *= -0.8  # Bounce off wall
    
    def simulate_step(self):
        """Perform one simulation step"""
        # Update which voxels are active
        self.update_active_voxels()
        
        # Check for and resolve collisions
        self.check_rigid_body_collision()
        
        # Update positions
        self.update_voxel_positions()
        
        # Update rigid body
        self.update_rigid_body()
        
        # Update frame count
        self.frame_count += 1

# Initialize the simulation
sim = VoxelTerrainSimulation()

# Set up the figure and 3D axis for animation
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Set up a good initial view
ax.view_init(elev=30, azim=45)

# Initialize empty scatter plots for terrain and active voxels
terrain_scatter = ax.scatter([], [], [], c='brown', marker='s', alpha=0.5, s=20)
active_scatter = ax.scatter([], [], [], c='red', marker='s', s=25)

# Initialize sphere plot for rigid body
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = y = z = np.zeros_like(u)
# Instead of wireframe, use a surface plot with empty arrays initially
rigid_body = ax.plot_surface(x, y, z, color='blue', alpha=0.3)

# Set labels and limits
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(0, sim.grid_size[0])
ax.set_ylim(0, sim.grid_size[1])
ax.set_zlim(0, sim.grid_size[2])
ax.set_title('Voxel Terrain Simulation with Granular Mechanics\nUse mouse to rotate view', fontsize=12)

# Add a text annotation explaining controls
fig.text(0.5, 0.01, 'Click and drag to rotate | Scroll to zoom', 
         ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.7))

# Sample rate for terrain visualization (for performance)
sample_rate = 3

# Function to sample terrain points efficiently
def get_terrain_points():
    terrain_points = []
    # Only plot visible terrain (top surface)
    for x in range(0, sim.grid_size[0], sample_rate):
        for y in range(0, sim.grid_size[1], sample_rate):
            for z in range(sim.grid_size[2]-1, -1, -1):
                if sim.terrain[x, y, z]:
                    terrain_points.append([x, y, z])
                    break  # Only add the top visible point
    
    return np.array(terrain_points) if terrain_points else np.empty((0, 3))

# Animation update function
def update(frame):
    global rigid_body
    
    # Simulate one step
    sim.simulate_step()
    
    # Get terrain points
    terrain_points = get_terrain_points()
    
    # Update terrain scatter plot
    if len(terrain_points) > 0:
        terrain_scatter._offsets3d = (terrain_points[:, 0], terrain_points[:, 1], terrain_points[:, 2])
    else:
        terrain_scatter._offsets3d = ([], [], [])
    
    # Update active voxels scatter plot
    active_points = np.array(list(sim.active_voxels)) if sim.active_voxels else np.empty((0, 3))
    if len(active_points) > 0:
        active_scatter._offsets3d = (active_points[:, 0], active_points[:, 1], active_points[:, 2])
    else:
        active_scatter._offsets3d = ([], [], [])
    
    # Update rigid body - remove the old surface first
    if rigid_body:
        rigid_body.remove()
    
    # Create new surface for rigid body
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = sim.rigid_body_pos[0] + sim.rigid_body_radius * np.cos(u) * np.sin(v)
    y = sim.rigid_body_pos[1] + sim.rigid_body_radius * np.sin(u) * np.sin(v)
    z = sim.rigid_body_pos[2] + sim.rigid_body_radius * np.cos(v)
    rigid_body = ax.plot_surface(x, y, z, color='blue', alpha=0.3)
    
    # Update title with info
    ax.set_title(f'Frame: {frame}, Active Voxels: {len(sim.active_voxels)}', fontsize=12)
    
    return terrain_scatter, active_scatter, rigid_body

# Create animation
ani = FuncAnimation(fig, update, frames=300, interval=50, blit=False)

# Enable interactive mode
plt.ion()
plt.tight_layout()
plt.show()

# This keeps the plot window open
try:
    plt.pause(300)  # Keep animation running for 5 minutes
except KeyboardInterrupt:
    print("Animation stopped by user")