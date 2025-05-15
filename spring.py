import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib as mpl

# Use TkAgg backend for interactive rotation
mpl.use('TkAgg')

class VoxelTerrainSimulation:
    def __init__(self, grid_size=(100, 50, 20), ball_weight=1.0):
        # Initialize terrain grid
        self.grid_size = grid_size
        self.terrain = np.zeros(grid_size, dtype=bool)
        
        # Create stiffness map for different zones
        self.stiffness_map = np.zeros((grid_size[0], grid_size[1]))
        
        # Divide the terrain into 4 stiffness zones (left to right)
        zone_width = grid_size[0] // 4
        for x in range(grid_size[0]):
            if x < zone_width:  # Very soft zone
                stiffness = 0.1
            elif x < zone_width * 2:  # Medium soft zone
                stiffness = 0.4
            elif x < zone_width * 3:  # Medium hard zone
                stiffness = 0.7
            else:  # Very hard zone
                stiffness = 0.95
                
            for y in range(grid_size[1]):
                self.stiffness_map[x, y] = stiffness
        
        # Create color map for visualization based on stiffness
        self.color_map = np.zeros((grid_size[0], grid_size[1], 3))
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                # Color gradient from soft (light brown) to hard (dark brown)
                stiffness = self.stiffness_map[x, y]
                self.color_map[x, y] = [
                    0.8 - stiffness * 0.4,  # R
                    0.6 - stiffness * 0.4,  # G
                    0.4 - stiffness * 0.3   # B
                ]
        
        # Ball physical properties
        self.ball_weight = ball_weight
        self.ball_radius = 4.0
        self.ball_pos = np.array([5.0, grid_size[1]//2, grid_size[2]*0.5])
        self.ball_velocity = np.array([0.5, 0.1, 0.0])
        self.ball_angular_velocity = np.array([0.0, 0.0, 0.0])
        self.ball_orientation = np.eye(3)  # Rotation matrix
        
        # Physics parameters
        self.gravity = np.array([0.0, 0.0, -0.15 * self.ball_weight])
        self.restitution = 0.3
        self.friction = 0.2
        
        # Terrain setup - Create a flat terrain with varying heights based on stiffness
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                stiffness = self.stiffness_map[x, y]
                base_height = int(grid_size[2] * 0.2)
                noise = int(grid_size[2] * 0.01 * (1 - stiffness) * np.random.randn())
                height = max(1, min(int(grid_size[2] * 0.3), base_height + noise))
                
                for z in range(height):
                    self.terrain[x, y, z] = True
        
        # Tracking for moving/active voxels
        self.active_voxels = set()
        self.voxel_velocities = {}
        self.deformation_trail = set()  # Track deformation for visualization
        
        # Spring physics parameters
        self.spring_constant_base = 0.5  # Base spring constant
        
    def get_local_stiffness(self):
        """Get ground stiffness at the ball's current position"""
        x, y = int(self.ball_pos[0]), int(self.ball_pos[1])
        x = max(0, min(x, self.grid_size[0] - 1))
        y = max(0, min(y, self.grid_size[1] - 1))
        return self.stiffness_map[x, y]
    
    def find_contact_points(self):
        """Find all terrain voxels in contact with the ball"""
        contacts = []
        
        # Define search region around the ball
        search_min = np.maximum(np.floor(self.ball_pos - self.ball_radius - 1).astype(int), [0, 0, 0])
        search_max = np.minimum(np.ceil(self.ball_pos + self.ball_radius + 1).astype(int), 
                              [self.grid_size[0]-1, self.grid_size[1]-1, self.grid_size[2]-1])
        
        for x in range(search_min[0], search_max[0] + 1):
            for y in range(search_min[1], search_max[1] + 1):
                for z in range(search_min[2], search_max[2] + 1):
                    if not self.terrain[x, y, z]:
                        continue
                    
                    voxel_pos = np.array([x, y, z])
                    dist = np.linalg.norm(voxel_pos - self.ball_pos)
                    
                    if dist < self.ball_radius:
                        # Calculate contact normal
                        normal = (self.ball_pos - voxel_pos) / dist
                        contacts.append((voxel_pos, normal, dist))
        
        return contacts
    
    def apply_spring_forces(self, contacts):
        """Apply spring forces to both the ball and terrain voxels"""
        if not contacts:
            return
        
        current_stiffness = self.get_local_stiffness()
        spring_constant = self.spring_constant_base * current_stiffness
        
        total_force = np.zeros(3)
        total_torque = np.zeros(3)
        
        # Keep track of deformed voxels for this frame
        deformed_voxels = set()
        
        for voxel_pos, normal, dist in contacts:
            # Calculate spring force based on Hooke's Law: F = -kx
            # where x is the penetration depth
            penetration = self.ball_radius - dist
            spring_force_magnitude = spring_constant * penetration
            
            # Force vector points along the normal
            spring_force = normal * spring_force_magnitude
            
            # Apply force to ball
            total_force += spring_force
            
            # Calculate torque: τ = r × F
            # r is the vector from center of ball to contact point
            r = (voxel_pos - self.ball_pos) + normal * 0.5  # Offset slightly to prevent singularities
            torque = np.cross(r, spring_force)
            total_torque += torque
            
            # Apply force to terrain voxel (moving it if soft enough)
            voxel_force = -spring_force  # Equal and opposite reaction
            voxel_tuple = tuple(voxel_pos.astype(int))
            
            # Calculate how much the voxel should deform based on stiffness
            # Softer terrain = more deformation
            deformation_factor = (1.0 - current_stiffness) * penetration * 0.8
            
            # Always create some deformation in the terrain, more visible on softer terrain
            if deformation_factor > 0.1:
                # Add this voxel to deformed set
                deformed_voxels.add(voxel_tuple)
                
                # Remove original voxel to create deformation
                self.terrain[voxel_tuple] = False
                
                # Calculate new position (compressed inward)
                compression_vector = normal * deformation_factor
                new_pos = np.array(voxel_pos) - compression_vector
                new_pos_int = tuple(np.round(new_pos).astype(int))
                
                # Ensure new position is valid
                if (0 <= new_pos_int[0] < self.grid_size[0] and
                    0 <= new_pos_int[1] < self.grid_size[1] and
                    0 <= new_pos_int[2] < self.grid_size[2] and
                    not self.terrain[new_pos_int]):
                    
                    # Create new voxel at compressed position
                    self.terrain[new_pos_int] = True
                    
                    # Add to deformation trail for visualization
                    self.deformation_trail.add(new_pos_int)
                    
                    # Mark as active to track physics
                    self.active_voxels.add(new_pos_int)
                    
                    # If very soft terrain, some voxels might actually detach and fly off
                    if current_stiffness < 0.3 and np.random.random() < 0.2:
                        velocity_scale = 0.3 * (1 - current_stiffness)
                        self.voxel_velocities[new_pos_int] = voxel_force * velocity_scale
        
        # Apply accumulated forces to ball
        # F = ma, so acceleration = F/m
        acceleration = total_force / self.ball_weight
        self.ball_velocity += acceleration
        
        # Apply accumulated torque to angular velocity
        # τ = I*α, where I is moment of inertia and α is angular acceleration
        # For a sphere, I = (2/5)*m*r²
        moment_of_inertia = 0.4 * self.ball_weight * self.ball_radius**2
        angular_acceleration = total_torque / moment_of_inertia
        self.ball_angular_velocity += angular_acceleration
    
    def apply_friction(self):
        """Apply friction to ball's motion when in contact with ground"""
        # Only apply friction when ball is near the ground
        terrain_height = self.find_terrain_height_below_ball()
        if self.ball_pos[2] - terrain_height > self.ball_radius + 0.5:
            return
        
        current_stiffness = self.get_local_stiffness()
        
        # Calculate friction coefficient based on terrain stiffness
        # Harder terrain = higher friction
        friction_coef = self.friction * (0.5 + current_stiffness * 0.5)
        
        # Calculate normal force (approximation)
        normal_force = self.ball_weight * 9.8  # weight * gravity
        
        # Friction force magnitude (F = μN)
        friction_force_mag = friction_coef * normal_force
        
        # Get horizontal velocity components
        horizontal_vel = self.ball_velocity.copy()
        horizontal_vel[2] = 0  # Zero out vertical component
        horizontal_speed = np.linalg.norm(horizontal_vel)
        
        if horizontal_speed > 0.01:
            # Direction of friction force (opposite to motion)
            friction_dir = -horizontal_vel / horizontal_speed
            
            # Calculate friction force
            friction_force = friction_dir * min(friction_force_mag, horizontal_speed * 0.1)
            
            # Apply friction force to linear velocity - reduced to prevent stopping
            self.ball_velocity += friction_force
            
            # Calculate rolling torque from friction (this couples linear and angular motion)
            # For a rolling ball, the angular velocity should correspond to the linear velocity
            # w = v/r (for pure rolling)
            rolling_axis = np.cross(np.array([0, 0, 1]), horizontal_vel)
            if np.linalg.norm(rolling_axis) > 0:
                rolling_axis = rolling_axis / np.linalg.norm(rolling_axis)
                
                # For rolling motion: v = ω × r
                # Target angular velocity based on linear speed
                target_angular_vel = rolling_axis * (horizontal_speed / self.ball_radius)
                
                # Apply torque to move toward target angular velocity - stronger coupling
                torque_coefficient = 0.4  # Increased for better rolling
                rolling_torque = torque_coefficient * (target_angular_vel - self.ball_angular_velocity)
                self.ball_angular_velocity += rolling_torque
    
    def find_terrain_height_below_ball(self):
        """Find the height of terrain directly below the ball"""
        x, y = int(self.ball_pos[0]), int(self.ball_pos[1])
        x = max(0, min(x, self.grid_size[0] - 1))
        y = max(0, min(y, self.grid_size[1] - 1))
        
        for z in range(self.grid_size[2]-1, -1, -1):
            if self.terrain[x, y, z]:
                return z
        return 0
    
    def update_active_voxels(self):
        """Update positions of active voxels"""
        voxels_to_remove = []
        
        for voxel, velocity in list(self.voxel_velocities.items()):
            # Apply gravity
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
                    if voxel in self.active_voxels:
                        self.active_voxels.remove(voxel)
                    self.active_voxels.add(new_pos_int)
                    
                    # Apply damping
                    self.voxel_velocities[new_pos_int] = velocity * 0.9
                    voxels_to_remove.append(voxel)
            
            # Remove if stopped or out of bounds
            if velocity.dot(velocity) < 0.01:
                voxels_to_remove.append(voxel)
        
        # Clean up removed voxels
        for voxel in voxels_to_remove:
            if voxel in self.voxel_velocities:
                del self.voxel_velocities[voxel]
    
    def update_ball_rotation(self):
        """Update ball's orientation based on angular velocity"""
        # Calculate rotation angle and axis
        rotation_angle = np.linalg.norm(self.ball_angular_velocity)
        if rotation_angle > 0.001:
            # Normalize rotation axis
            rotation_axis = self.ball_angular_velocity / rotation_angle
            
            # Create rotation matrix (Rodrigues' rotation formula)
            c = np.cos(rotation_angle)
            s = np.sin(rotation_angle)
            t = 1.0 - c
            x, y, z = rotation_axis
            
            rotation_matrix = np.array([
                [t*x*x + c, t*x*y - z*s, t*x*z + y*s],
                [t*x*y + z*s, t*y*y + c, t*y*z - x*s],
                [t*x*z - y*s, t*y*z + x*s, t*z*z + c]
            ])
            
            # Apply rotation to orientation
            self.ball_orientation = np.dot(rotation_matrix, self.ball_orientation)
            
            # Apply rolling resistance based on stiffness
            current_stiffness = self.get_local_stiffness()
            resistance = 0.01 * (1 + current_stiffness)
            self.ball_angular_velocity *= (1.0 - resistance)
    
    def update(self):
        """Update simulation state for one time step"""
        # Find contact points
        contacts = self.find_contact_points()
        
        # Apply spring forces at contacts
        self.apply_spring_forces(contacts)
        
        # Apply friction forces
        self.apply_friction()
        
        # Update ball position
        self.ball_pos += self.ball_velocity
        
        # Apply gravity - reduce gravity effect to prevent bouncing
        self.ball_velocity += self.gravity * 0.7
        
        # Update ball rotation
        self.update_ball_rotation()
        
        # Update active voxels
        self.update_active_voxels()
        
        # Boundary conditions
        for i in range(2):  # For x and y dimensions
            if self.ball_pos[i] < self.ball_radius:
                self.ball_pos[i] = self.ball_radius
                self.ball_velocity[i] *= -0.7  # Bounce
            elif self.ball_pos[i] > self.grid_size[i] - self.ball_radius:
                self.ball_pos[i] = self.grid_size[i] - self.ball_radius
                self.ball_velocity[i] *= -0.7  # Bounce
        
        # Ground collision (to prevent falling through)
        if self.ball_pos[2] < self.ball_radius * 0.7:  # Allow slight sinking for better contact
            self.ball_pos[2] = self.ball_radius * 0.7
            
            # Dampen vertical velocity more to reduce bouncing
            if self.ball_velocity[2] < 0:
                self.ball_velocity[2] = -self.ball_velocity[2] * 0.2  # Reduced bounce
            
            # Apply more damping to vertical movement overall
            self.ball_velocity[2] *= 0.9
        
        # Small nudge to keep ball rolling forward
        if np.linalg.norm(self.ball_velocity[:2]) < 0.3:
            self.ball_velocity[0] += 0.02
            
        # Additional damping to vertical velocity to prevent bouncing
        if abs(self.ball_velocity[2]) < 0.5:
            self.ball_velocity[2] *= 0.8
    
    def get_zone_name(self, x_pos):
        """Return the name of the stiffness zone at a given x position"""
        zone_width = self.grid_size[0] // 4
        if x_pos < zone_width:
            return "Very Soft"
        elif x_pos < zone_width * 2:
            return "Medium Soft" 
        elif x_pos < zone_width * 3:
            return "Medium Hard"
        else:
            return "Very Hard"

# Set up visualization function
def visualize_simulation(sim):
    # Set up the figure and 3D axis
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Turn off axes
    ax.set_axis_off()
    
    # Set good initial view
    ax.view_init(elev=30, azim=45)
    
    # Create zone markers
    zone_width = sim.grid_size[0] // 4
    zone_labels = ['Very Soft', 'Medium Soft', 'Medium Hard', 'Very Hard']
    
    for i in range(4):
        start_x = i * zone_width
        end_x = (i + 1) * zone_width
        mid_x = (start_x + end_x) // 2
        mid_y = sim.grid_size[1] // 2
        
        # Add zone label
        ax.text(mid_x, mid_y, 0, zone_labels[i], 
                ha='center', va='center', fontsize=10, color='black')
        
        # Add zone divider
        if i > 0:
            ax.plot([start_x, start_x], [0, sim.grid_size[1]], [0, 0], 'k--', alpha=0.5)
    
    # Set limits
    ax.set_xlim(0, sim.grid_size[0])
    ax.set_ylim(0, sim.grid_size[1])
    ax.set_zlim(0, sim.grid_size[2])
    ax.set_title('Voxel Terrain Simulation with Spring Physics')
    
    # Status text
    zone_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=10,
                         bbox=dict(facecolor='white', alpha=0.7))
    
    # Sample rate for terrain visualization
    sample_rate = 2
    
    # Function to create terrain surface
    def get_terrain_surface():
        # Create heightmap
        heightmap = np.zeros((sim.grid_size[0], sim.grid_size[1]))
        
        for x in range(sim.grid_size[0]):
            for y in range(sim.grid_size[1]):
                for z in range(sim.grid_size[2]-1, -1, -1):
                    if sim.terrain[x, y, z]:
                        heightmap[x, y] = z
                        break
        
        # Subsample for performance
        X = np.arange(0, sim.grid_size[0], sample_rate)
        Y = np.arange(0, sim.grid_size[1], sample_rate)
        X_grid, Y_grid = np.meshgrid(X, Y)
        Z_grid = heightmap[::sample_rate, ::sample_rate].T
        
        # Create color map
        colors = np.zeros((len(Y), len(X), 3))
        for i, y in enumerate(Y):
            for j, x in enumerate(X):
                if x < sim.grid_size[0] and y < sim.grid_size[1]:
                    colors[i, j] = sim.color_map[int(x), int(y)]
        
        return X_grid, Y_grid, Z_grid, colors
    
    # Initialize empty plots
    terrain_surface = None
    active_scatter = ax.scatter([], [], [], c='red', marker='s', s=25)
    deformation_scatter = ax.scatter([], [], [], c='darkred', marker='s', s=15)
    rigid_body = None
    
    # Animation update function
    def update(frame):
        nonlocal terrain_surface, rigid_body
        
        # Simulate multiple steps per frame for smoother rolling
        for _ in range(2):  # Simulate 2 physics steps per frame
            sim.update()
        
        # Remove old terrain surface
        if terrain_surface:
            terrain_surface.remove()
        
        # Get terrain surface data
        X_grid, Y_grid, Z_grid, colors = get_terrain_surface()
        
        # Plot terrain surface
        terrain_surface = ax.plot_surface(X_grid, Y_grid, Z_grid, 
                                     facecolors=colors,
                                     rstride=1, cstride=1,
                                     linewidth=0,
                                     antialiased=True,
                                     alpha=0.9)
        
        # Update active voxels scatter - these are particles flying around
        active_points = np.array(list(sim.active_voxels)) if sim.active_voxels else np.empty((0, 3))
        if len(active_points) > 0:
            active_scatter._offsets3d = (active_points[:, 0], active_points[:, 1], active_points[:, 2])
        else:
            active_scatter._offsets3d = ([], [], [])
            
        # Add deformation trail to terrain
        # This visualizes the path the ball has taken
        deformation_points = np.array(list(sim.deformation_trail)) if sim.deformation_trail else np.empty((0, 3))
        if len(deformation_points) > 0:
            deformation_scatter._offsets3d = (deformation_points[:, 0], deformation_points[:, 1], deformation_points[:, 2])
        else:
            deformation_scatter._offsets3d = ([], [], [])
        
        # Update ball visualization
        if rigid_body:
            rigid_body.remove()
        
        # Create a texture with grid lines to better visualize rotation
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        
        # Apply rotation matrix to create a properly rotating ball
        # Correct rotation application
        rotation_matrix = sim.ball_orientation
        
        # Create sphere coordinates with rotation
        # Base sphere points (unit sphere)
        sphere_x = np.sin(v) * np.cos(u)
        sphere_y = np.sin(v) * np.sin(u)
        sphere_z = np.cos(v)
        
        # Grid texture for better visualization of rotation
        grid_step = 4
        grid_mask = ((u % grid_step < 0.3) | (v % grid_step < 0.3))
        colors = np.ones(shape=grid_mask.shape + (3,)) * np.array([0.3, 0.3, 0.8])
        colors[grid_mask] = np.array([0.1, 0.1, 0.5])
        
        # Apply the rotation and position to the sphere
        x = sim.ball_pos[0] + sim.ball_radius * sphere_x
        y = sim.ball_pos[1] + sim.ball_radius * sphere_y
        z = sim.ball_pos[2] + sim.ball_radius * sphere_z
        
        rigid_body = ax.plot_surface(x, y, z, facecolors=colors, alpha=0.7)
        
        # Update status text
        current_zone = sim.get_zone_name(sim.ball_pos[0])
        current_stiffness = sim.get_local_stiffness()
        velocity = np.linalg.norm(sim.ball_velocity)
        angular_velocity = np.linalg.norm(sim.ball_angular_velocity)
        
        zone_text.set_text(f"Zone: {current_zone}\nStiffness: {current_stiffness:.2f}\n"
                          f"Speed: {velocity:.2f}\nSpin: {angular_velocity:.2f}")
        
        # Update title
        ax.set_title(f'Ball Rolling with Spring Physics - Frame: {frame}, Active Voxels: {len(sim.active_voxels)}')
        
        return terrain_surface, active_scatter, deformation_scatter, rigid_body, zone_text
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=600, interval=50, blit=False)
    
    # Enable interactive mode
    plt.ion()
    plt.tight_layout()
    plt.show()
    
    # Keep plot window open
    try:
        plt.pause(600)  # Run for 10 minutes
    except KeyboardInterrupt:
        print("Animation stopped by user")

# Create and run simulation
sim = VoxelTerrainSimulation(grid_size=(100, 50, 20), ball_weight=1.0)
visualize_simulation(sim)