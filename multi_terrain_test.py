import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from matplotlib.animation import FuncAnimation
import matplotlib as mpl

# HYPERPARAMETERS
LINEAR_VEL = 2.0  # Linear velocity of the ball

# For better interactive performance
mpl.use('TkAgg')  # Use TkAgg backend for interactive rotation

class VoxelTerrainSimulation:
    def __init__(self, grid_size=(100, 50, 20), voxel_size=.01, ball_weight=1.0):
        # Initialize terrain grid
        self.grid_size = grid_size
        self.voxel_size = voxel_size
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
        
        # Added parameters for ground stiffness and ball weight
        self.ground_stiffness = 0.5  # Default, will be overridden by the map
        self.ball_weight = ball_weight  # Relative weight (affects momentum and terrain deformation)
        
        # Create a flat terrain surface with slight noise
        # base_height = int(grid_size[2] * 0.2)
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                # Base height varies slightly by stiffness - harder terrain is higher
                stiffness = self.stiffness_map[x, y]
                base_height = int(grid_size[2] * (0.18 + stiffness * 0.05))
                
                # Add noise to height based on stiffness
                noise_factor = 0.01 * (1 - stiffness)
                height = base_height + int(grid_size[2] * noise_factor * np.random.randn())
                height = max(1, min(int(grid_size[2] * 0.3), height))
                
                for z in range(height):
                    self.terrain[x, y, z] = True
        
        # Rigid body properties
        self.rigid_body_pos = np.array([5.0, grid_size[1]//2, grid_size[2]*0.5])
        self.rigid_body_radius = 4.0
        self.rigid_body_velocity = np.array([LINEAR_VEL, 0.1, 0.0])  # Initial velocity for rolling
        
        # Ball physical properties
        self.rigid_body_mass = ball_weight
        self.rigid_body_angular_velocity = np.array([0.0, 0.0, 0.0])
        self.rigid_body_orientation = np.eye(3)  # Rotation matrix for the ball
        
        # Active voxels for simulation
        self.active_voxels = set()
        self.voxel_velocities = {}
        
        # Simulation parameters - modified to use ball weight
        self.base_gravity = 0.15
        self.gravity = np.array([0.0, 0.0, -self.base_gravity * self.ball_weight])
        self.restitution = 0.3  # Base restitution value
        self.friction = 0.2     # Base friction coefficient
        self.base_active_radius = self.rigid_body_radius * 1.5
        
        # Physics constraints
        self.max_angular_velocity = 0.5
        self.rolling_resistance = 0.01  # Base resistance to rolling
        
        # Identify initial active voxels
        self.update_active_voxels()
        
        # Animation properties
        self.frame_count = 0
        
    def get_local_stiffness(self):
        """Get ground stiffness at the ball's current position"""
        x, y = int(self.rigid_body_pos[0]), int(self.rigid_body_pos[1])
        x = max(0, min(x, self.grid_size[0] - 1))
        y = max(0, min(y, self.grid_size[1] - 1))
        return self.stiffness_map[x, y]
        
    def update_active_voxels(self):
        """Find voxels that are within the active radius of the rigid body"""
        current_stiffness = self.get_local_stiffness()
        self.active_radius = self.base_active_radius * (1.0 + 0.5 * (1.0 - current_stiffness))
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
        collision_count = 0
        total_normal = np.zeros(3)
        total_impulse = np.zeros(3)
        
        # Get local ground stiffness
        current_stiffness = self.get_local_stiffness()
        
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
            displacement = voxel_pos - self.rigid_body_pos
            dist = np.linalg.norm(displacement)
            
            # Check for collision
            if dist < self.rigid_body_radius:
                collision_detected = True
                collision_count += 1
                
                # Calculate collision normal
                collision_normal = displacement / (dist + 1e-10)  # Avoid division by zero
                total_normal += collision_normal
                
                # Calculate penetration depth
                penetration = self.rigid_body_radius - dist
                
                # Calculate impact velocity
                impact_velocity = self.rigid_body_velocity.dot(collision_normal)
                
                # Only displace voxels if ground is soft enough and impact is significant
                voxel_displacement_threshold = 0.1 * (1 - current_stiffness) * self.ball_weight
                
                if impact_velocity > voxel_displacement_threshold:
                    # Apply impulse to voxel - modified based on ground stiffness
                    voxel_impulse = impact_velocity * self.ball_weight * (1 - current_stiffness)
                    self.voxel_velocities[voxel] = collision_normal * voxel_impulse * self.restitution
                    
                    # Add slight random displacement for more natural behavior
                    random_factor = 0.05 * (1 - current_stiffness)
                    self.voxel_velocities[voxel] += np.random.uniform(-random_factor, random_factor, 3)
                    
                    # For softer ground, displace voxels based on stiffness
                    if np.random.random() > current_stiffness:
                        # Flag this voxel as no longer part of the terrain
                        self.terrain[x, y, z] = False
                
                # Calculate impulse for ball based on stiffness
                restitution = self.restitution * (0.5 + current_stiffness * 0.5)
                impulse_magnitude = penetration * current_stiffness * 0.1 + impact_velocity * restitution
                impulse = collision_normal * impulse_magnitude
                total_impulse += impulse
                
                # Calculate torque from collision point
                torque = np.cross(displacement, impulse)
                
                # Update ball's angular velocity
                angular_impulse = torque * 0.2 / self.rigid_body_mass
                self.rigid_body_angular_velocity += angular_impulse
        
        # Apply aggregated impulse to rigid body if collision occurred
        if collision_detected and collision_count > 0:
            # Normalize the response
            avg_impulse = total_impulse / collision_count
            
            # Apply the impulse to change velocity
            self.rigid_body_velocity += avg_impulse
            
            # Apply friction based on stiffness
            friction_coefficient = self.friction * (1 + current_stiffness)
            
            # Calculate tangential velocity component (parallel to surface)
            normal_dir = total_normal / (np.linalg.norm(total_normal) + 1e-10)
            normal_vel = np.dot(self.rigid_body_velocity, normal_dir) * normal_dir
            tangent_vel = self.rigid_body_velocity - normal_vel
            
            # Apply friction to tangential velocity
            if np.linalg.norm(tangent_vel) > 0.001:
                friction_dir = -tangent_vel / np.linalg.norm(tangent_vel)
                friction_magnitude = min(np.linalg.norm(tangent_vel), 
                                      friction_coefficient * np.linalg.norm(normal_vel))
                friction_impulse = friction_dir * friction_magnitude
                
                self.rigid_body_velocity += friction_impulse
    
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
                        
                        # Apply damping
                        self.voxel_velocities[new_pos_int] = velocity * 0.95
                
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
            
    def apply_rolling(self):
        """Apply rolling physics to the ball based on its angular velocity"""
        # Get unit vectors for ball's local coordinate system
        x_axis = self.rigid_body_orientation[0]
        y_axis = self.rigid_body_orientation[1]
        z_axis = self.rigid_body_orientation[2]
        
        # Update orientation based on angular velocity
        rotation_angle = np.linalg.norm(self.rigid_body_angular_velocity)
        if rotation_angle > 0.001:
            # Normalize axis of rotation
            rotation_axis = self.rigid_body_angular_velocity / rotation_angle
            
            # Create rotation matrix
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
            self.rigid_body_orientation = np.dot(rotation_matrix, self.rigid_body_orientation)
        
        # Calculate rolling resistance based on local stiffness
        current_stiffness = self.get_local_stiffness()
        resistance_factor = self.rolling_resistance * (1 + current_stiffness * 2)
        
        # Apply rolling resistance to angular velocity
        if np.linalg.norm(self.rigid_body_angular_velocity) > 0:
            self.rigid_body_angular_velocity *= (1.0 - resistance_factor)
        
        # Calculate rolling-induced linear velocity on flat surface
        if self.rigid_body_pos[2] <= self.rigid_body_radius + 0.5:
            # Cross product of angular velocity and up vector
            rolling_velocity = np.cross(self.rigid_body_angular_velocity, [0, 0, 1])
            rolling_velocity *= self.rigid_body_radius
            
            # Blend between current velocity and rolling velocity
            blend_factor = 0.1
            self.rigid_body_velocity = (1 - blend_factor) * self.rigid_body_velocity + blend_factor * rolling_velocity
    
    def update_rigid_body(self):
        """Update rigid body position based on velocity"""
        # Store previous position for contact detection
        prev_pos = np.copy(self.rigid_body_pos)
        
        # Apply velocity to position
        self.rigid_body_pos += self.rigid_body_velocity
        
        # Apply gravity
        self.rigid_body_velocity += self.gravity
        
        # Get local ground properties
        current_stiffness = self.get_local_stiffness()
        
        # Ground collision detection and response
        # Question: ML to tune these hyperparameters?
        if self.rigid_body_pos[2] < self.rigid_body_radius:
            # Calculate penetration depth
            penetration = self.rigid_body_radius - self.rigid_body_pos[2]
            
            # Adjust position to prevent excessive sinking
            # More stiffness = less penetration - MODIFY THIS LINE
            adjustment_factor = 0.2 + current_stiffness * 0.8  # Changed from 0.8 + current_stiffness * 0.2
            self.rigid_body_pos[2] += penetration * adjustment_factor
            
            # Calculate bounce factor based on stiffness
            bounce_factor = 0.2 + current_stiffness * 0.4
            
            # Only bounce if velocity is significant
            if self.rigid_body_velocity[2] < -0.1:
                self.rigid_body_velocity[2] = -self.rigid_body_velocity[2] * bounce_factor
            else:
                self.rigid_body_velocity[2] = 0
            
            # Calculate friction based on stiffness
            friction = 0.01 + current_stiffness * 0.04
            
            # Apply friction to horizontal movement
            horizontal_speed = np.linalg.norm(self.rigid_body_velocity[:2])
            if horizontal_speed > 0.001:
                friction_decel = friction * 9.8 * self.ball_weight  # F = μmg
                friction_decel = min(friction_decel, horizontal_speed)  # Can't reverse direction
                
                direction = self.rigid_body_velocity[:2] / horizontal_speed
                self.rigid_body_velocity[0] -= direction[0] * friction_decel
                self.rigid_body_velocity[1] -= direction[1] * friction_decel
            
            # Calculate and apply angular velocity from linear velocity on contact
            if np.linalg.norm(self.rigid_body_velocity[:2]) > 0.01:
                # Rolling direction is perpendicular to velocity
                roll_axis = np.array([-self.rigid_body_velocity[1], self.rigid_body_velocity[0], 0])
                roll_axis_len = np.linalg.norm(roll_axis)
                
                if roll_axis_len > 0:
                    roll_axis = roll_axis / roll_axis_len
                    linear_speed = np.linalg.norm(self.rigid_body_velocity[:2])
                    angular_speed = linear_speed / self.rigid_body_radius
                    
                    # Apply some portion of the calculated angular velocity
                    target_angular_vel = roll_axis * angular_speed
                    blend_factor = 0.2
                    self.rigid_body_angular_velocity = (1 - blend_factor) * self.rigid_body_angular_velocity + blend_factor * target_angular_vel
            
            # Apply torque to add realism to the ball's motion
            # This simulates the ball's rotation affected by the ground
            if np.linalg.norm(self.rigid_body_velocity) > 0.01:
                # Create a slight rolling torque in direction of travel
                vel_dir = self.rigid_body_velocity / np.linalg.norm(self.rigid_body_velocity)
                rolling_torque = np.cross([0, 0, 1], vel_dir) * 0.01
                self.rigid_body_angular_velocity += rolling_torque
        
        # Boundary conditions to keep the ball in the grid
        margin = self.rigid_body_radius
        for i in range(2):  # For x and y dimensions
            if self.rigid_body_pos[i] < margin:
                self.rigid_body_pos[i] = margin
                self.rigid_body_velocity[i] *= -0.7  # Bounce off wall
                
                # Add spin when hitting wall
                if i == 0:  # X boundary - add Y angular velocity
                    self.rigid_body_angular_velocity[1] += self.rigid_body_velocity[0] * 0.1
                else:       # Y boundary - add X angular velocity
                    self.rigid_body_angular_velocity[0] -= self.rigid_body_velocity[1] * 0.1
                    
            elif self.rigid_body_pos[i] > self.grid_size[i] - margin:
                self.rigid_body_pos[i] = self.grid_size[i] - margin
                self.rigid_body_velocity[i] *= -0.7  # Bounce off wall
                
                # Add spin when hitting wall
                if i == 0:  # X boundary - add Y angular velocity
                    self.rigid_body_angular_velocity[1] += self.rigid_body_velocity[0] * 0.1
                else:       # Y boundary - add X angular velocity
                    self.rigid_body_angular_velocity[0] -= self.rigid_body_velocity[1] * 0.1
        
        # Apply rolling physics
        self.apply_rolling()
        
        # Limit angular velocity to prevent extreme spinning
        angular_speed = np.linalg.norm(self.rigid_body_angular_velocity)
        if angular_speed > self.max_angular_velocity:
            self.rigid_body_angular_velocity = self.rigid_body_angular_velocity * self.max_angular_velocity / angular_speed * bounce_factor
            
            # More surface friction on stiff ground, less on soft ground
            friction = 0.9 + (self.ground_stiffness * 0.1)
            self.rigid_body_velocity[0] *= (1 - friction * 0.05)
            self.rigid_body_velocity[1] *= (1 - friction * 0.05)
            
        # Add some random horizontal movement to make it more interesting
        # Heavier balls should move less randomly
        random_move_chance = 0.05 / max(1, self.ball_weight * 0.5)
        if np.random.random() < random_move_chance:
            random_force = 0.1 / max(1, self.ball_weight * 0.5)
            self.rigid_body_velocity[0] += np.random.uniform(-random_force, random_force)
            self.rigid_body_velocity[1] += np.random.uniform(-random_force, random_force)
            
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
        # Get current terrain stiffness
        current_stiffness = self.get_local_stiffness()
        
        # Update which voxels are active
        self.update_active_voxels()
        
        # Check for and resolve collisions
        self.check_rigid_body_collision()
        
        # Update positions - vary displacement effect based on stiffness
        # On soft terrain, more voxels move in more dramatic ways
        if current_stiffness < 0.3:  # Very soft terrain
            # Perform additional voxel disturbance steps for very soft terrain
            for _ in range(2):  # Extra displacement steps for soft terrain
                self.update_voxel_positions()
        else:
            # Normal voxel update for harder terrain
            self.update_voxel_positions()
        
        # Apply sinking effect based on terrain stiffness
        if self.rigid_body_pos[2] < self.rigid_body_radius + 1:
            # Calculate how much the ball should sink based on stiffness
            # On soft terrain, the ball sinks more
            sink_factor = 0.2 * (1.0 - current_stiffness) * self.ball_weight
            self.rigid_body_pos[2] -= sink_factor
            
            # Ensure ball doesn't sink too far
            min_height = self.rigid_body_radius * 0.4
            if self.rigid_body_pos[2] < min_height:
                self.rigid_body_pos[2] = min_height
        
        # Update rigid body
        self.update_rigid_body()
        
        # Add a stronger nudge to x-direction if ball is too slow
        # Make the nudge dependent on stiffness - easier to move on hard terrain
        speed_threshold = 0.4 + 0.2 * (1.0 - current_stiffness)
        nudge_strength = 0.05 + 0.05 * current_stiffness
        
        if (np.abs(self.rigid_body_velocity[0]) < speed_threshold and 
            self.rigid_body_pos[0] < self.grid_size[0] * 0.9):
            self.rigid_body_velocity[0] += nudge_strength
        
        # On very soft terrain, make horizontal movement more difficult
        if current_stiffness < 0.3:
            # Slow down horizontal movement in soft terrain
            resistance = 0.05 * (1.0 - current_stiffness)
            self.rigid_body_velocity[0] *= (1.0 - resistance)
            self.rigid_body_velocity[1] *= (1.0 - resistance)
        
        # Create more particle effects in soft terrain
        if current_stiffness < 0.5 and np.random.random() < 0.3:
            # Find voxels near the ball that could be dislodged
            ball_pos_int = np.round(self.rigid_body_pos).astype(int)
            check_radius = int(self.rigid_body_radius * 1.2)
            
            for dx in range(-check_radius, check_radius + 1):
                for dy in range(-check_radius, check_radius + 1):
                    x = ball_pos_int[0] + dx
                    y = ball_pos_int[1] + dy
                    z = ball_pos_int[2] - 1  # Check just below the ball
                    
                    # Skip if out of bounds
                    if (x < 0 or x >= self.grid_size[0] or
                        y < 0 or y >= self.grid_size[1] or
                        z < 0 or z >= self.grid_size[2]):
                        continue
                    
                    # Randomly displace terrain voxels based on stiffness
                    if self.terrain[x, y, z] and np.random.random() > current_stiffness:
                        # Add to active voxels with small initial velocity
                        self.terrain[x, y, z] = False
                        self.active_voxels.add((x, y, z))
                        
                        # Random small velocity
                        random_vel = np.random.uniform(-0.2, 0.2, 3)
                        random_vel[2] = abs(random_vel[2]) * 0.5  # Small upward bias
                        self.voxel_velocities[(x, y, z)] = random_vel
        
        # Update frame count
        self.frame_count += 1
        
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

# Initialize the simulation with configurable parameters
sim = VoxelTerrainSimulation(
    grid_size=(100, 50, 20),  # Wider to show all zones
    ball_weight=2.0          # Ball weight
)

# Set up the figure and 3D axis for animation
fig = plt.figure(figsize=(14, 10))
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
rigid_body = ax.plot_surface(x, y, z, color='blue', alpha=0.5)
# Initialize terrain surface (will be created in the first update)
terrain_surface = None

# Create zone markers to visualize different stiffness zones
zone_width = sim.grid_size[0] // 4
zone_colors = ['lightgreen', 'yellowgreen', 'orange', 'brown']
zone_labels = ['Very Soft', 'Medium Soft', 'Medium Hard', 'Very Hard']

for i in range(4):
    start_x = i * zone_width
    end_x = (i + 1) * zone_width
    mid_x = (start_x + end_x) // 2
    mid_y = sim.grid_size[1] // 2
    
    # Add zone marker on the ground
    ax.text(mid_x, mid_y, 0, zone_labels[i], 
            ha='center', va='center', fontsize=10, color='black',
            bbox=dict(facecolor=zone_colors[i], alpha=0.5, boxstyle="round,pad=0.3"))
    
    # Add vertical lines to separate zones
    if i > 0:
        ax.plot([start_x, start_x], [0, sim.grid_size[1]], [0, 0], 'k--', alpha=0.5)

# Set labels and limits
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(0, sim.grid_size[0])
ax.set_ylim(0, sim.grid_size[1])
ax.set_zlim(0, sim.grid_size[2])
ax.set_title('Voxel Terrain Simulation with Different Ground Stiffness Zones\nUse mouse to rotate view', fontsize=12)

# Add a text annotation explaining controls
fig.text(0.5, 0.01, 'Click and drag to rotate | Scroll to zoom | Watch how ball behaves differently in each zone', 
         ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.7))

# Status text for current zone and properties
zone_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=10,
                     bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.3"))

# Sample rate for terrain visualization (for performance)
sample_rate = 2

# Function to sample terrain points efficiently
# Replace this function entirely
def get_terrain_surface():
    # Create a heightmap for the terrain
    heightmap = np.zeros((sim.grid_size[0], sim.grid_size[1]))
    
    # Find the highest point at each x,y position
    for x in range(sim.grid_size[0]):
        for y in range(sim.grid_size[1]):
            for z in range(sim.grid_size[2]-1, -1, -1):
                if sim.terrain[x, y, z]:
                    heightmap[x, y] = z
                    break
    
    # Subsample the heightmap for better performance
    subsample = sample_rate
    X = np.arange(0, sim.grid_size[0], subsample)
    Y = np.arange(0, sim.grid_size[1], subsample)
    X_grid, Y_grid = np.meshgrid(X, Y)
    
    # Subsample heightmap and colormap
    Z_grid = heightmap[::subsample, ::subsample].T  # Note: transpose needed for correct orientation
    from scipy.ndimage import gaussian_filter
    if Z_grid.size > 0:
        Z_grid = gaussian_filter(Z_grid, sigma=0.5)
    
    # Create color map for the surface
    colors = np.zeros((len(Y), len(X), 3))
    for i, y in enumerate(Y):
        for j, x in enumerate(X):
            # Get stiffness-based color
            if x < sim.grid_size[0] and y < sim.grid_size[1]:
                colors[i, j] = sim.color_map[int(x), int(y)]
    
    return X_grid, Y_grid, Z_grid, colors

# Animation update function
def update(frame):
    global rigid_body, terrain_scatter, zone_text, terrain_surface
    
    # Simulate one step
    sim.simulate_step()
    
    # Remove old terrain surface
    if 'terrain_surface' in globals() and terrain_surface:
        terrain_surface.remove()
    
    # Get terrain surface data
    X_grid, Y_grid, Z_grid, colors = get_terrain_surface()
    
    # Plot terrain as a surface
    terrain_surface = ax.plot_surface(X_grid, Y_grid, Z_grid, 
                                 facecolors=colors, 
                                 rstride=1, cstride=1,
                                 linewidth=0, 
                                 antialiased=True,
                                 alpha=0.9,
                                 shade=True)  # Enable shading for more realistic look
    
    # Keep the active voxels as scatter for particles
    active_points = np.array(list(sim.active_voxels)) if sim.active_voxels else np.empty((0, 3))
    if len(active_points) > 0:
        active_scatter._offsets3d = (active_points[:, 0], active_points[:, 1], active_points[:, 2])
    else:
        active_scatter._offsets3d = ([], [], [])
    
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
    
    # Get ball's current rotation matrix
    rot_matrix = sim.rigid_body_orientation
    
    # Create a sphere
    sphere_x = np.sin(v) * np.cos(u)
    sphere_y = np.sin(v) * np.sin(u)
    sphere_z = np.cos(v)
    
    # Create coordinates for visualization grid on the ball
    grid_step = 4
    grid_mask = ((u % grid_step < 0.3) | (v % grid_step < 0.3))
    

    # Create new surface for rigid body without complex rotation
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = sim.rigid_body_pos[0] + sim.rigid_body_radius * np.cos(u) * np.sin(v)
    y = sim.rigid_body_pos[1] + sim.rigid_body_radius * np.sin(u) * np.sin(v)
    z = sim.rigid_body_pos[2] + sim.rigid_body_radius * np.cos(v)

    # Create a texture with grid lines for the ball
    grid_step = 4
    grid_mask = ((u % grid_step < 0.3) | (v % grid_step < 0.3))
    colors = np.ones(shape=grid_mask.shape + (3,)) * np.array([0.3, 0.3, 0.8])
    colors[grid_mask] = np.array([0.1, 0.1, 0.5])
    

    

    
    rigid_body = ax.plot_surface(x, y, z, facecolors=colors, alpha=0.7)
    
    # Update zone info text
    current_zone = sim.get_zone_name(sim.rigid_body_pos[0])
    current_stiffness = sim.get_local_stiffness()
    velocity = np.linalg.norm(sim.rigid_body_velocity)
    angular_velocity = np.linalg.norm(sim.rigid_body_angular_velocity)
    
    zone_text.set_text(f"Zone: {current_zone}\nStiffness: {current_stiffness:.2f}\n"
                      f"Speed: {velocity:.2f}\nSpin: {angular_velocity:.2f}")
    
    # Update title with info
    ax.set_title(f'Ball Rolling Across Different Ground Stiffness Zones\n'
                 f'Frame: {frame}, Active Voxels: {len(sim.active_voxels)}', fontsize=12)
    
    return terrain_surface, active_scatter, rigid_body, zone_text

# Create animation
ani = FuncAnimation(fig, update, frames=600, interval=50, blit=False)

# Enable interactive mode
plt.ion()
plt.tight_layout()
plt.show()

# This keeps the plot window open
try:
    plt.pause(600)  # Keep animation running for 10 minutes
except KeyboardInterrupt:
    print("Animation stopped by user")