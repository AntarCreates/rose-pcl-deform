import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
from matplotlib.widgets import Slider, Button

# Use TkAgg backend for interactive rotation
mpl.use('TkAgg')

class VoxelTerrainSimulation:
    def __init__(self, grid_size=(100, 50, 20), ball_weight=1.0, zone_stiffness=None):
        self.grid_size = grid_size
        self.terrain = np.zeros(grid_size, dtype=bool)
        self.stiffness_map = np.zeros((grid_size[0], grid_size[1]))
        self.color_map = np.zeros((grid_size[0], grid_size[1], 3))
        
        # Use default values if zone_stiffness is not provided
        if zone_stiffness is None:
            zone_stiffness = [0.1, 0.4, 0.7, 0.95]
        
        # Setup terrain with stiffness zones
        self.setup_terrain(zone_stiffness)
        
        # Ball physical properties
        self.ball_weight = ball_weight
        self.ball_radius = 4.0
        self.ball_pos = np.array([5.0, grid_size[1]//2, grid_size[2]*0.5])
        self.ball_velocity = np.array([1.5, 0.1, 0.0])  # Reduced speed
        self.ball_angular_velocity = np.array([0.0, 0.0, 0.0])
        self.ball_orientation = np.eye(3)
        
        # Physics parameters
        self.gravity = np.array([0.0, 0.0, -0.15 * self.ball_weight])
        self.restitution = 0.3
        self.friction = 0.2
        self.spring_constant_base = 0.5
        
        # Tracking for terrain deformation
        self.active_voxels = set()
        self.voxel_velocities = {}
        self.deformation_trail = set()
    
    def setup_terrain(self, zone_stiffness):
        """Setup terrain with stiffness zones"""
        zone_width = self.grid_size[0] // 4
        
        # Set stiffness for each zone
        for x in range(self.grid_size[0]):
            zone = min(3, x // zone_width)  # Determine which zone (0-3)
            stiffness = zone_stiffness[zone]
            
            for y in range(self.grid_size[1]):
                self.stiffness_map[x, y] = stiffness
                
                # Color based on stiffness (brown gradient)
                self.color_map[x, y] = [
                    0.8 - stiffness * 0.4,  # R
                    0.6 - stiffness * 0.4,  # G
                    0.4 - stiffness * 0.3   # B
                ]
        
        # Create terrain height
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                stiffness = self.stiffness_map[x, y]
                base_height = int(self.grid_size[2] * 0.2)
                noise = int(self.grid_size[2] * 0.01 * (1 - stiffness) * np.random.randn())
                height = max(1, min(int(self.grid_size[2] * 0.3), base_height + noise))
                
                for z in range(height):
                    self.terrain[x, y, z] = True
    
    def update_stiffness(self, zone_stiffness):
        """Update stiffness values for each zone"""
        zone_width = self.grid_size[0] // 4
        
        for x in range(self.grid_size[0]):
            zone = min(3, x // zone_width)
            stiffness = zone_stiffness[zone]
            
            for y in range(self.grid_size[1]):
                self.stiffness_map[x, y] = stiffness
                
                # Update color map
                self.color_map[x, y] = [
                    0.8 - stiffness * 0.4,
                    0.6 - stiffness * 0.4,
                    0.4 - stiffness * 0.3
                ]
    
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
        
        for voxel_pos, normal, dist in contacts:
            # Calculate spring force based on Hooke's Law: F = -kx
            penetration = self.ball_radius - dist
            spring_force_magnitude = spring_constant * penetration
            spring_force = normal * spring_force_magnitude
            
            # Apply force to ball
            total_force += spring_force
            
            # Calculate torque
            r = (voxel_pos - self.ball_pos) + normal * 0.5
            torque = np.cross(r, spring_force)
            total_torque += torque
            
            # Apply force to terrain - deform based on stiffness
            voxel_tuple = tuple(voxel_pos.astype(int))
            
            # More accurate deformation calculation - make it more sensitive to stiffness differences
            # Deformation is inversely proportional to stiffness (softer = more deformation)
            deformation_factor = (1.0 - current_stiffness**0.7) * penetration * 1.2
            
            if deformation_factor > 0.1:
                # Remove original voxel
                self.terrain[voxel_tuple] = False
                
                # Calculate compressed position
                compression_vector = normal * deformation_factor
                new_pos = np.array(voxel_pos) - compression_vector
                new_pos_int = tuple(np.round(new_pos).astype(int))
                
                # Create new voxel at compressed position if valid
                if (0 <= new_pos_int[0] < self.grid_size[0] and
                    0 <= new_pos_int[1] < self.grid_size[1] and
                    0 <= new_pos_int[2] < self.grid_size[2] and
                    not self.terrain[new_pos_int]):
                    
                    self.terrain[new_pos_int] = True
                    self.deformation_trail.add(new_pos_int)
                    self.active_voxels.add(new_pos_int)
                    
                    # Detach some voxels in soft terrain
                    if current_stiffness < 0.3 and np.random.random() < 0.2:
                        voxel_force = -spring_force
                        velocity_scale = 0.3 * (1 - current_stiffness)
                        self.voxel_velocities[new_pos_int] = voxel_force * velocity_scale
        
        # Apply accumulated forces to ball
        acceleration = total_force / self.ball_weight
        self.ball_velocity += acceleration
        
        # Apply torque to ball
        moment_of_inertia = 0.4 * self.ball_weight * self.ball_radius**2
        angular_acceleration = total_torque / moment_of_inertia
        self.ball_angular_velocity += angular_acceleration
    
    def nudge_ball(self, strength=1.0):
        """Apply a forward nudge to the ball with given strength"""
        # Apply a nudge in the x direction (forward)
        self.ball_velocity[0] += 0.5 * strength
        # Add a small upward component to help unstick the ball
        self.ball_velocity[2] += 0.2 * strength
    
    def apply_friction(self):
        """Apply friction to create proper rolling"""
        terrain_height = self.find_terrain_height_below_ball()
        if self.ball_pos[2] - terrain_height > self.ball_radius + 0.5:
            return
        
        current_stiffness = self.get_local_stiffness()
        friction_coef = self.friction * (0.5 + current_stiffness * 0.5)
        normal_force = self.ball_weight * 9.8
        friction_force_mag = friction_coef * normal_force
        
        # Get horizontal velocity components
        horizontal_vel = self.ball_velocity.copy()
        horizontal_vel[2] = 0
        horizontal_speed = np.linalg.norm(horizontal_vel)
        
        if horizontal_speed > 0.01:
            # Direction of friction force (opposite to motion)
            friction_dir = -horizontal_vel / horizontal_speed
            
            # Apply reduced friction to allow continued movement
            friction_force = friction_dir * min(friction_force_mag, horizontal_speed * 0.1)
            self.ball_velocity += friction_force
            
            # Calculate torque for proper rolling
            rolling_axis = np.cross(np.array([0, 0, 1]), horizontal_vel)
            if np.linalg.norm(rolling_axis) > 0:
                rolling_axis = rolling_axis / np.linalg.norm(rolling_axis)
                target_angular_vel = rolling_axis * (horizontal_speed / self.ball_radius)
                
                # Apply torque for better rolling
                torque_coefficient = 0.4
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
            
            # Move if position valid
            if (0 <= new_pos_int[0] < self.grid_size[0] and
                0 <= new_pos_int[1] < self.grid_size[1] and
                0 <= new_pos_int[2] < self.grid_size[2] and
                new_pos_int != voxel and
                not self.terrain[new_pos_int]):
                
                self.terrain[new_pos_int] = True
                if voxel in self.active_voxels:
                    self.active_voxels.remove(voxel)
                self.active_voxels.add(new_pos_int)
                
                # Apply damping
                self.voxel_velocities[new_pos_int] = velocity * 0.9
                voxels_to_remove.append(voxel)
            
            # Remove if stopped
            if velocity.dot(velocity) < 0.01:
                voxels_to_remove.append(voxel)
        
        # Clean up
        for voxel in voxels_to_remove:
            if voxel in self.voxel_velocities:
                del self.voxel_velocities[voxel]
    
    def update_ball_rotation(self):
        """Update ball's orientation based on angular velocity"""
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
            
            # Apply rotation
            self.ball_orientation = np.dot(rotation_matrix, self.ball_orientation)
            
            # Apply rolling resistance
            current_stiffness = self.get_local_stiffness()
            resistance = 0.01 * (1 + current_stiffness)
            self.ball_angular_velocity *= (1.0 - resistance)
    
    def update(self):
        """Update simulation state for one time step"""
        # Find contact points
        contacts = self.find_contact_points()
        
        # Apply physics
        self.apply_spring_forces(contacts)
        self.apply_friction()
        
        # Update ball position
        self.ball_pos += self.ball_velocity
        
        # Apply reduced gravity to prevent bouncing
        self.ball_velocity += self.gravity * 0.7
        
        self.update_ball_rotation()
        self.update_active_voxels()
        
        # Boundary conditions
        for i in range(2):
            if self.ball_pos[i] < self.ball_radius:
                self.ball_pos[i] = self.ball_radius
                self.ball_velocity[i] *= -0.7
            elif self.ball_pos[i] > self.grid_size[i] - self.ball_radius:
                self.ball_pos[i] = self.grid_size[i] - self.ball_radius
                self.ball_velocity[i] *= -0.7
        
        # Prevent excessive bouncing
        if self.ball_pos[2] < self.ball_radius * 0.7:
            self.ball_pos[2] = self.ball_radius * 0.7
            if self.ball_velocity[2] < 0:
                self.ball_velocity[2] = -self.ball_velocity[2] * 0.2
            self.ball_velocity[2] *= 0.9
        
        # Small nudge to keep rolling
        if np.linalg.norm(self.ball_velocity[:2]) < 0.3:
            self.ball_velocity[0] += 0.02
            
        # Additional damping to vertical velocity
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

    def reset_ball(self):
        """Reset the ball to starting position and refresh terrain"""
        self.ball_pos = np.array([5.0, self.grid_size[1]//2, self.grid_size[2]*0.5])
        self.ball_velocity = np.array([1.5, 0.1, 0.0])
        self.ball_angular_velocity = np.array([0.0, 0.0, 0.0])
        self.ball_orientation = np.eye(3)
        
        # Clear tracking sets
        self.deformation_trail.clear()
        self.active_voxels.clear()
        self.voxel_velocities.clear()
        
        # Refresh terrain - create a fresh terrain with current stiffness settings
        self.terrain = np.zeros(self.grid_size, dtype=bool)
        
        # Recreate terrain height based on current stiffness
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                stiffness = self.stiffness_map[x, y]
                base_height = int(self.grid_size[2] * 0.2)
                noise = int(self.grid_size[2] * 0.01 * (1 - stiffness) * np.random.randn())
                height = max(1, min(int(self.grid_size[2] * 0.3), base_height + noise))
                
                for z in range(height):
                    self.terrain[x, y, z] = True

def visualize_simulation(sim):
    """Visualize the simulation with interactive sliders"""
    # Set up the figure and 3D axis
    fig = plt.figure(figsize=(12, 12))
    
    # Main 3D plot area (leave room for sliders and buttons at bottom)
    ax = fig.add_subplot(111, projection='3d', position=[0.1, 0.3, 0.8, 0.65])
    ax.set_axis_off()
    ax.view_init(elev=30, azim=45)
    
    # Create slider axes
    ax_zone1 = plt.axes([0.25, 0.20, 0.5, 0.02])
    ax_zone2 = plt.axes([0.25, 0.17, 0.5, 0.02])
    ax_zone3 = plt.axes([0.25, 0.14, 0.5, 0.02])
    ax_zone4 = plt.axes([0.25, 0.11, 0.5, 0.02])
    ax_ball_weight = plt.axes([0.25, 0.08, 0.5, 0.02])
    ax_speed = plt.axes([0.25, 0.05, 0.5, 0.02])
    
    # Get current stiffness values
    zone_width = sim.grid_size[0] // 4
    zone1_val = sim.stiffness_map[zone_width//2, sim.grid_size[1]//2]
    zone2_val = sim.stiffness_map[zone_width + zone_width//2, sim.grid_size[1]//2]
    zone3_val = sim.stiffness_map[2*zone_width + zone_width//2, sim.grid_size[1]//2]
    zone4_val = sim.stiffness_map[3*zone_width + zone_width//2, sim.grid_size[1]//2]
    
    # Create sliders
    s_zone1 = Slider(ax_zone1, 'Zone 1 Stiffness', 0.01, 1.0, valinit=zone1_val)
    s_zone2 = Slider(ax_zone2, 'Zone 2 Stiffness', 0.01, 1.0, valinit=zone2_val)
    s_zone3 = Slider(ax_zone3, 'Zone 3 Stiffness', 0.01, 1.0, valinit=zone3_val)
    s_zone4 = Slider(ax_zone4, 'Zone 4 Stiffness', 0.01, 1.0, valinit=zone4_val)
    s_ball_weight = Slider(ax_ball_weight, 'Ball Weight', 0.5, 5.0, valinit=sim.ball_weight)
    s_speed = Slider(ax_speed, 'Ball Speed', 0.1, 5.0, valinit=np.linalg.norm(sim.ball_velocity[:2]))
    
    # Add buttons
    reset_ax = plt.axes([0.8, 0.05, 0.1, 0.04])
    reset_button = Button(reset_ax, 'Reset', color='lightgoldenrodyellow', hovercolor='0.975')
    
    # Add nudge button
    nudge_ax = plt.axes([0.8, 0.11, 0.1, 0.04])
    nudge_button = Button(nudge_ax, 'Nudge', color='lightblue', hovercolor='0.8')
    
    # Create zone markers for visualization
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
    
    # Status text for current zone and properties
    zone_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=10,
                         bbox=dict(facecolor='white', alpha=0.7))
    
    # Sample rate for terrain visualization
    sample_rate = 2
    
    # Flag to prevent animation updates during slider changes
    slider_active = False
    paused = False
    
    # Create terrain surface visualization
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
    
    # Initialize empty visualization elements
    terrain_surface = None
    active_scatter = ax.scatter([], [], [], c='red', marker='s', s=25)
    deformation_scatter = ax.scatter([], [], [], c='darkred', marker='s', s=15)
    rigid_body = None
    
    # Update function for sliders
    def update_params(val):
        nonlocal slider_active
        
        # Set the flag to prevent animation updates while slider is being changed
        slider_active = True
        
        # Store current position and velocities
        current_pos = sim.ball_pos.copy()
        current_vel = sim.ball_velocity.copy()
        
        # Get new values from sliders
        new_stiffness = [s_zone1.val, s_zone2.val, s_zone3.val, s_zone4.val]
        sim.update_stiffness(new_stiffness)
        
        # Update ball weight
        sim.ball_weight = s_ball_weight.val
        sim.gravity = np.array([0.0, 0.0, -0.15 * sim.ball_weight])
        
        # Update ball speed (normalize direction, apply new magnitude)
        if np.linalg.norm(current_vel[:2]) > 0.01:  # Avoid division by zero
            direction = current_vel[:2] / np.linalg.norm(current_vel[:2])
            new_speed = s_speed.val
            # Only update horizontal velocity components
            sim.ball_velocity = current_vel.copy()
            sim.ball_velocity[:2] = direction * new_speed
        
        # Restore position to prevent "nudging" when sliders change
        sim.ball_pos = current_pos
        
        # Reset the flag
        slider_active = False
    
    # Connect sliders to update function
    s_zone1.on_changed(update_params)
    s_zone2.on_changed(update_params)
    s_zone3.on_changed(update_params)
    s_zone4.on_changed(update_params)
    s_ball_weight.on_changed(update_params)
    s_speed.on_changed(update_params)
    
    # Reset button function
    def reset(event):
        nonlocal slider_active
        slider_active = True
        
        # Store current values that we want to preserve
        current_weight = sim.ball_weight
        current_speed = np.linalg.norm(sim.ball_velocity[:2])
        current_stiffness = [s_zone1.val, s_zone2.val, s_zone3.val, s_zone4.val]
        
        # Reset the ball and terrain
        sim.reset_ball()
        
        # Make sure stiffness values are preserved
        sim.update_stiffness(current_stiffness)
        
        # Restore weight and speed rather than resetting to defaults
        sim.ball_weight = current_weight
        
        # Normalize and apply the preserved speed
        if np.linalg.norm(sim.ball_velocity[:2]) > 0.01:
            direction = sim.ball_velocity[:2] / np.linalg.norm(sim.ball_velocity[:2])
            sim.ball_velocity[:2] = direction * current_speed
            
        slider_active = False
    
    reset_button.on_clicked(reset)
    
    # Nudge button function
    def nudge(event):
        # Apply a forward nudge to help when the ball gets stuck
        sim.nudge_ball(strength=1.5)
    
    nudge_button.on_clicked(nudge)
    
    # Animation update function
    def update(frame):
        nonlocal terrain_surface, rigid_body
        
        # Skip updates if sliders are being dragged to prevent position changes
        if slider_active:
            return terrain_surface, active_scatter, deformation_scatter, rigid_body, zone_text
        
        
        # Simulate multiple steps per frame for smoother rolling
        for _ in range(2):
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
        
        # Update active voxels scatter - particles flying around
        active_points = np.array(list(sim.active_voxels)) if sim.active_voxels else np.empty((0, 3))
        if len(active_points) > 0:
            active_scatter._offsets3d = (active_points[:, 0], active_points[:, 1], active_points[:, 2])
        else:
            active_scatter._offsets3d = ([], [], [])
            
        # Visualize deformation trail
        deformation_points = np.array(list(sim.deformation_trail)) if sim.deformation_trail else np.empty((0, 3))
        if len(deformation_points) > 0:
            deformation_scatter._offsets3d = (deformation_points[:, 0], deformation_points[:, 1], deformation_points[:, 2])
        else:
            deformation_scatter._offsets3d = ([], [], [])
        
        # Update ball visualization
        if rigid_body:
            rigid_body.remove()
        
        # Create sphere with grid texture for rotation visualization
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        sphere_x = np.sin(v) * np.cos(u)
        sphere_y = np.sin(v) * np.sin(u)
        sphere_z = np.cos(v)
        
        # Grid texture for visualizing rotation
        grid_step = 4
        grid_mask = ((u % grid_step < 0.3) | (v % grid_step < 0.3))
        colors = np.ones(shape=grid_mask.shape + (3,)) * np.array([0.3, 0.3, 0.8])
        colors[grid_mask] = np.array([0.1, 0.1, 0.5])
        
        # Apply position and radius
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
        ax.set_title(f'Interactive Terrain Simulation - Frame: {frame}')
        
        return terrain_surface, active_scatter, deformation_scatter, rigid_body, zone_text
            
        # Visualize deformation trail
        deformation_points = np.array(list(sim.deformation_trail)) if sim.deformation_trail else np.empty((0, 3))
        if len(deformation_points) > 0:
            deformation_scatter._offsets3d = (deformation_points[:, 0], deformation_points[:, 1], deformation_points[:, 2])
        else:
            deformation_scatter._offsets3d = ([], [], [])
        
        # Update ball visualization
        if rigid_body:
            rigid_body.remove()
        
        # Create sphere with grid texture for rotation visualization
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        sphere_x = np.sin(v) * np.cos(u)
        sphere_y = np.sin(v) * np.sin(u)
        sphere_z = np.cos(v)
        
        # Grid texture for visualizing rotation
        grid_step = 4
        grid_mask = ((u % grid_step < 0.3) | (v % grid_step < 0.3))
        colors = np.ones(shape=grid_mask.shape + (3,)) * np.array([0.3, 0.3, 0.8])
        colors[grid_mask] = np.array([0.1, 0.1, 0.5])
        
        # Apply position and radius
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
        ax.set_title(f'Interactive Terrain Simulation - Frame: {frame}')
        
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
sim = VoxelTerrainSimulation(grid_size=(100, 50, 20))
visualize_simulation(sim)