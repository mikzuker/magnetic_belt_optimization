import numpy as np
import magpylib as magpy
import matplotlib.pyplot as plt
import os


class HalbachRing:
    def __init__(
        self, 
        dimensions, 
        polarizations, 
        radius, 
        num_magnets, 
        start_angle, 
        end_angle,
        max_dimension=(0.03, 0.02, 0.01),
        max_polarization=1.4,
        max_radius=0.15,
        validate=True
    ):
        """Initialize a half-ring of magnets

        Args:
            dimensions: List of relative magnet dimensions (dx, dy, dz) from 0-1
            polarizations: List of relative magnet polarization directions (px, py, pz)
            radius: List of relative ring radii (0-1) or single value
            num_magnets: List of number of magnets in each ring or single number
            start_angle: List of starting angles in degrees or single angle
            end_angle: List of ending angles in degrees or single angle
            max_dimension: Maximum absolute dimensions in meters to scale relative values
            max_polarization: Maximum polarization strength in Tesla
            max_radius: Maximum radius in meters to scale relative values
            validate: Whether to validate the configuration during initialization
        """
        # Store scaling factors
        self.max_dimension = max_dimension
        self.max_polarization = max_polarization
        self.max_radius = max_radius
        
        # Convert single values to lists
        self.radius = [radius] if isinstance(radius, (int, float)) else radius
        self.num_magnets = (
            [num_magnets] if isinstance(num_magnets, int) else num_magnets
        )
        self.start_angle = (
            [start_angle] if isinstance(start_angle, (int, float)) else start_angle
        )
        self.end_angle = (
            [end_angle] if isinstance(end_angle, (int, float)) else end_angle
        )

        # Convert relative dimensions to absolute
        self.dimensions = self._scale_dimensions(dimensions)
        self.polarizations = self._scale_polarizations(polarizations)
        self.absolute_radius = self._scale_radius(self.radius)

        # Calculate angles for each ring
        self.angles = []
        for start, end, n in zip(self.start_angle, self.end_angle, self.num_magnets):
            ring_angles = np.linspace(start, end, n)
            self.angles.append(ring_angles)

        # Create magnets
        self._update_magnets()
        
        # Validate configuration if requested
        if validate and self.check_intersections():
            raise ValueError("Invalid configuration: magnets intersect with each other")

    def _scale_dimensions(self, dimensions):
        """Scale relative dimensions (0-1) to absolute dimensions"""
        scaled_dimensions = []
        for dim in dimensions:
            scaled_dim = [
                rel_d * max_d for rel_d, max_d in zip(dim, self.max_dimension)
            ]
            scaled_dimensions.append(scaled_dim)
        return scaled_dimensions
    
    def _scale_polarizations(self, polarizations):
        """Scale relative polarization directions to absolute Tesla values"""
        scaled_polarizations = []
        for pol in polarizations:
            # Get the direction vector
            pol_vec = np.array(pol)
            # Normalize if not a zero vector
            norm = np.linalg.norm(pol_vec)
            if norm > 0:
                pol_vec = pol_vec / norm
            # Scale to max polarization value
            scaled_pol = pol_vec * self.max_polarization
            scaled_polarizations.append(scaled_pol)
        return scaled_polarizations
    
    def _scale_radius(self, radii):
        """Scale relative radii (0-1) to absolute values in meters"""
        return [r * self.max_radius for r in radii]

    def _update_magnets(self):
        """Update magnets with current parameters"""
        self.magnets = []
        # Track the index into dimensions and polarizations lists
        idx = 0
        for i, (r, n_magnets) in enumerate(zip(self.absolute_radius, self.num_magnets)):
            for j in range(n_magnets):
                # Use the current running index
                dimension = np.array(self.dimensions[idx]).flatten()
                polarization = np.array(self.polarizations[idx]).flatten()
                
                cube = magpy.magnet.Cuboid(
                    dimension=dimension, polarization=polarization, position=(r, 0, 0)
                )
                cube.rotate_from_angax(self.angles[i][j], "z", anchor=0)
                self.magnets.append(cube)
                idx += 1  # Move to next magnet index

    def get_collection(self):
        """Return a magpylib Collection of all magnets"""
        return magpy.Collection(self.magnets, override_parent=True)

    def get_field_at_point(self, point):
        """Calculate magnetic field at a specific point"""
        collection = self.get_collection()
        return collection.getB(point)

    def check_intersections(self):
        """Check if any magnets in the structure intersect with each other
        
        Returns:
            bool: True if intersections found, False otherwise
        """
        for i in range(len(self.magnets)):
            for j in range(i+1, len(self.magnets)):
                # Get bounding boxes in global coordinates
                mag1 = self.magnets[i]
                mag2 = self.magnets[j]
                
                # Get positions and dimensions
                pos1 = np.array(mag1.position)
                dim1 = np.array(mag1.dimension) / 2  # Half-dimensions for box check
                
                pos2 = np.array(mag2.position)
                dim2 = np.array(mag2.dimension) / 2
                
                # Transform dimensions by rotation
                # This is a simplified check that works for our case
                # A more accurate check would use the full transformation matrices
                rot1 = mag1.orientation.as_matrix()
                rot2 = mag2.orientation.as_matrix()
                
                # Create oriented bounding box corners
                corners1 = []
                corners2 = []
                
                for dx in [-1, 1]:
                    for dy in [-1, 1]:
                        for dz in [-1, 1]:
                            # Get corner in local coordinates
                            corner_local = np.array([dx * dim1[0], dy * dim1[1], dz * dim1[2]])
                            # Transform to global
                            corner_global = pos1 + rot1 @ corner_local
                            corners1.append(corner_global)
                            
                            # Same for second magnet
                            corner_local = np.array([dx * dim2[0], dy * dim2[1], dz * dim2[2]])
                            corner_global = pos2 + rot2 @ corner_local
                            corners2.append(corner_global)
                
                # Simplified collision check by comparing extremes
                corners1 = np.array(corners1)
                corners2 = np.array(corners2)
                
                min1 = np.min(corners1, axis=0)
                max1 = np.max(corners1, axis=0)
                min2 = np.min(corners2, axis=0)
                max2 = np.max(corners2, axis=0)
                
                # Check for overlap along all axes
                if (min1[0] <= max2[0] and max1[0] >= min2[0] and
                    min1[1] <= max2[1] and max1[1] >= min2[1] and
                    min1[2] <= max2[2] and max1[2] >= min2[2]):
                    return True
                
        return False

    def get_field_amplitude_at_point(self, point):
        """Calculate magnetic field amplitude at a specific point
        
        Returns 0 if the configuration is physically invalid (magnets intersect)
        """
        # Check for intersections first
        if self.check_intersections():
            return 0.0  # Return zero for invalid configurations
            
        # Calculate field as normal for valid configurations
        B = self.get_field_at_point(point)
        return np.linalg.norm(B)

    def visualize(self):
        fig, ax = plt.subplots()

        x = np.linspace(-0.2, 0.2, 100)
        y = np.linspace(-0.2, 0.2, 100)
        X, Y = np.meshgrid(x, y)
        points = np.array([X.flatten(), Y.flatten(), np.zeros_like(X.flatten())]).T

        B = np.array([self.get_field_at_point(point) for point in points])
        Bx = B[:, 0].reshape(X.shape)
        By = B[:, 1].reshape(X.shape)
        Bamp = np.linalg.norm(B, axis=1).reshape(X.shape)

        pc = ax.contourf(X, Y, Bamp, levels=50, cmap="coolwarm")
        ax.streamplot(X, Y, Bx, By, color="k", density=1.5, linewidth=1)

        fig.colorbar(pc, ax=ax, label="|B| [T]")
        ax.set(xlabel="x-position [m]", ylabel="y-position [m]", aspect=1)
        plt.xlim(-0.2, 0.2)
        plt.ylim(-0.2, 0.2)
        plt.show()

    def visualize_combined(self, show=True, save_path=None):
        """Create a combined visualization with 3D views and magnetic field plots
        
        Args:
            show: Boolean indicating whether to display the plot interactively
            save_path: File path to save the figure (None to skip saving)
        """
        original_backend = plt.get_backend()
        
        if not show:
            plt.switch_backend('Agg')
        
        # Create directory if save_path is provided
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Get the collection of magnets
        collection = self.get_collection()
        
        # Create a single figure with three subplots
        fig = plt.figure(figsize=(18, 6))
        
        # First subplot: 3D halbach visualization (default view)
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        magpy.show(collection, backend="matplotlib", canvas=ax1, 
                   style_legend_show=False, 
                   style_magnetization_show=True)
        ax1.set_title("Default View")
        # Remove axis labels and ticks
        ax1.set_xlabel('')
        ax1.set_ylabel('')
        ax1.set_zlabel('')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_zticks([])
        # Remove legend if it exists
        if ax1.get_legend():
            ax1.get_legend().remove()
        
        # Second subplot: 3D halbach visualization (top view)
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        magpy.show(collection, backend="matplotlib", canvas=ax2, 
                   style_legend_show=False,
                   style_magnetization_show=True)
        ax2.view_init(elev=90, azim=0)
        ax2.set_title("Top View")
        # Remove axis labels and ticks
        ax2.set_xlabel('')
        ax2.set_ylabel('')
        ax2.set_zlabel('')
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_zticks([])
        # Remove legend if it exists
        if ax2.get_legend():
            ax2.get_legend().remove()
        
        # Third subplot: Magnetic field plot
        ax3 = fig.add_subplot(1, 3, 3)
        
        grid = np.mgrid[-0.2:0.2:100j, -0.2:0.2:100j, 0:0:1j].T[0]
        X, Y, _ = np.moveaxis(grid, 2, 0)
        
        B = collection.getB(grid)
        Bx, By, _ = np.moveaxis(B, 2, 0)
        Bamp = np.linalg.norm(B, axis=2)
        
        pc = ax3.contourf(X, Y, Bamp, levels=50, cmap="coolwarm")
        ax3.streamplot(X, Y, Bx, By, color="k", density=1.5, linewidth=1)
        fig.colorbar(pc, ax=ax3, label="|B| [T]")
        
        ax3.set(
            xlabel="",
            ylabel="",
            aspect=1,
            title="Magnetic Field"
        )
        
        ax3.set_xlim(-0.2, 0.2)
        ax3.set_ylim(-0.2, 0.2)
        # Remove ticks
        # ax3.set_xticks([])
        # ax3.set_yticks([])
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        
        # Show or close
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        # Restore original backend
        if not show:
            plt.switch_backend(original_backend)


def create_double_ring_halbach_ring():
    # Create a double-ring Halbach array with relative values
    # Inner ring with 8 magnets, outer ring with 12 magnets
    
    # Relative dimensions (will be scaled by max_dimension)
    dimensions = [(0.83, 0.75, 1.0) for _ in range(8)] + [(1.0, 1.0, 1.0) for _ in range(12)]

    # Alternating polarization pattern for Halbach effect (directions only)
    polarizations = []
    for i in range(8):
        angle = i * (360/8)
        px = np.cos(np.radians(angle))
        py = np.sin(np.radians(angle))
        polarizations.append((px, py, 0))

    for i in range(12):
        angle = i * (360/12) + 15  # 15 degree offset from inner ring
        px = -np.cos(np.radians(angle))
        py = -np.sin(np.radians(angle)) 
        polarizations.append((px, py, 0))

    # Calculate correct end angles to avoid intersections
    # Formula: end_angle = 360 * (n-1)/n where n is the number of magnets
    inner_end_angle = 360 * (8-1)/8  # = 315°
    outer_end_angle = 360 * (12-1)/12  # = 330°

    # Create the Halbach ring with relative values
    ring = HalbachRing(
        dimensions=dimensions,
        polarizations=polarizations,
        radius=[0.67, 1.0],            # Relative radii: inner=0.67, outer=1.0
        num_magnets=[8, 12],           # 8 magnets in inner ring, 12 in outer
        start_angle=[0, 0],            # Both rings start at 0 degrees
        end_angle=[inner_end_angle, outer_end_angle],  # Use correct end angles
        max_dimension=(0.03, 0.02, 0.01),  # Maximum dimensions in meters
        max_polarization=1.4,              # Maximum polarization in Tesla
        max_radius=0.15,                   # Maximum radius in meters
        validate=True                      # Validate configuration
    )
    return ring


def generate_random_halbach_ring(seed=None):
    """Generate a random Halbach ring configuration with valid parameters
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        HalbachRing object with random parameters
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random parameters within reasonable ranges
    num_rings = np.random.randint(1, 3)  # 1-2 rings (reduced from 1-3)
    
    # Use fewer magnets to reduce intersection probability
    num_magnets_per_ring = [np.random.randint(6, 12) for _ in range(num_rings)]  # 6-11 magnets per ring
    
    # Generate relative radii with more spacing between rings
    if num_rings == 1:
        radii = [0.7]  # Single ring at 70% of max radius
    else:
        # Multiple rings with more spacing
        min_radii = [0.4, 0.7]  # At least 30% difference between rings
        max_radii = [0.6, 0.95]  # Keep outer ring slightly inside max radius
        radii = [np.random.uniform(min_r, max_r) for min_r, max_r in zip(min_radii[:num_rings], max_radii[:num_rings])]
    
    # Start all rings at 0 degrees
    start_angles = [0 for _ in range(num_rings)]
    
    # Calculate end angles to avoid intersections
    end_angles = [360 * (n-1)/n for n in num_magnets_per_ring]
    
    # Generate dimensions for each magnet - smaller relative sizes to reduce intersections
    total_magnets = sum(num_magnets_per_ring)
    dimensions = [(
        np.random.uniform(0.4, 0.7),  # Reduced from 0.5-1.0
        np.random.uniform(0.4, 0.7),  # Reduced from 0.5-1.0
        np.random.uniform(0.4, 0.7)   # Reduced from 0.5-1.0
    ) for _ in range(total_magnets)]
    
    # Generate polarization directions
    polarizations = []
    magnet_count = 0
    
    for ring_idx, num_magnets in enumerate(num_magnets_per_ring):
        for i in range(num_magnets):
            # Alternate polarization direction based on position
            angle = i * (360/num_magnets)
            
            # Randomly decide between radial, tangential, or mixed polarization
            pol_type = np.random.choice(['radial', 'tangential', 'mixed'])
            
            if pol_type == 'radial':
                # Radial polarization (pointing inward/outward)
                px = np.cos(np.radians(angle))
                py = np.sin(np.radians(angle))
                if ring_idx % 2 == 0:  # Alternate between rings
                    px, py = -px, -py
            elif pol_type == 'tangential':
                # Tangential polarization (perpendicular to radial)
                px = -np.sin(np.radians(angle))
                py = np.cos(np.radians(angle))
                if ring_idx % 2 == 0:  # Alternate between rings
                    px, py = -px, -py
            else:
                # Mixed polarization
                px = np.cos(np.radians(angle + np.random.uniform(0, 90)))
                py = np.sin(np.radians(angle + np.random.uniform(0, 90)))
            
            # No Z-component to keep things simpler and more likely valid
            pz = 0
            
            polarizations.append((px, py, pz))
            magnet_count += 1
    
    # Random maximum dimensions (in meters) - slightly smaller
    max_dimension = (
        np.random.uniform(0.015, 0.03),  # x: 1.5-3 cm
        np.random.uniform(0.01, 0.02),   # y: 1-2 cm
        np.random.uniform(0.005, 0.015)  # z: 0.5-1.5 cm
    )
    
    # Random maximum radius (in meters)
    max_radius = np.random.uniform(0.1, 0.18)  # 10-18 cm
    
    # Random polarization strength (in Tesla)
    max_polarization = np.random.uniform(1.0, 1.5)  # 1.0-1.5 T
    
    # Create the ring with validation enabled
    try:
        ring = HalbachRing(
            dimensions=dimensions,
            polarizations=polarizations,
            radius=radii if isinstance(radii, list) else [radii],
            num_magnets=num_magnets_per_ring,
            start_angle=start_angles,
            end_angle=end_angles,
            max_dimension=max_dimension,
            max_polarization=max_polarization,
            max_radius=max_radius,
            validate=True
        )
        return ring, True
    except ValueError as e:
        # If validation fails, return the error
        return str(e), False

def test_random_halbach_rings(num_examples=5, max_attempts=20):
    """Generate and test multiple random Halbach ring configurations
    
    Args:
        num_examples: Number of valid random examples to generate
        max_attempts: Maximum number of attempts to generate valid examples
        
    Returns:
        List of valid HalbachRing objects
    """
    valid_examples = []
    attempts = 0
    base_seed = 42
    
    while len(valid_examples) < num_examples and attempts < max_attempts:
        # Use a different seed for each attempt
        ring, is_valid = generate_random_halbach_ring(seed=base_seed + attempts)
        
        if is_valid:
            valid_examples.append(ring)
        
        attempts += 1
    
    return valid_examples

def visualize_random_examples(show=True, save_dir="figures/random_examples"):
    """Generate, test, and visualize multiple random Halbach ring configurations
    
    Args:
        show: Whether to display the visualizations interactively
        save_dir: Directory to save visualizations
    """
    # Create save directory if it doesn't exist
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Generate valid examples
    examples = test_random_halbach_rings(5)
    
    print(f"Generated {len(examples)} valid Halbach ring configurations:")
    
    for i, ring in enumerate(examples):
        # Get ring details
        num_rings = len(ring.num_magnets)
        total_magnets = sum(ring.num_magnets)
        
        # Calculate field at center
        B_amplitude = ring.get_field_amplitude_at_point([0, 0, 0])
        
        print(f"Example {i+1}:")
        print(f"  - Rings: {num_rings}")
        print(f"  - Total magnets: {total_magnets}")
        print(f"  - Field strength at center: {B_amplitude:.4f} T")
        
        # Save visualization
        if save_dir:
            save_path = f"{save_dir}/example_{i+1}.png"
            ring.visualize_combined(show=show, save_path=save_path)
            print(f"  - Visualization saved to {save_path}")
        
        print()
    
    return examples

# Example usage:
if __name__ == "__main__":
    ring = create_double_ring_halbach_ring()

    point = np.array([0, 0, 0])
    B_amplitude = ring.get_field_amplitude_at_point(point)
    print(f"Magnetic field amplitude at (0,0): {B_amplitude:.6f} T")

    # Basic visualization (interactive)
    # ring.visualize()
    
    # Combined visualization options:
    # 1. Show interactive plot
    # ring.visualize_combined(show=True)
    
    # 2. Save without displaying (useful for automated scripts)
    ring.visualize_combined(show=False, save_path="figures/halbach_combined.png")
    
    # 3. Both show and save
    # ring.visualize_combined(show=True, save_path="figures/halbach_combined.png")

    # Generate and visualize random examples
    print("\nGenerating random Halbach ring examples...")
    visualize_random_examples(show=True, save_dir="figures/random_examples")
