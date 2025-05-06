import numpy as np
import magpylib as magpy
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple

class HalbachRing:
    def __init__(self, dimensions, polarizations, radius, num_magnets, start_angle, end_angle):
        """Initialize a half-ring of magnets
        
        Args:
            dimensions: List of magnet dimensions (dx, dy, dz) in meters
            polarizations: List of magnet polarizations (px, py, pz) in Tesla
            radius: List of ring radii in meters or single radius value
            num_magnets: List of number of magnets in each ring or single number
            start_angle: List of starting angles in degrees or single angle
            end_angle: List of ending angles in degrees or single angle
        """
        # Convert single values to lists
        self.radius = [radius] if isinstance(radius, (int, float)) else radius
        self.num_magnets = [num_magnets] if isinstance(num_magnets, int) else num_magnets
        self.start_angle = [start_angle] if isinstance(start_angle, (int, float)) else start_angle
        self.end_angle = [end_angle] if isinstance(end_angle, (int, float)) else end_angle
        
        self.dimensions = dimensions
        self.polarizations = polarizations
        
        # Calculate angles for each ring
        self.angles = []
        for start, end, n in zip(self.start_angle, self.end_angle, self.num_magnets):
            ring_angles = np.linspace(start, end, n)
            self.angles.append(ring_angles)
        
        # Create magnets
        self._update_magnets()
            
    def _update_magnets(self):
        """Update magnets with current parameters"""
        self.magnets = []
        for i, (r, n_magnets) in enumerate(zip(self.radius, self.num_magnets)):
            for j in range(n_magnets):
                # Calculate the index for dimensions and polarizations
                idx = i * n_magnets + j
                # Convert dimension to numpy array and ensure it's 1D
                dimension = np.array(self.dimensions[idx]).flatten()
                polarization = np.array(self.polarizations[idx]).flatten()
                
                cube = magpy.magnet.Cuboid(
                    dimension=dimension,
                    polarization=polarization,
                    position=(r, 0, 0)
                )
                cube.rotate_from_angax(self.angles[i][j], 'z', anchor=0)
                self.magnets.append(cube)
            
    def get_collection(self):
        """Return a magpylib Collection of all magnets"""
        return magpy.Collection(self.magnets, override_parent=True)
    
    def get_field_at_point(self, point):
        """Calculate magnetic field at a specific point"""
        collection = self.get_collection()
        return collection.getB(point)
    
    def get_field_amplitude_at_point(self, point):
        """Calculate magnetic field amplitude at a specific point"""
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

# Example usage:
if __name__ == "__main__":
    dimensions = [(0.035, 0.035, 0.035) for _ in range(78)]
    polarizations = [(-1.6, 0, 0) for _ in range(42)] + [(1.6, 0, 0) for _ in range(36)]
    
    ring = HalbachRing(
        dimensions=dimensions,
        polarizations=polarizations,
        radius=[0.12, 0.155, 0.19, 0.12, 0.155, 0.19],
        num_magnets=[15, 15, 15, 13, 13, 13],
        start_angle=[0, 0, 0, 194, 194, 194],
        end_angle=[180, 180, 180, 346, 346, 346]
    )
    
    point = np.array([0, 0, 0])
    B_amplitude = ring.get_field_amplitude_at_point(point)
    print(f"Magnetic field amplitude at (0,0): {B_amplitude:.6f} T")
    
    ring.visualize()