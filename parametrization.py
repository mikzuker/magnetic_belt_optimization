import numpy as np
import magpylib as magpy
import matplotlib.pyplot as plt
from typing import Optional

class HalbachRing:
    def __init__(self, 
                 dimensions: list[tuple[float, float, float]], 
                 polarizations: list[tuple[float, float, float]], 
                 angles: Optional[list[float]] = None, 
                 radius=0.12, 
                 num_magnets=10, 
                 start_angle=0, 
                 end_angle=180):
        """
        Initialize a half-ring of magnets
        
        Parameters:
        -----------
        radius : float
            Radius of the ring in meters
        num_magnets : int
            Number of magnets in the ring
        start_angle : float
            Starting angle in degrees
        end_angle : float
            Ending angle in degrees
        """
        self.radius = radius
        self.num_magnets = num_magnets
        self.start_angle = start_angle
        self.end_angle = end_angle
        
        if angles is None:
            angles = list(np.linspace(start_angle, end_angle, num_magnets, endpoint=True))
            
        self.magnets = []
        for dimension, polarization, angle in zip(dimensions, polarizations, angles):
            cube = magpy.magnet.Cuboid(
                dimension=dimension,
                polarization=polarization,
                position=(self.radius, 0, 0)
            )
            cube.rotate_from_angax(angle, 'z', anchor=0)
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
    dimensions = [(0.035, 0.035, 0.035) for _ in range(10)] 
    polarizations = [(-1.5, 0, 0) for _ in range(10)] 
    
    ring = HalbachRing(
        dimensions=dimensions,
        polarizations=polarizations,
        radius=0.12,
        num_magnets=10,
        start_angle=0,
        end_angle=180
    )
    
    point = np.array([0, 0, 0])
    B_amplitude = ring.get_field_amplitude_at_point(point)
    print(f"Magnetic field amplitude at (0,0): {B_amplitude:.6f} T")
    
    ring.visualize()