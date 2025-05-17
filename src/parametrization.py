import numpy as np
import magpylib as magpy
import matplotlib.pyplot as plt
from typing import Union, List, Tuple

class HalbachRing:
    def __init__(self,
                 dimensions: List[float],
                 polarizations: List[Tuple[float, float, float]],
                 min_radius: float,
                 num_rings: int,
                 angles: List[float],
                 num_magnets: Union[int, List[int]],
                 start_angle: Union[float, List[float]],
                 end_angle: Union[float, List[float]]):
        """
        Initialize a ring of magnets.

        Args:
            dimensions: List of scalar size factors (0-1) per magnet
            polarizations: List of 3D vectors (px, py, pz) per magnet
            min_radius: Minimum radius for each ring (in meters)
            num_rings: Number of rings
            angles: List of relative angle parameters (0-1) per magnet
            num_magnets: Number of magnets per ring (or list)
            start_angle: Start angle (degrees) or list per ring
            end_angle: End angle (degrees) or list per ring
        """
        self.min_radius = min_radius
        self.num_rings = num_rings
        self.num_magnets = [num_magnets] if isinstance(num_magnets, int) else num_magnets
        self.start_angle = [start_angle] if isinstance(start_angle, (int, float)) else start_angle
        self.end_angle = [end_angle] if isinstance(end_angle, (int, float)) else end_angle
        self.dimensions = dimensions
        self.polarizations = polarizations
        self.angles = angles

        self.compute_magnets()
        self._update_magnets()

    def compute_magnets(self):
        """Compute positions and parameters of magnets in the ring"""
        self.magnets_params = []
        idx = 0
        s_allowed_list = []
        self.radiuses = [self.min_radius]

        for j in range(self.num_rings):
            radius = self.radiuses[j]
            N = self.num_magnets[j]
            theta_start = np.radians(self.start_angle[j])
            theta_end = np.radians(self.end_angle[j])
            total_angle = theta_end - theta_start
            delta_theta = total_angle / N
            s_min = 0.001
            s_max = 2 * radius * np.abs(np.sin(delta_theta/2))
            s_allowed = 0.65 * s_max
            s_allowed_list.append(s_allowed)
            s_allowed = min(s_allowed_list)
            self.radiuses.append(radius + (max(s_allowed_list)+0.005))

            for i in range(N):
                s_i = self.dimensions[idx] * (s_allowed - s_min)
                t_i = self.angles[idx]
                p_i = self.polarizations[idx]
                idx += 1

                theta_i = 2 * np.arcsin(s_i / (2 * radius))
                sector_center_angle = theta_start + i * delta_theta + delta_theta / 2

                angle_min = sector_center_angle - (delta_theta - theta_i) / 2
                angle_max = sector_center_angle + (delta_theta - theta_i) / 2

                angle = angle_min + t_i * (angle_max - angle_min)

                self.magnets_params.append([angle, radius, s_i, p_i])

    def _update_magnets(self):
        """Create Magpylib Cuboid magnets from parameters"""
        self.magnets = []
        for angle, radius, size, polarization in self.magnets_params:
            cube = magpy.magnet.Cuboid(
                dimension=(size, size, size),
                polarization=polarization,
                position=(radius, 0, 0)
            )
            cube.rotate_from_angax(np.degrees(angle), 'z', anchor=0)
            self.magnets.append(cube)

    def get_collection(self):
        return magpy.Collection(self.magnets, override_parent=True)

    def get_field_at_point(self, point: Tuple[float, float, float]) -> np.ndarray:
        return self.get_collection().getB(point)

    def get_field_amplitude_at_point(self, point: Tuple[float, float, float]) -> float:
        return np.linalg.norm(self.get_field_at_point(point))

    def visualize(self):
        """2D field plot in XY plane at z=0"""
        fig, ax = plt.subplots()
        x = np.linspace(-0.2, 0.2, 100)
        y = np.linspace(-0.2, 0.2, 100)
        X, Y = np.meshgrid(x, y)
        points = np.array([X.flatten(), Y.flatten(), np.zeros_like(X.flatten())]).T

        B = np.array([self.get_field_at_point(pt) for pt in points])
        Bx = B[:, 0].reshape(X.shape)
        By = B[:, 1].reshape(X.shape)
        Bamp = np.linalg.norm(B, axis=1).reshape(X.shape)

        pc = ax.contourf(X, Y, Bamp, levels=50, cmap="coolwarm")
        ax.streamplot(X, Y, Bx, By, color="k", density=1.5, linewidth=1)
        fig.colorbar(pc, ax=ax, label="|B| [T]")
        ax.set(xlabel="x [m]", ylabel="y [m]", aspect=1)
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(-0.2, 0.2)
        plt.show()

    def visualize_structure(self):
        self.get_collection().show(backend='plotly')

if __name__ == "__main__":
    N = [6, 3]
    dimensions = [1] * 9
    polarizations = [(-1.6, 0, 0)] * 9
    angles = [0.4, 0.01, 0.99, 0.4, 0.01, 0.99, 0.4, 0.01, 0.99]
    ring = HalbachRing(
        dimensions=dimensions,
        polarizations=polarizations,
        min_radius=0.12,
        num_rings=2,
        num_magnets=N,
        start_angle=[0, 0],
        end_angle=[180, 160],
        angles=angles
    )

    point = np.array([0, 0, 0])
    B_amplitude = ring.get_field_amplitude_at_point(point)
    print(f"Magnetic field amplitude at (0,0): {B_amplitude:.6f} T")

    ring.visualize()
    ring.visualize_structure()
