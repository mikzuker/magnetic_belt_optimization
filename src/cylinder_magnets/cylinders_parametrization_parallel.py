import numpy as np
import magpylib as magpy
import matplotlib.pyplot as plt
from typing import Union, List, Tuple
from pathlib import Path
from typing import Optional
from scipy.spatial.transform import Rotation

class HalbachRing_Cylinders(object):
    def __init__(self,
                 diameter: float,
                 height: float,
                 polarizations: List[Tuple[float, float, float]],
                 min_radius: float,
                 num_rings: int,
                 angles: List[float],
                 num_magnets: Union[int, List[int]],
                 start_angle: Union[float, List[float]],
                 end_angle: Union[float, List[float]]):
        """
        Initialize a ring of cylinder magnets arranged on circular arcs.

        Args:
            diameter: Diameter of a magnet [m]
            height: Height of a magnet [m]
            polarizations: Base polarization vector per ring (will be rotated)
            min_radius: Radius of first ring [m]
            num_rings: Number of concentric rings
            angles: Not used (placeholder for future angle-dependent properties)
            num_magnets: Number of magnets per ring (int or list)
            start_angle: Start angle (degrees) per ring (scalar or list)
            end_angle: End angle (degrees) per ring (scalar or list)
        """
        self.diameter = diameter
        self.height = height
        self.polarizations = polarizations
        self.min_radius = min_radius
        self.num_rings = num_rings
        self.angles = angles
        self.num_magnets = [num_magnets] * num_rings if isinstance(num_magnets, int) else num_magnets
        self.start_angle = [start_angle] * num_rings if isinstance(start_angle, (int, float)) else start_angle
        self.end_angle = [end_angle] * num_rings if isinstance(end_angle, (int, float)) else end_angle

        self.compute_magnets()
        self._update_magnets()

    def compute_magnets(self):
        self.magnets_params = []
        self.radiuses = [self.min_radius]
        idx = 0

        for j in range(self.num_rings):
            num = self.num_magnets[j]
            radius = self.radiuses[j]

            start_deg = self.start_angle[j]
            end_deg = self.end_angle[j]
            arc_deg = end_deg - start_deg
            sector_width = arc_deg / num 

            # Minimum angle between magnets (to avoid intersections)
            min_angle = 2 * np.arcsin(self.diameter / (2 * radius))
            min_angle_deg = np.degrees(min_angle)

            # Available space for movement inside the sector
            free_space = sector_width - min_angle_deg

            for i in range(num):
                # Uniform grid: center of i-th sector
                sector_center = start_deg + (i + 0.5) * sector_width

                # Offset inside the sector (0 = left, 1 = right)
                angle_param = self.angles[idx]  # 0..1
                angle_offset = (angle_param - 0.5) * free_space  # [-free_space/2, +free_space/2]
                angle_deg = sector_center + angle_offset
                angle_rad = np.radians(angle_deg)

                x = radius * np.cos(angle_rad)
                y = radius * np.sin(angle_rad)
                position = (x, y, 0)

                """Radial polarization"""
                # polarization = (
                #     np.cos(angle_rad) * self.polarizations[j][0],
                #     np.sin(angle_rad) * self.polarizations[j][1],
                #     self.polarizations[j][2],
                # )

                """Free polarization"""
                polarization = self.polarizations[j]

                self.magnets_params.append([self.diameter, self.height, polarization, position])
                idx += 1

            self.radiuses.append(radius + 1.1 * self.diameter)

    def _update_magnets(self):
        """Create Magpylib Cylinder objects from magnet parameters."""
        self.magnets = []
        for diameter, height, polarization, position in self.magnets_params:
            cyl = magpy.magnet.Cylinder(
                dimension=(diameter, height),
                polarization=polarization,
                position=position
            )

            self.magnets.append(cyl)

    def get_collection(self):
        return magpy.Collection(self.magnets, override_parent=True)

    def get_field_at_point(self, point: Tuple[float, float, float]) -> np.ndarray:
        return self.get_collection().getB(point)

    def get_gradient_at_point(self, point: np.ndarray) -> float:
        epsilon = 1e-6
        grad_squared = 0.0

        for i in range(3):  # x, y, z
            point_plus = point.copy()
            point_minus = point.copy()
            point_plus[i] += epsilon
            point_minus[i] -= epsilon

            B_plus = self.get_collection().getB(point_plus)  # shape (3,)
            B_minus = self.get_collection().getB(point_minus)  # shape (3,)
        
            dB = (B_plus - B_minus) / (2 * epsilon)  # dB/dx_i
            grad_squared += np.sum(dB ** 2)  # sum over components of B

        return grad_squared

    def get_field_amplitude_at_point(self, point: Tuple[float, float, float]) -> float:
        return np.linalg.norm(self.get_field_at_point(point))

    def visualize(self, path: Optional[Union[str, Path]] = None):
        """2D field plot in XY plane at z=0."""
        fig, ax = plt.subplots()
        x = np.linspace(-0.3, 0.3, 150)
        y = np.linspace(-0.3, 0.3, 150)
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
        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.3, 0.3)

        if path is not None:
            save_path = Path(path)
            save_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path / "field_plot.pdf")
            plt.close()
        else:
            plt.show()

    def visualize_structure(self):
        self.get_collection().show(backend='matplotlib')


if __name__ == "__main__":
    N = [2]
    diameters = 0.03
    heights = 0.09
    polarizations = [(1, 1, 0)]*2
    angles = [1, 0]

    ring = HalbachRing_Cylinders(
        diameter=diameters,
        height=heights,
        polarizations=polarizations,
        min_radius=0.1,
        num_rings=1,
        num_magnets=[2],
        start_angle=[0],
        end_angle=[360],
        angles=angles,
    )
    print(ring.get_gradient_at_point([0, 0, 0]))
    ring.visualize_structure()
    # ring.visualize(path="cylinders_optimization")
