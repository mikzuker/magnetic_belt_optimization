import numpy as np
import magpylib as magpy
import matplotlib.pyplot as plt
from typing import Union, List, Tuple
from pathlib import Path
from typing import Optional
from scipy.spatial.transform import Rotation


def compute_virtual_sphere(diameter: float, height: float):
    """Compute virtual sphere built around a cylinder with specified diameter and height."""

    radius = diameter / 2
    half_height = height / 2
    sphere_radius = np.sqrt(radius**2 + half_height**2)

    return sphere_radius


def relative_to_absolute_angle(x, theta_min, theta_max, R, r):
    assert 0 <= x <= 1
    assert r < R

    alpha = np.arcsin(r / R)  

    theta_start = theta_min + alpha
    theta_end   = theta_max - alpha
  
    theta = theta_start + x * (theta_end - theta_start)

    return theta


class HalbachRing_Cylinders(object):
    def __init__(self,
                 diameter: float,
                 height: float,
                 polarizations: List[float],
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
            polarization: Base polarization vector per ring (will be rotated)
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

        magnet_radius = self.diameter / 2
    
        for j in range(self.num_rings):
            num = self.num_magnets[j]
            radius = self.radiuses[j]
        
            min_angle_step = np.rad2deg(2 * np.arcsin(magnet_radius / radius))
        
            start_deg = self.start_angle[j]
            end_deg = self.end_angle[j]
            total_arc = end_deg - start_deg
        
            if num * min_angle_step > total_arc:
                raise ValueError(f"Недостаточно места для {num} магнитов в кольце {j}")
        
            angle_step = total_arc / num
        
            for i in range(num):
                angle_deg = start_deg + i * angle_step
                angle_rad = np.deg2rad(angle_deg)
            
                x = radius * np.cos(angle_rad)
                y = radius * np.sin(angle_rad)
                position = (x, y, 0)

                polarization = self.polarizations[idx]
                self.magnets_params.append([self.diameter, self.height, position, angle_deg, polarization])
                idx += 1
        
            self.radiuses.append(radius + self.height * 1.2)


    def _update_magnets(self):
        self.magnets = []

        for diameter, height, position, angle_deg, polarization in self.magnets_params:
            angle_rad = np.deg2rad(angle_deg)
            
            cyl = magpy.magnet.Cylinder(
            dimension=(diameter, height),
            polarization=(0, 0, -polarization),
            position=position
            )
            
            # Поворачиваем цилиндр, чтобы его ось была радиальной
            rot_around_z = Rotation.from_euler('z', angle_rad, degrees=False)
            rot_tilt = Rotation.from_euler('y', np.pi/2, degrees=False)
            total_rotation = rot_around_z * rot_tilt
            cyl.rotate(total_rotation)

            self.magnets.append(cyl)
        
        # cyl = magpy.magnet.Cylinder(
        #     dimension=(self.diameter, self.height),
        #     polarization=(0, 0, -polarization),
        #     position=(0.045, 0, 0)
        #     )
        
        # rot_tilt = Rotation.from_euler('y', np.pi/2, degrees=False)
        # cyl.rotate(rot_tilt)
        # self.magnets.append(cyl)


    def get_collection(self):
        return magpy.Collection(self.magnets, override_parent=True)

    def get_field_at_point(self, point: Tuple[float, float, float]) -> np.ndarray:
        return self.get_collection().getB(point)

    def get_gradient_at_point(self, point: np.ndarray) -> float:
        epsilon = 1e-8
        grad_squared = 0.0

        for i in range(3): 
            point_plus = point.copy()
            point_minus = point.copy()
            point_plus[i] += epsilon
            point_minus[i] -= epsilon

            B_plus = self.get_collection().getB(point_plus)
            B_minus = self.get_collection().getB(point_minus) 
        
            dB = (B_plus - B_minus) / (2 * epsilon) 
            grad_squared += np.sum(dB ** 2) 

        return grad_squared

    def get_field_amplitude_at_point(self, point: Tuple[float, float, float]) -> float:
        return np.linalg.norm(self.get_field_at_point(point))

    def visualize(self, path: Optional[Union[str, Path]] = None):
        """2D field plot in XY plane at z=0."""
        fig, ax = plt.subplots()
        x = np.linspace(-0.02, 0.02, 150)
        y = np.linspace(-0.02, 0.02, 150)
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
        ax.set_xlim(-0.02, 0.02)
        ax.set_ylim(-0.02, 0.02)

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
    diameters = 0.008
    heights = 0.001
    polarizations = [0.5, -0.1] 
    angles = [1.0, 0.0]

    ring = HalbachRing_Cylinders(
        diameter=diameters,
        height=heights,
        polarizations=polarizations,
        min_radius=0.015,
        num_rings=1,
        num_magnets=N,
        start_angle=[0],
        end_angle=[360],
        angles=angles,
    )
    print(ring.get_gradient_at_point([0, 0, 0]))
    ring.visualize_structure()
    ring.visualize()