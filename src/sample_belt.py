import numpy as np
import magpylib as magpy
import matplotlib.pyplot as plt
import os

# Create figures directory if it doesn't exist
os.makedirs("figures", exist_ok=True)

# Force non-interactive backend to prevent display
plt.switch_backend('Agg')

N_up = 10
N_down = 8
angles_up = np.linspace(0, 180, N_up, endpoint=True)
angles_down = np.linspace(200, 360, N_down, endpoint=False)
print(angles_up)
print(angles_down)
halbach = magpy.Collection()

for a in angles_up:
    cube = magpy.magnet.Cuboid(
        dimension=(0.035,0.035,0.035),
        polarization=(-1.5,0,0),
        position=(0.12,0,0)
    )
    cube.rotate_from_angax(a, 'z', anchor=0)
    # cube.rotate_from_angax(a, 'z')
    halbach.add(cube)

# Create a single figure with three subplots
fig = plt.figure(figsize=(18, 6))

# First subplot: 3D halbach visualization (default view)
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
magpy.show(halbach, backend="matplotlib", canvas=ax1, 
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
magpy.show(halbach, backend="matplotlib", canvas=ax2, 
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

B = halbach.getB(grid)
Bx, By, _ = np.moveaxis(B, 2, 0)
Bamp = np.linalg.norm(B, axis=2)

pc = ax3.contourf(X, Y, Bamp, levels=50, cmap="coolwarm")
ax3.streamplot(X, Y, Bx, By, color="k", density=1.5, linewidth=1)
# Remove colorbar
fig.colorbar(pc, ax=ax3, label="|B|")

ax3.set(
    xlabel="",
    ylabel="",
    aspect=1,
    title="Magnetic Field"
)

ax3.set_xlim(-0.2, 0.2)
ax3.set_ylim(-0.2, 0.2)
# Remove ticks
ax3.set_xticks([])
ax3.set_yticks([])

# Adjust layout and save combined figure
plt.tight_layout()
fig.savefig("figures/combined_visualization.png", dpi=300, bbox_inches="tight")
plt.close(fig)  # Close the figure to prevent display

point = np.array([0, 0, 0])
B_at_origin = halbach.getB(point)
B_amplitude_at_origin = np.linalg.norm(B_at_origin)
print(f"Magnetic field amplitude at (0,0): {B_amplitude_at_origin:.6f} T")