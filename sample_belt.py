import numpy as np
import magpylib as magpy
import matplotlib.pyplot as plt

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

# for a in angles_down:
#     cube = magpy.magnet.Cuboid(
#         dimension=(0.035,0.035,0.035),
#         polarization=(1.5,0,0),
#         position=(0.12,0,0)
#     )
#     cube.rotate_from_angax(a, 'z', anchor=0)
#     # cube.rotate_from_angax(a, 'z')
#     halbach.add(cube)

halbach.show(backend='plotly')

fig, ax = plt.subplots()

# Compute and plot field on x-y grid
grid = np.mgrid[-0.2:0.2:100j, -0.2:0.2:100j, 0:0:1j].T[0]
X, Y, _ = np.moveaxis(grid, 2, 0)

B = halbach.getB(grid)
Bx, By, _ = np.moveaxis(B, 2, 0)
Bamp = np.linalg.norm(B, axis=2)

pc = ax.contourf(X, Y, Bamp, levels=50, cmap="coolwarm")
ax.streamplot(X, Y, Bx, By, color="k", density=1.5, linewidth=1)

# Add colorbar
fig.colorbar(pc, ax=ax, label="|B|")

# Figure styling
ax.set(
    xlabel="x-position",
    ylabel="y-position",
    aspect=1,
)

plt.xlim(-0.2, 0.2)
plt.ylim(-0.2, 0.2)
# plt.show()

# Calculate field at point (0,0)
point = np.array([0, 0, 0])
B_at_origin = halbach.getB(point)
B_amplitude_at_origin = np.linalg.norm(B_at_origin)
print(f"Magnetic field amplitude at (0,0): {B_amplitude_at_origin:.6f} T")