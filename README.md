# Magnetic Field Optimization for Halbach Ring Configuration

This project provides tools for optimizing magnetic field configurations using Halbach rings. It implements a flexible parametrization system for magnet placement and size optimization.

## Features

- Flexible parametrization of magnet positions and sizes
- Support for multiple rings with different configurations
- Automatic calculation of maximum allowed magnet sizes
- Prevention of magnet intersections
- Magnetic field visualization and analysis
- Built-in optimization capabilities

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mikzuker/magnetic_belt_optimization.git
cd magnetic_belt_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from src.parametrization import HalbachRing

# Define parameters
N = [6, 3]  # Number of magnets in each ring
dimensions = [1] * 9  # Normalized size factors (0-1) for each magnet
polarizations = [(-1.6, 0, 0)] * 9  # Polarization vectors for each magnet
angles = [0.4, 0.01, 0.99, 0.4, 0.01, 0.99, 0.4, 0.01, 0.99]  # Normalized angles (0-1)

# Create Halbach ring configuration
ring = HalbachRing(
    dimensions=dimensions,
    polarizations=polarizations,
    min_radius=0.12,  # Minimum radius in meters
    num_rings=2,      # Number of rings
    num_magnets=N,    # Magnets per ring
    start_angle=[0, 0],    # Start angles in degrees
    end_angle=[180, 160],  # End angles in degrees
    angles=angles
)

# Calculate field at a point
point = (0, 0, 0)
B_amplitude = ring.get_field_amplitude_at_point(point)
print(f"Magnetic field amplitude at origin: {B_amplitude:.6f} T")

# Visualize
ring.visualize()  # 2D field plot
ring.visualize_structure()  # 3D structure visualization
```

### Parameters

- `dimensions`: List of normalized size factors (0-1) for each magnet
- `polarizations`: List of 3D polarization vectors (px, py, pz) for each magnet
- `min_radius`: Minimum radius for the innermost ring in meters
- `num_rings`: Number of rings in the configuration
- `num_magnets`: List of number of magnets in each ring
- `start_angle`: List of starting angles in degrees for each ring
- `end_angle`: List of ending angles in degrees for each ring
- `angles`: List of normalized angles (0-1) for each magnet's position

### Key Features

1. **Automatic Size Calculation**:
   - Maximum magnet sizes are calculated based on ring geometry
   - Prevents magnet intersections
   - Maintains minimum gaps between magnets

2. **Flexible Positioning**:
   - Normalized angle parameters (0-1) for easy optimization
   - Automatic conversion to actual positions
   - Support for partial rings and custom angle ranges

3. **Field Analysis**:
   - Calculate magnetic field at any point
   - Visualize field distribution
   - 3D structure visualization

## License

[Your License Here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.