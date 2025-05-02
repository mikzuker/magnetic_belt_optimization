# Magnetic Field Optimization for Halbach Ring

The following project implements a magnetic field optimization system for a Halbach half-ring configuration to create speciefied magnetic field.

## Features

- Parametric Halbach ring configuration
- Magnetic field calculation using magpylib
- Visualization of magnetic field distribution
- Support for custom magnet dimensions and polarizations

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

Basic usage example:

```python
from parametrization import HalbachRing
import numpy as np

# Create a half-ring with 10 magnets
dimensions = [(0.035, 0.035, 0.035) for _ in range(10)]
polarizations = [(-1.5, 0, 0) for _ in range(10)]

ring = HalbachRing(
    dimensions=dimensions,
    polarizations=polarizations,
    radius=0.12,
    num_magnets=10
)
```

## Parameters

### HalbachRing Class

- `dimensions`: List of magnet dimensions (x, y, z) in meters
- `polarizations`: List of polarization vectors (Bx, By, Bz) in Tesla
- `angles`: Optional list of rotation angles in degrees, if not specified all magnets are placed uniformly from the starting angle to the ending angle.
- `radius`: Ring radius in meters
- `num_magnets`: Number of magnets in the ring
- `start_angle`: Starting angle in degrees
- `end_angle`: Ending angle in degrees

## Visualization

The code includes built-in visualization capabilities:
- Contour plot of field amplitude
- Streamlines showing field direction
- Colorbar indicating field strength