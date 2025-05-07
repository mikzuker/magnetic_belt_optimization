import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# Import our modules
from test_optimization import create_double_ring, run_test_optimization

def compare_field_distributions(initial_ring, optimized_ring, save_path=None):
    """Compare the magnetic field distributions of two Halbach ring configurations"""
    # Create a grid for measurement
    x = np.linspace(-0.1, 0.1, 100)
    y = np.linspace(-0.1, 0.1, 100)
    X, Y = np.meshgrid(x, y)
    grid_points = np.array([X.flatten(), Y.flatten(), np.zeros_like(X.flatten())]).T
    
    # Calculate field for initial ring
    B_initial = np.array([initial_ring.get_collection().getB(point) for point in grid_points])
    Bx_initial = B_initial[:, 0].reshape(X.shape)
    By_initial = B_initial[:, 1].reshape(X.shape)
    Bamp_initial = np.linalg.norm(B_initial, axis=1).reshape(X.shape)
    
    # Calculate field for optimized ring
    B_optimized = np.array([optimized_ring.get_collection().getB(point) for point in grid_points])
    Bx_optimized = B_optimized[:, 0].reshape(X.shape)
    By_optimized = B_optimized[:, 1].reshape(X.shape)
    Bamp_optimized = np.linalg.norm(B_optimized, axis=1).reshape(X.shape)
    
    # Make comparison plot
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(1, 3, figure=fig)
    
    # Initial field plot
    ax1 = fig.add_subplot(gs[0, 0])
    pc1 = ax1.contourf(X, Y, Bamp_initial, levels=50, cmap="coolwarm")
    ax1.streamplot(X, Y, Bx_initial, By_initial, color="k", density=1.5, linewidth=1)
    ax1.set_title("Initial Field Distribution")
    ax1.set_xlabel("x-position [m]")
    ax1.set_ylabel("y-position [m]")
    plt.colorbar(pc1, ax=ax1, label="|B| [T]")
    
    # Optimized field plot
    ax2 = fig.add_subplot(gs[0, 1])
    pc2 = ax2.contourf(X, Y, Bamp_optimized, levels=50, cmap="coolwarm")
    ax2.streamplot(X, Y, Bx_optimized, By_optimized, color="k", density=1.5, linewidth=1)
    ax2.set_title("Optimized Field Distribution")
    ax2.set_xlabel("x-position [m]")
    plt.colorbar(pc2, ax=ax2, label="|B| [T]")
    
    # Difference plot
    ax3 = fig.add_subplot(gs[0, 2])
    diff = Bamp_optimized - Bamp_initial
    pc3 = ax3.contourf(X, Y, diff, levels=50, cmap="RdBu_r")
    ax3.set_title("Difference (Optimized - Initial)")
    ax3.set_xlabel("x-position [m]")
    plt.colorbar(pc3, ax=ax3, label="Δ|B| [T]")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if required
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.close()

def compare_configurations():
    """Compare the initial and optimized Halbach ring configurations"""
    # Create output directory
    os.makedirs("comparison_results", exist_ok=True)
    
    # Create the initial double-ring configuration
    initial_ring = create_double_ring()
    
    # Run the optimization (this returns the optimized ring)
    optimized_ring, _ = run_test_optimization(use_double_ring=True)
    
    # Print basic comparison
    point = np.array([0, 0, 0])
    initial_amplitude = initial_ring.get_field_amplitude_at_point(point)
    final_amplitude = optimized_ring.get_field_amplitude_at_point(point)
    improvement = (final_amplitude / initial_amplitude - 1) * 100
    
    print(f"Magnetic field comparison at center point:")
    print(f"Initial: {initial_amplitude:.6f} T")
    print(f"Optimized: {final_amplitude:.6f} T")
    print(f"Improvement: {improvement:.2f}%")
    
    # Compare field distributions
    compare_field_distributions(
        initial_ring, 
        optimized_ring, 
        save_path="comparison_results/field_comparison.png"
    )
    
    # Generate combined visualizations
    initial_ring.visualize_combined(
        show=False,
        save_path="comparison_results/initial_configuration.png"
    )
    
    optimized_ring.visualize_combined(
        show=False,
        save_path="comparison_results/optimized_configuration.png"
    )
    
    print("Comparison completed. Results saved to comparison_results/ directory.")

if __name__ == "__main__":
    print("Comparing initial and optimized double-ring Halbach configurations...")
    compare_configurations() 