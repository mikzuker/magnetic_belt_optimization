import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time
import datetime
from cube_magnets.cuboids_parametrization import HalbachRing_Cuboids


class AdamHalbachOptimizer:
    def __init__(self, initial_halbach_ring, experiment_dir=None, learning_rate=0.01):
        """
        Initialize the Adam optimizer for a Halbach ring.
        
        Args:
            initial_halbach_ring: Initial HalbachRing object to optimize
            experiment_dir: Directory to save experiment results
            learning_rate: Learning rate for Adam optimizer
        """
        self.initial_ring = initial_halbach_ring
        self.learning_rate = learning_rate
        
        # Set experiment directory
        self.experiment_dir = experiment_dir or "optimization_results/adam_experiment_default"
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Store the initial field amplitude
        point = np.array([0, 0, 0])
        self.initial_amplitude = initial_halbach_ring.get_field_amplitude_at_point(point)
        print(f"Initial field amplitude stored: {self.initial_amplitude:.6f} T")
        
        # Initialize optimization parameters
        self.setup_parameters()
        
        # Adam optimizer parameters
        self.beta1 = 0.9  # First moment decay rate
        self.beta2 = 0.995  # Second moment decay rate
        self.epsilon = 1e-8  # Small constant for numerical stability
        
        # Initialize Adam moments
        self.m = np.zeros_like(self.initial_params)  # First moment
        self.v = np.zeros_like(self.initial_params)  # Second moment
        self.t = 0  # Time step
        
        # Track optimization history
        self.history = []
        self.best_solution = None
        self.best_amplitude = self.initial_amplitude
    
    def setup_parameters(self):
        """Set up parameters for optimization"""
        # Flatten all parameters into a single array
        self.initial_params = []
        
        # Number of magnets per ring (discrete parameter)
        self.initial_params.extend(self.initial_ring.num_magnets)
        
        # Dimensions for each magnet
        self.initial_params.extend(self.initial_ring.dimensions)
        
        # Polarizations (flatten 3D vectors)
        for pol in self.initial_ring.polarizations:
            self.initial_params.extend(pol)
            
        # Angles
        self.initial_params.extend(self.initial_ring.angles)
        
        # Convert to numpy array
        self.initial_params = np.array(self.initial_params)
        
        # Set parameter bounds
        self.bounds = []
        n_rings = self.initial_ring.num_rings
        n_magnets = sum(self.initial_ring.num_magnets)
        n_pols = n_magnets * 3  # 3 components per polarization
        
        # Bounds for number of magnets per ring (3 to 3)
        for _ in range(n_rings):
            self.bounds.append((3, 3))
        
        # Bounds for dimensions (0.1 to 0.95)
        for _ in range(n_magnets):
            self.bounds.append((0.1, 0.95))
        
        # Bounds for polarizations (-0.5 to 0.5)
        for _ in range(n_pols):
            self.bounds.append((-0.5, 0.5))
            
        # Bounds for angles (0.2 to 0.6)
        for _ in range(n_magnets):
            self.bounds.append((0.2, 0.6))
        
        print(f"Total number of parameters: {len(self.initial_params)}")
        print(f"Parameters breakdown:")
        print(f"- Number of magnets per ring: {n_rings}")
        print(f"- Dimensions: {n_magnets}")
        print(f"- Polarizations: {n_pols}")
        print(f"- Angles: {n_magnets}")
    
    def params_to_halbach(self, params):
        """Convert flattened parameter array to HalbachRing parameters"""
        n_rings = self.initial_ring.num_rings
        
        # Extract number of magnets per ring and round to integers
        num_magnets = [int(round(params[i])) for i in range(n_rings)]
        
        # Calculate offsets for other parameters
        n_magnets_total = sum(num_magnets)
        dims_start = n_rings
        pols_start = dims_start + n_magnets_total
        angles_start = pols_start + n_magnets_total * 3
        
        # Extract dimensions
        dimensions = params[dims_start:dims_start + n_magnets_total].tolist()
        
        # Extract and reshape polarizations
        polarizations = []
        for i in range(n_magnets_total):
            start_idx = pols_start + i * 3
            pol = params[start_idx:start_idx + 3]
            # Normalize polarization vector
            norm = np.linalg.norm(pol)
            if norm > 1e-10:
                pol = pol / norm
            else:
                pol = np.array([1.0, 0.0, 0.0])
            polarizations.append(tuple(pol))
            
        # Extract angles
        angles = params[angles_start:angles_start + n_magnets_total].tolist()
        
        # Calculate end angles based on number of magnets
        end_angles = []
        for n in num_magnets:
            end_angle = 360 * (n-1)/n
            end_angles.append(end_angle)
        
        return {
            'dimensions': dimensions,
            'polarizations': polarizations,
            'min_radius': self.initial_ring.min_radius,
            'num_rings': n_rings,
            'num_magnets': num_magnets,
            'start_angle': [0] * n_rings,
            'end_angle': end_angles,
            'angles': angles
        }
    
    def create_ring(self, params):
        """Create a HalbachRing from parameter array"""
        halbach_params = self.params_to_halbach(params)
        
        try:
            ring = HalbachRing_Cuboids(
                dimensions=halbach_params['dimensions'],
                polarizations=halbach_params['polarizations'],
                min_radius=halbach_params['min_radius'],
                num_rings=halbach_params['num_rings'],
                num_magnets=halbach_params['num_magnets'],
                start_angle=halbach_params['start_angle'],
                end_angle=halbach_params['end_angle'],
                angles=halbach_params['angles']
            )
            return ring
        except ValueError as e:
            print(f"Error creating ring: {str(e)}")
            return None
    
    def objective_function(self, params):
        """Calculate magnetic field amplitude at center point"""
        ring = self.create_ring(params)
        if ring is None:
            return -self.initial_amplitude  # Return negative initial amplitude for invalid configurations
        
        point = np.array([0, 0.00, 0])
        try:
            amplitude = ring.get_field_amplitude_at_point(point)
            return -amplitude  # Negative because we want to maximize
        except Exception:
            return -self.initial_amplitude
    
    def gradient(self, params, epsilon=1e-6):
        """Calculate numerical gradient of objective function"""
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += epsilon
            params_minus[i] -= epsilon
            
            f_plus = self.objective_function(params_plus)
            f_minus = self.objective_function(params_minus)
            
            grad[i] = (f_plus - f_minus) / (2 * epsilon)
        return grad
    
    def optimize(self, max_iter=1000, save_frequency=10):
        """
        Run Adam optimization
        
        Args:
            max_iter: Maximum number of iterations
            save_frequency: How often to save progress
            
        Returns:
            optimized_ring: The optimized HalbachRing
            history: List of magnetic field amplitudes during optimization
        """
        # Initialize parameters
        params = self.initial_params.copy()
        self.history = []
        self.best_solution = None
        self.best_amplitude = self.initial_amplitude
        
        # print(f"Starting Adam optimization from amplitude: {self.initial_amplitude:.6f} T")
        
        # # Save initial ring visualization
        # plt.figure(figsize=(10, 8))
        # self.initial_ring.visualize()
        # plt.savefig(f"{self.experiment_dir}/initial_ring.png")
        # plt.close()
        
        for iteration in range(max_iter):
            # Calculate gradient
            grad = self.gradient(params)
            
            # Update Adam moments
            self.t += 1
            self.m = self.beta1 * self.m + (1 - self.beta1) * grad
            self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
            
            # Bias correction
            m_hat = self.m / (1 - self.beta1**self.t)
            v_hat = self.v / (1 - self.beta2**self.t)
            
            # Update parameters
            params = params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
            # Apply bounds
            for i, (lower, upper) in enumerate(self.bounds):
                params[i] = np.clip(params[i], lower, upper)
            
            # Evaluate current solution
            current_amplitude = -self.objective_function(params)
            self.history.append(current_amplitude)
            
            # Update best solution
            if current_amplitude > self.best_amplitude:
                self.best_amplitude = current_amplitude
                self.best_solution = params.copy()
                print(f"Iteration {iteration}: New best amplitude: {self.best_amplitude:.6f} T")
            
            # Save progress periodically
            if iteration % save_frequency == 0:
                self.save_progress(iteration, params)
        
        # Create final optimized ring
        optimized_ring = self.create_ring(self.best_solution)
        
        # Save final visualizations
        plt.figure(figsize=(10, 8))
        optimized_ring.visualize()
        plt.savefig(f"{self.experiment_dir}/best_halbach_amplitude_{self.best_amplitude:.6f}T.pdf")
        plt.close()
            
        # optimized_ring.visualize_structure()
        # plt.savefig(f"{self.experiment_dir}/best_halbach_structure.pdf")
        # plt.close()
            
        # Save optimization history
        self.plot_history()
            
        # Save parameters
        self.save_parameters(optimized_ring)
        
        return optimized_ring, self.history
    
    def save_progress(self, iteration, params):
        """Save current progress"""
        ring = self.create_ring(params)
        if ring:
            plt.figure(figsize=(10, 8))
            ring.visualize()
            plt.savefig(f"{self.experiment_dir}/iteration_{iteration}.png")
            plt.close()
    
    def plot_history(self):
        """Plot optimization history"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history)
        plt.axhline(y=self.initial_amplitude, color='r', linestyle='--', 
                   label=f'Initial amplitude: {self.initial_amplitude:.6f} T')
        plt.xlabel('Iteration')
        plt.ylabel('Magnetic Field Amplitude (T)')
        plt.title('Adam Optimization Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.experiment_dir}/optimization_history.png")
        plt.close()
    
    def save_parameters(self, optimized_ring):
        """Save optimized parameters to JSON file"""
        params_json = {
            "dimensions": optimized_ring.dimensions,
            "polarizations": [list(pol) for pol in optimized_ring.polarizations],
            "min_radius": optimized_ring.min_radius,
            "num_rings": optimized_ring.num_rings,
            "num_magnets": optimized_ring.num_magnets,
            "start_angle": optimized_ring.start_angle,
            "end_angle": optimized_ring.end_angle,
            "angles": [0.5] * sum(optimized_ring.num_magnets),
            "amplitude": float(self.best_amplitude),
            "improvement_percent": float((self.best_amplitude / self.initial_amplitude - 1) * 100)
        }
        
        with open(f"{self.experiment_dir}/best_parameters.json", "w") as f:
            json.dump(params_json, f, indent=4)


def main():
    experiment_dir = f"optimization_results/3_cubiods_experiment"
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"Saving results to: {experiment_dir}")
    
    # Create the initial Halbach ring
    initial_ring = HalbachRing_Cuboids(
        dimensions=[0.5, 0.5, 0.5],
        polarizations=[(0.5, 0.5, 0), (0.5, 0.5, 0), (0.5, 0.5, 0)],
        min_radius=0.055,
        num_rings=1,
        num_magnets=[3],
        start_angle=[0],
        end_angle=[360],
        angles=[0.5] * 3
    )
    
    # Create optimizer
    optimizer = AdamHalbachOptimizer(
        initial_ring,
        experiment_dir=experiment_dir,
        learning_rate=0.3
    )
    
    # Run optimization
    optimized_ring, history = optimizer.optimize(
        max_iter=100,
        save_frequency=100
    )


if __name__ == "__main__":
    main() 