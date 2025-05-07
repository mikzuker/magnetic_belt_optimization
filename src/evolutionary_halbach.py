import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time
import datetime
from scipy.optimize import differential_evolution
from parametrization import HalbachRing
from tqdm import tqdm


class EvolutionaryHalbachOptimizer:
    def __init__(self, initial_halbach_ring, fixed_params=None, experiment_dir=None):
        """
        Initialize the evolutionary optimizer for a Halbach ring.
        
        Args:
            initial_halbach_ring: Initial HalbachRing object to optimize
            fixed_params: Set of parameter names to keep fixed (not optimize)
            experiment_dir: Directory to save experiment results
        """
        self.initial_ring = initial_halbach_ring
        self.fixed_params = fixed_params or set()
        
        # Set experiment directory
        self.experiment_dir = experiment_dir or "optimization_results/experiment_default"
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Store scaling parameters from initial ring
        self.max_dimension = initial_halbach_ring.max_dimension
        self.max_polarization = initial_halbach_ring.max_polarization
        self.max_radius = initial_halbach_ring.max_radius
        
        # Store the initial field amplitude
        point = np.array([0, 0, 0])
        self.initial_amplitude = initial_halbach_ring.get_field_amplitude_at_point(point)
        print(f"Initial field amplitude stored: {self.initial_amplitude:.6f} T")
        
        # Set up optimization bounds
        self.setup_bounds()
        
        # Track optimization history
        self.history = []
        self.best_solution = None
        self.best_amplitude = self.initial_amplitude
        self.iteration_count = 0
        self.eval_count = 0
        self.save_frequency = 5  # Save best figure every 5 iterations
    
    def setup_bounds(self):
        """Set up parameter bounds for evolutionary optimization"""
        self.bounds = []
        self.param_structure = {}
        param_idx = 0
        
        # Dimensions - use relative values directly (0-1)
        if 'dimensions' not in self.fixed_params:
            dimensions = np.array(self.initial_ring.dimensions)
            flat_dims = dimensions.flatten()
            
            # Store starting index and length of dimensions
            self.param_structure['dimensions'] = {
                'start': param_idx,
                'length': len(flat_dims),
                'shape': dimensions.shape
            }
            
            # Add bounds for each dimension value (0.1 to 0.95)
            for _ in range(len(flat_dims)):
                self.bounds.append((0.1, 0.95))
                param_idx += 1
        
        # Polarizations - use direction vectors directly
        if 'polarizations' not in self.fixed_params:
            polarizations = np.array(self.initial_ring.polarizations)
            flat_pols = polarizations.flatten()
            
            # Store starting index and length of polarizations
            self.param_structure['polarizations'] = {
                'start': param_idx,
                'length': len(flat_pols),
                'shape': polarizations.shape
            }
            
            # Add bounds for each polarization value (-1 to 1)
            for _ in range(len(flat_pols)):
                self.bounds.append((-1.0, 1.0))
                param_idx += 1
        
        # Radii - use relative values directly (0-1)
        if 'radius' not in self.fixed_params:
            radius = np.array(self.initial_ring.radius)
            
            # Store starting index and length of radius
            self.param_structure['radius'] = {
                'start': param_idx,
                'length': len(radius),
                'shape': radius.shape
            }
            
            # Add bounds for each radius value with higher lower limit for inner radius
            for i in range(len(radius)):
                if i == 0:  # Inner radius
                    self.bounds.append((0.5, 0.95))  # Increased lower limit for inner radius
                else:  # Outer radius
                    self.bounds.append((0.1, 0.95))
                param_idx += 1
        
        # Total number of parameters
        self.n_params = param_idx
        print(f"Total number of optimizable parameters: {self.n_params}")
    
    def params_to_halbach(self, params):
        """Convert flattened parameter array to HalbachRing parameters"""
        # Initialize with initial parameters
        dimensions = np.array(self.initial_ring.dimensions)
        polarizations = np.array(self.initial_ring.polarizations)
        radius = np.array(self.initial_ring.radius)
        
        # Update dimensions if not fixed
        if 'dimensions' in self.param_structure:
            start = self.param_structure['dimensions']['start']
            length = self.param_structure['dimensions']['length']
            shape = self.param_structure['dimensions']['shape']
            flat_dims = params[start:start+length]
            dimensions = flat_dims.reshape(shape)
        
        # Update polarizations if not fixed
        if 'polarizations' in self.param_structure:
            start = self.param_structure['polarizations']['start']
            length = self.param_structure['polarizations']['length']
            shape = self.param_structure['polarizations']['shape']
            flat_pols = params[start:start+length]
            
            # Get raw polarization vectors
            raw_pols = flat_pols.reshape(shape)
            
            # Normalize polarization vectors
            polarizations = np.zeros_like(raw_pols)
            for i in range(len(raw_pols)):
                norm = np.linalg.norm(raw_pols[i])
                if norm > 1e-10:  # Avoid division by zero
                    polarizations[i] = raw_pols[i] / norm
                else:
                    # If too small, use default values
                    polarizations[i] = np.array([1.0, 0.0, 0.0])
        
        # Update radius if not fixed
        if 'radius' in self.param_structure:
            start = self.param_structure['radius']['start']
            length = self.param_structure['radius']['length']
            radius = params[start:start+length]
        
        return {
            'dimensions': dimensions.tolist(),
            'polarizations': polarizations.tolist(),
            'radius': radius.tolist(),
            'num_magnets': self.initial_ring.num_magnets,
            'start_angle': self.initial_ring.start_angle,
            'end_angle': self.initial_ring.end_angle,
            'max_dimension': self.max_dimension,
            'max_polarization': self.max_polarization,
            'max_radius': self.max_radius
        }
    
    def create_ring(self, params):
        """Create a HalbachRing from parameter array"""
        halbach_params = self.params_to_halbach(params)
        
        try:
            # Create ring with validation
            ring = HalbachRing(
                dimensions=halbach_params['dimensions'],
                polarizations=halbach_params['polarizations'],
                radius=halbach_params['radius'],
                num_magnets=halbach_params['num_magnets'],
                start_angle=halbach_params['start_angle'],
                end_angle=halbach_params['end_angle'],
                max_dimension=halbach_params['max_dimension'],
                max_polarization=halbach_params['max_polarization'],
                max_radius=halbach_params['max_radius'],
                validate=True
            )
            return ring
        except ValueError:
            # If validation fails, return None
            return None
    
    def objective_function(self, params):
        """
        Objective function for evolutionary optimization
        
        Returns:
            negative_amplitude: Negative magnetic field amplitude (to maximize amplitude)
                              or 0 for invalid configurations
        """
        # Increment evaluation counter
        self.eval_count += 1
        
        # Create a ring with current parameters
        ring = self.create_ring(params)
        
        # If configuration is invalid, return 0
        if ring is None:
            return 0.0
        
        # Calculate magnetic field amplitude at (0, 0, 0)
        point = np.array([0, 0, 0])
        try:
            B_amplitude = ring.get_field_amplitude_at_point(point)
            
            # Update history
            self.history.append(float(B_amplitude))
            
            # Update best solution if better (with small tolerance to avoid numerical issues)
            if B_amplitude > self.best_amplitude + 1e-5:  # Only update if meaningfully better
                self.best_amplitude = B_amplitude
                self.best_solution = params.copy()
                print(f"New best amplitude: {self.best_amplitude:.6f} T")
                
                # Only increment iteration counter (but don't save intermediate results)
                self.iteration_count += 1
            
            # Return negative amplitude for minimization
            return -B_amplitude
        except Exception as e:
            # If calculation fails, return 0
            print(f"Error in field calculation: {str(e)}")
            return 0.0
    
    def callback(self, xk, convergence):
        """Callback function for differential evolution"""
        # Only report every 10 iterations to reduce output
        if self.eval_count % 500 == 0:
            # Create a ring with current parameters
            ring = self.create_ring(xk)
            
            if ring is not None:
                # Calculate magnetic field amplitude at (0, 0, 0)
                point = np.array([0, 0, 0])
                try:
                    amplitude = ring.get_field_amplitude_at_point(point)
                    # Print progress with iteration count
                    print(f"Evaluation {self.eval_count} - Current: {amplitude:.6f} T, Best: {self.best_amplitude:.6f} T, Convergence: {convergence:.6f}")
                except Exception as e:
                    print(f"Evaluation {self.eval_count} - Error calculating field: {str(e)}")
    
    def optimize(self, max_iter=100, popsize=40, mutation=(0.5, 1.0), recombination=0.7, seed=None):
        """
        Run evolutionary optimization using SciPy's differential_evolution
        
        Args:
            max_iter: Maximum number of generations
            popsize: Population size as a multiplier of the number of parameters
            mutation: Differential weight. Either a scalar or a tuple (min, max)
            recombination: Crossover probability
            seed: Random seed for reproducibility
            
        Returns:
            optimized_ring: The optimized HalbachRing
            history: List of magnetic field amplitudes during optimization
        """
        # Reset history and counters
        self.history = []
        self.best_solution = None
        self.best_amplitude = self.initial_amplitude
        self.iteration_count = 0
        self.eval_count = 0
        
        # Initial field amplitude
        print(f"Starting evolutionary optimization from amplitude: {self.initial_amplitude:.6f} T")
        
        # Save initial ring
        self.initial_ring.visualize_combined(
            show=False, 
            save_path=f"{self.experiment_dir}/initial_ring.png"
        )
        
        # Use a smaller population size if we have many parameters
        actual_popsize = min(popsize, max(5, int(2000 / self.n_params)))
        print(f"Using population size: {actual_popsize} (total population: {actual_popsize * self.n_params})")
        
        # Estimate total number of evaluations
        estimated_evals = actual_popsize * self.n_params * max_iter
        
        # Create progress bar for evaluations
        progress_bar = tqdm(total=estimated_evals, desc="Optimizing", unit="eval")
        self.last_eval = 0
        
        # Define callback that updates the progress bar
        original_callback = self.callback
        def progress_callback(xk, convergence):
            # Update progress bar
            progress = self.eval_count - self.last_eval
            if progress > 0:
                progress_bar.update(progress)
                self.last_eval = self.eval_count
            
            # Call original callback
            return original_callback(xk, convergence)
        
        try:
            result = differential_evolution(
                self.objective_function,
                bounds=self.bounds,
                maxiter=max_iter,
                popsize=actual_popsize,
                mutation=mutation,
                recombination=recombination,
                callback=progress_callback,
                seed=seed,
                init='latinhypercube',  # Use Latin hypercube for better initialization
                workers=1  # Use single worker for history tracking
            )
            
            # Ensure progress bar is up to date
            progress_bar.update(self.eval_count - self.last_eval)
            
            print(f"Optimization completed: {result.message}")
            print(f"Function evaluations: {result.nfev}")
            print(f"Final objective value: {result.fun}")
            
            # Create final optimized ring using best solution found
            best_params = self.best_solution if self.best_solution is not None else result.x
            
        except Exception as e:
            print(f"Optimization failed with error: {str(e)}")
            print("Using best solution found so far...")
            best_params = self.best_solution if self.best_solution is not None else None
        finally:
            # Make sure progress bar is closed properly
            progress_bar.close()
        
        # If we have a best solution, create the optimized ring
        optimized_ring = None
        if best_params is not None:
            optimized_ring = self.create_ring(best_params)
            
            # Final visualization - save only the best result
            if optimized_ring:
                # Save the visualization of the best ring
                optimized_ring.visualize_combined(
                    show=False, 
                    save_path=f"{self.experiment_dir}/best_halbach_amplitude_{self.best_amplitude:.6f}T.png"
                )
                
                # Final field amplitude
                point = np.array([0, 0, 0])
                final_amplitude = optimized_ring.get_field_amplitude_at_point(point)
                
                # Print summary
                improvement = (final_amplitude / self.initial_amplitude - 1) * 100
                print(f"\nEvolutionary optimization completed:")
                print(f"Initial amplitude: {self.initial_amplitude:.6f} T")
                print(f"Final amplitude: {final_amplitude:.6f} T")
                print(f"Improvement: {improvement:.2f}%")
                
                # Save best parameters as JSON
                params_json = {
                    "dimensions": [list(dim) for dim in optimized_ring.dimensions],
                    "polarizations": [list(pol) for pol in optimized_ring.polarizations],
                    "radius": optimized_ring.radius,
                    "num_magnets": optimized_ring.num_magnets,
                    "start_angle": optimized_ring.start_angle,
                    "end_angle": optimized_ring.end_angle,
                    "max_dimension": optimized_ring.max_dimension,
                    "max_polarization": optimized_ring.max_polarization,
                    "max_radius": optimized_ring.max_radius,
                    "amplitude": float(final_amplitude),
                    "improvement_percent": float(improvement)
                }
                
                # Save to experiment directory
                with open(f"{self.experiment_dir}/best_parameters.json", "w") as f:
                    json.dump(params_json, f, indent=4)
                
                # Also save to main optimization_results directory like in optimize_halbach.py
                with open("optimization_results/best_parameters.json", "w") as f:
                    json.dump(params_json, f, indent=4)
                
                # Save visualization to main directory too
                optimized_ring.visualize_combined(
                    show=False, 
                    save_path=f"optimization_results/best_halbach_config.png"
                )
                
                print(f"Best parameters saved to: {self.experiment_dir}/best_parameters.json")
                print(f"Best parameters also saved to: optimization_results/best_parameters.json")
        else:
            print("No valid solution found during optimization.")
        
        # Plot and save optimization history
        if len(self.history) > 0:
            self.plot_history()
            
            # Save history data to JSON file
            history_data = {
                "initial_amplitude": float(self.initial_amplitude),
                "final_amplitude": float(self.best_amplitude),
                "field_amplitudes": [float(amp) for amp in self.history],
                "improvement_percent": float((self.best_amplitude / self.initial_amplitude - 1) * 100),
                "total_evaluations": len(self.history)
            }
            
            with open(f"{self.experiment_dir}/optimization_history.json", "w") as f:
                json.dump(history_data, f, indent=4)
            
            print(f"Optimization history saved ({len(self.history)} evaluations) to: {self.experiment_dir}/optimization_history.json")
        else:
            print("No optimization history to save - no evaluations were recorded")
        
        return optimized_ring, self.history
    
    def plot_history(self):
        """Plot optimization history"""
        if not self.history or len(self.history) == 0:
            print("No optimization history to plot")
            return
            
        print(f"Plotting optimization history with {len(self.history)} data points")
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Get evaluation numbers for x-axis
        evals = np.arange(len(self.history))
        
        # Plot magnetic field amplitude
        ax1.plot(evals, self.history)
        ax1.set_xlabel('Function Evaluation')
        ax1.set_ylabel('Magnetic Field Amplitude (T)')
        ax1.set_title('Evolutionary Optimization Progress - Field Amplitude')
        ax1.grid(True)
        
        # Add horizontal line for initial amplitude
        ax1.axhline(y=self.initial_amplitude, color='r', linestyle='--', 
                   label=f'Initial amplitude: {self.initial_amplitude:.6f} T')
        
        # Add horizontal line for best amplitude
        if self.best_amplitude > self.initial_amplitude:
            ax1.axhline(y=self.best_amplitude, color='g', linestyle='--', 
                       label=f'Best amplitude: {self.best_amplitude:.6f} T')
        
        # Add legend
        ax1.legend()
        
        # Calculate running best amplitude
        running_best = np.maximum.accumulate(self.history)
        
        # Plot running best amplitude
        ax2.plot(evals, running_best)
        ax2.set_xlabel('Function Evaluation')
        ax2.set_ylabel('Best Magnetic Field Amplitude So Far (T)')
        ax2.set_title('Optimization Progress - Best Amplitude Found')
        ax2.grid(True)
        
        # Add annotations for improvement percentages
        improvement = (self.best_amplitude / self.initial_amplitude - 1) * 100
        ax2.annotate(f'Total improvement: {improvement:.2f}%', 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=12, ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f"{self.experiment_dir}/optimization_history.png", dpi=300)
        plt.close()
        
        # Create a second figure with histogram of field amplitudes
        plt.figure(figsize=(10, 6))
        plt.hist(self.history, bins=50, alpha=0.7)
        plt.axvline(x=self.initial_amplitude, color='r', linestyle='--', 
                   label=f'Initial amplitude: {self.initial_amplitude:.6f} T')
        plt.axvline(x=self.best_amplitude, color='g', linestyle='--', 
                   label=f'Best amplitude: {self.best_amplitude:.6f} T')
        plt.xlabel('Magnetic Field Amplitude (T)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Field Amplitudes During Optimization')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.experiment_dir}/amplitude_distribution.png", dpi=300)
        plt.close()


def main():
    # Create a timestamp for this experiment
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"optimization_results/experiment_{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"Saving results to: {experiment_dir}")
    
    # Create a simpler test configuration for faster execution
    print("Running with simplified test configuration for faster execution")
    
    # Relative dimensions (will be scaled by max_dimension)
    dimensions = [(0.83, 0.75, 1.0) for _ in range(4)] + [(1.0, 1.0, 1.0) for _ in range(6)]

    # Alternating polarization pattern for Halbach effect (directions only)
    polarizations = []
    for i in range(4):
        angle = i * (360/4)
        px = np.cos(np.radians(angle))
        py = np.sin(np.radians(angle))
        polarizations.append((px, py, 0))

    for i in range(6):
        angle = i * (360/6) + 15  # 15 degree offset from inner ring
        px = -np.cos(np.radians(angle))
        py = -np.sin(np.radians(angle)) 
        polarizations.append((px, py, 0))

    # Calculate correct end angles to avoid intersections
    inner_end_angle = 360 * (4-1)/4  # = 270°
    outer_end_angle = 360 * (6-1)/6  # = 300°

    # Create the initial Halbach ring with relative values
    initial_ring = HalbachRing(
        dimensions=dimensions,
        polarizations=polarizations,
        radius=[0.67, 1.0],            # Relative radii: inner=0.67, outer=1.0
        num_magnets=[5, 5],           # 4 magnets in inner ring, 6 in outer
        start_angle=[0, 0],            # Both rings start at 0 degrees
        end_angle=[inner_end_angle, outer_end_angle],  # Use correct end angles
        max_dimension=(0.03, 0.02, 0.01),  # Maximum dimensions in meters
        max_polarization=1.4,              # Maximum polarization in Tesla
        max_radius=0.15,                   # Maximum radius in meters
        validate=True                      # Validate configuration
    )

    # Check initial magnetic field amplitude
    point = np.array([0, 0, 0])
    initial_amplitude = initial_ring.get_field_amplitude_at_point(point)
    print(f"Initial magnetic field amplitude at (0,0,0): {initial_amplitude:.6f} T")
    
    # Create optimizer (fix num_magnets, start_angle, and end_angle)
    optimizer = EvolutionaryHalbachOptimizer(
        initial_ring, 
        fixed_params={'num_magnets', 'start_angle', 'end_angle'},
        experiment_dir=experiment_dir
    )
    
    # Run evolutionary optimization with reduced settings for faster testing
    optimized_ring, history = optimizer.optimize(
        max_iter=10,               # Very few iterations for quick test
        popsize=20,                # Smaller population size for faster testing
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42
    )


if __name__ == "__main__":
    main() 