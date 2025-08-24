import numpy as np
import matplotlib.pyplot as plt
import os
import json
from cylinders_parametrization_orthogonal import HalbachRing_Cylinders
from pathlib import Path
from tqdm import tqdm

class Optimizer_Cylinders(object):
    def __init__(self, 
                 seed, 
                 number_of_magnets, 
                 min_radius, 
                 num_rings, 
                 start_angle,
                 end_angle,
                 point_to_maximize,
                 iterations,
                 experiment_dir, 
                 learning_rate,
                 diameter=0.06,
                 height=0.06,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 ):
        self.number_of_magnets = number_of_magnets
        self.min_radius = min_radius
        self.num_rings = num_rings
        self.seed = seed
        self.experiment_dir = experiment_dir
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.point_to_maximize = point_to_maximize
        self.diameter = diameter
        self.height = height
        self.best_params = []
        self.best_loss = 0
        self.params = []
        self.losses = []
        
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.bounds = []

        # Bounds for angles (0.0 to 1.0)
        self.bounds.extend([(0.0, 1.0) for _ in range(sum(self.number_of_magnets))])
        # Bounds for polarizations (-0.5 to 0.5)
        self.bounds.extend([(-0.5, 0.5) for _ in range(sum(self.number_of_magnets))])
    
    def generate_initial_parameters(self):
        np.random.seed(self.seed)
        # Generate initial angles for the magnets
        self.params.extend([0.5] * sum(self.number_of_magnets))
        # Generate initial polarizations for the magnets
        self.params.extend([0.5 for _ in range(sum(self.number_of_magnets))])

        initial_ring = HalbachRing_Cylinders(
            diameter=self.diameter,
            height=self.height,
            polarizations=self.params[sum(self.number_of_magnets):],
            min_radius=self.min_radius,
            num_rings=self.num_rings,
            num_magnets=self.number_of_magnets,
            start_angle=self.start_angle,
            end_angle=self.end_angle,
            angles=self.params[:sum(self.number_of_magnets)],
        )

        return initial_ring
    
    def flatten_params(self, params):
        """Convert a list of parameters into a single one-dimensional list."""
        flattened = []
        for param in params:
            if isinstance(param, (list, np.ndarray)):
                flattened.extend([float(x) for x in param])
            else:
                flattened.append(float(param))
        return flattened
    
    def unflatten_params(self, flattened):
        """Convert a single one-dimensional list back into a list of parameters."""
        unflattened = []
    
        # Extract angles
        unflattened.extend([float(x) for x in flattened[:sum(self.number_of_magnets)]])
    
        # Extract polarizations (simple floats, not vectors)
        start_idx = sum(self.number_of_magnets)
        for i in range(sum(self.number_of_magnets)):
            polarization = float(flattened[start_idx + i])
            unflattened.append(polarization)
        return unflattened
    
    def create_ring(self, params):
        ring = HalbachRing_Cylinders(
            diameter=self.diameter,
            height=self.height,
            polarizations=params[sum(self.number_of_magnets):],
            min_radius=self.min_radius,
            num_rings=self.num_rings,
            num_magnets=self.number_of_magnets,
            start_angle=self.start_angle,
            end_angle=self.end_angle,
            angles=params[:sum(self.number_of_magnets)],
        )
        return ring

    def objective_function(self, params):
        ring = self.create_ring(params)
    
        if self.check_collisions(ring):
            return 1e10
    
        point = np.array(self.point_to_maximize)
        amplitude = ring.get_field_amplitude_at_point(point)
        return -amplitude*1e3

    def check_collisions(self, ring):
        positions = [m.position for m in ring.magnets]
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))
                if dist < self.diameter:
                    return True
        return False

    def gradient(self, params, epsilon=1e-6):
        """Calculate numerical gradient of objective function"""
        params_flattened = self.flatten_params(params)
        grad = [0.0 for _ in range(len(params_flattened))]
        for i in range(len(params_flattened)):
            params_plus = params_flattened.copy()
            params_minus = params_flattened.copy()
            params_plus[i] = float(params_plus[i]) + epsilon
            params_minus[i] = float(params_minus[i]) - epsilon
            
            # Apply bounds during gradient calculation
            if i < len(self.bounds):
                lower, upper = self.bounds[i]
                params_plus[i] = np.clip(params_plus[i], lower, upper)
                params_minus[i] = np.clip(params_minus[i], lower, upper)
            
            f_plus = self.objective_function(self.unflatten_params(params_plus))
            f_minus = self.objective_function(self.unflatten_params(params_minus))
            grad[i] = float((f_plus - f_minus) / (2 * epsilon))
        return grad
    
    def save_progress(self, iteration, params):
        """Save progress of optimization"""
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir, exist_ok=True)
        with open(os.path.join(self.experiment_dir, f'optimization_progress.json'), 'w') as f:
                json.dump({'losses': self.losses}, f)
    
    def save_best_params(self, params):
        """Save optimized parameters to JSON file"""

        params_json = {
            "diameter": self.diameter,
            "height": self.height,
            "polarizations": params[sum(self.number_of_magnets):],
            "min_radius": self.min_radius,
            "num_rings": self.num_rings,
            "num_magnets": self.number_of_magnets,
            "start_angle": self.start_angle,
            "end_angle": self.end_angle,
            "angles": params[:sum(self.number_of_magnets)],
            "amplitude": -self.losses[-1],
            "point_to_maximize": self.point_to_maximize,
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "improvement_percent": float((self.losses[-1] / self.losses[0]) * 100),
            "optimizer": "Adam",
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
        }
        
        with open(f"{self.experiment_dir}/best_parameters.json", "w") as f:
            json.dump(params_json, f, indent=4)

    
    def plot_history(self):
        """Plot the history of the optimization"""
        amplitudes = [-loss for loss in self.losses]
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(amplitudes)), amplitudes)
        plt.xlabel('Iteration')
        plt.ylabel('Field Amplitude [T]')
        plt.title('Optimization History')
        plt.savefig(os.path.join(self.experiment_dir, f'optimization_history.pdf'))
        plt.close()

    def plot_field(self, params):
        """Plot the field of the ring"""
        ring = self.create_ring(params)
        ring.visualize_structure()

    def optimize(self):
        self.generate_initial_parameters()
        
        params_flat = self.flatten_params(self.params)
        m = np.zeros_like(params_flat) 
        v = np.zeros_like(params_flat) 
        t = 0 
        
        for i in tqdm(range(self.iterations), desc="Optimizing"):
            t += 1
            grad = self.gradient(self.params)
            grad = np.array(grad)
            
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
            
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)
            
            params_flat = params_flat - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
            for j, (lower, upper) in enumerate(self.bounds):
                params_flat[j] = np.clip(params_flat[j], lower, upper)
            
            self.params = self.unflatten_params(params_flat)
            self.losses.append(self.objective_function(self.params))
            self.best_params = self.params
            self.best_loss = self.losses[-1]

        save_path = Path(self.experiment_dir)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_progress(self.best_params, self.best_loss)
        self.save_best_params(self.best_params)
        self.plot_history()

        ring = self.create_ring(self.best_params)
        ring.visualize(path=save_path)
        print(f"Best loss: {self.best_loss}")
        ring.visualize_structure()
            
        return self.best_params, self.best_loss

if __name__ == "__main__":
    optimizer = Optimizer_Cylinders(
        seed=41,
        number_of_magnets=[7],
        min_radius=0.018,
        num_rings=1,
        start_angle=[0],
        end_angle=[360],
        point_to_maximize=[0, 0.00, 0],
        iterations=1000,
        experiment_dir="super_new_good_experiments/experiment_6",
        learning_rate=0.001,
        diameter=0.015,
        height=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-10,
    )
    optimizer.optimize() 