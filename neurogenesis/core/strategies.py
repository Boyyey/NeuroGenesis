"""
Evolution Strategies Module

Implements various evolution strategies for the NeuroGenesis framework.
"""

from enum import Enum, auto
from typing import List, Dict, Any, Tuple, Optional, Callable
import numpy as np
import tensorflow as tf
import random
from dataclasses import dataclass, field


class EvolutionStrategy(Enum):
    """Supported evolution strategies."""
    SIMPLE = auto()
    COVARIANCE_MATRIX_ADAPTATION = auto()
    DIFFERENTIAL_EVOLUTION = auto()
    PARTICLE_SWARM = auto()
    GENETIC_ALGORITHM = auto()
    NEUROEVOLUTION = auto()


@dataclass
class EvolutionConfig:
    """Configuration for evolution strategies."""
    strategy: EvolutionStrategy = EvolutionStrategy.SIMPLE
    population_size: int = 50
    elite_size: int = 5
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    sigma: float = 0.1  # For Gaussian mutation
    learning_rate: float = 0.01  # For gradient-based updates
    momentum: float = 0.9  # For momentum-based updates
    use_nesterov: bool = True  # For Nesterov momentum
    weight_decay: float = 1e-4  # L2 regularization
    batch_size: int = 32  # For batch processing
    max_grad_norm: float = 5.0  # For gradient clipping
    use_elitism: bool = True  # Whether to keep elite individuals
    use_adaptive_mutation: bool = True  # Whether to adapt mutation rates
    use_crossover: bool = True  # Whether to use crossover
    use_mutation: bool = True  # Whether to use mutation
    use_elite_crossover: bool = True  # Whether to include elites in crossover
    use_tournament_selection: bool = True  # Whether to use tournament selection
    tournament_size: int = 3  # Size of tournament for selection
    use_fitness_sharing: bool = False  # Whether to use fitness sharing
    fitness_sharing_sigma: float = 1.0  # Sigma for fitness sharing
    use_speciation: bool = False  # Whether to use speciation
    species_count: int = 5  # Number of species
    compatibility_threshold: float = 3.0  # Compatibility threshold for speciation
    min_species_size: int = 5  # Minimum size of a species
    max_stagnation: int = 15  # Max generations without improvement before reset
    reset_on_stagnation: bool = True  # Whether to reset on stagnation
    reset_type: str = 'partial'  # 'partial' or 'full' reset


class EvolutionStrategyBase:
    """Base class for all evolution strategies."""
    
    def __init__(self, config: EvolutionConfig):
        """Initialize the evolution strategy."""
        self.config = config
        self.population = []
        self.fitness_history = []
        self.best_solution = None
        self.best_fitness = -float('inf')
        self.generation = 0
        self.stagnation_counter = 0
    
    def initialize_population(self, model_fn, *args, **kwargs):
        """Initialize the population of models."""
        self.population = [model_fn(*args, **kwargs) for _ in range(self.config.population_size)]
    
    def evaluate_population(self, evaluate_fn):
        """Evaluate the fitness of each individual in the population."""
        fitness_scores = []
        for individual in self.population:
            fitness = evaluate_fn(individual)
            fitness_scores.append(fitness)
        
        # Update best solution
        best_idx = np.argmax(fitness_scores)
        if fitness_scores[best_idx] > self.best_fitness:
            self.best_fitness = fitness_scores[best_idx]
            self.best_solution = self.population[best_idx]
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        
        self.fitness_history.append(max(fitness_scores))
        return fitness_scores
    
    def evolve(self, evaluate_fn):
        """Perform one generation of evolution."""
        raise NotImplementedError("Subclasses must implement evolve()")
    
    def select_parents(self, fitness_scores):
        """Select parents for reproduction."""
        if self.config.use_tournament_selection:
            return self._tournament_selection(fitness_scores)
        else:
            return self._roulette_wheel_selection(fitness_scores)
    
    def _tournament_selection(self, fitness_scores):
        """Select parents using tournament selection."""
        parents = []
        for _ in range(2):  # Select 2 parents
            tournament = np.random.choice(
                len(self.population),
                size=min(self.config.tournament_size, len(self.population)),
                replace=False
            )
            winner = tournament[np.argmax([fitness_scores[i] for i in tournament])]
            parents.append(self.population[winner])
        return parents
    
    def _roulette_wheel_selection(self, fitness_scores):
        """Select parents using roulette wheel selection."""
        # Convert to probabilities
        probs = np.array(fitness_scores) - min(fitness_scores)
        if np.sum(probs) > 0:
            probs = probs / np.sum(probs)
        else:
            probs = np.ones_like(fitness_scores) / len(fitness_scores)
        
        # Select parents
        idx1, idx2 = np.random.choice(len(self.population), size=2, p=probs, replace=False)
        return [self.population[idx1], self.population[idx2]]
    
    def _mutate_weights(self, weights, mutation_rate=None):
        """Apply mutation to model weights."""
        if mutation_rate is None:
            mutation_rate = self.config.mutation_rate
        
        new_weights = []
        for w in weights:
            if np.random.random() < mutation_rate:
                # Gaussian mutation
                mutation = np.random.normal(scale=self.config.sigma, size=w.shape)
                new_weights.append(w + mutation)
            else:
                new_weights.append(w.copy())
        return new_weights
    
    def _crossover_weights(self, weights1, weights2):
        """Perform crossover between two sets of weights."""
        if not self.config.use_crossover or np.random.random() > self.config.crossover_rate:
            return weights1, weights2
        
        new_weights1 = []
        new_weights2 = []
        
        for w1, w2 in zip(weights1, weights2):
            if len(w1.shape) > 1:  # Weight matrix
                # Single point crossover
                idx = np.random.randint(0, w1.shape[0])
                new_w1 = np.concatenate([w1[:idx], w2[idx:]])
                new_w2 = np.concatenate([w2[:idx], w1[idx:]])
            else:  # Bias vector
                # Uniform crossover
                mask = np.random.random(size=w1.shape) < 0.5
                new_w1 = np.where(mask, w1, w2)
                new_w2 = np.where(mask, w2, w1)
            
            new_weights1.append(new_w1)
            new_weights2.append(new_w2)
        
        return new_weights1, new_weights2


class SimpleEvolution(EvolutionStrategyBase):
    """Simple genetic algorithm with mutation and crossover."""
    
    def __init__(self, config: EvolutionConfig):
        """Initialize the simple evolution strategy."""
        super().__init__(config)
        
    def evolve(self, evaluate_fn):
        """Perform one generation of evolution."""
        # Evaluate current population
        fitness_scores = self.evaluate_population(evaluate_fn)
        
        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]  # Descending order
        self.population = [self.population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]
        
        # Elitism: keep the best individuals
        new_population = []
        if self.config.use_elitism and self.config.elite_size > 0:
            elite_size = min(self.config.elite_size, len(self.population))
            new_population = self.population[:elite_size]
        
        # Generate offspring
        while len(new_population) < self.config.population_size:
            # Select parents
            parent1, parent2 = self.select_parents(fitness_scores)
            
            # Get model weights
            weights1 = parent1.get_weights()
            weights2 = parent2.get_weights()
            
            # Crossover
            child1_weights, child2_weights = self._crossover_weights(weights1, weights2)
            
            # Mutation
            if self.config.use_mutation:
                child1_weights = self._mutate_weights(child1_weights)
                child2_weights = self._mutate_weights(child2_weights)
            
            # Create new models with the new weights
            child1 = tf.keras.models.clone_model(parent1)
            child1.set_weights(child1_weights)
            
            child2 = tf.keras.models.clone_model(parent2)
            child2.set_weights(child2_weights)
            
            new_population.extend([child1, child2])
        
        # Update population
        self.population = new_population[:self.config.population_size]
        self.generation += 1
        
        # Check for stagnation
        if self.stagnation_counter >= self.config.max_stagnation and self.config.reset_on_stagnation:
            self._handle_stagnation()
        
        return max(fitness_scores)
    
    def _handle_stagnation(self):
        """Handle population stagnation."""
        if self.config.reset_type == 'full':
            # Full reset: reinitialize the entire population
            self.population = [tf.keras.models.clone_model(self.best_solution) 
                             for _ in range(self.config.population_size)]
        else:
            # Partial reset: keep the best individual and mutate the rest
            self.population = [self.best_solution]
            for _ in range(1, self.config.population_size):
                new_model = tf.keras.models.clone_model(self.best_solution)
                new_weights = self._mutate_weights(new_model.get_weights(), 
                                                 mutation_rate=0.5)  # Higher mutation rate
                new_model.set_weights(new_weights)
                self.population.append(new_model)
        
        self.stagnation_counter = 0
        print(f"Population reset due to stagnation at generation {self.generation}")


class CMAES(EvolutionStrategyBase):
    """Covariance Matrix Adaptation Evolution Strategy (CMA-ES)."""
    
    def __init__(self, config: EvolutionConfig):
        """Initialize the CMA-ES strategy."""
        super().__init__(config)
        self.sigma = 0.5  # Step size
        self.mean = None  # Mean of the distribution
        self.C = None  # Covariance matrix
        self.pc = None  # Evolution path for C
        self.ps = None  # Evolution path for sigma
        self.eigenvalues = None
        self.eigenvectors = None
        self.dim = None  # Dimension of the search space
    
    def initialize_population(self, model_fn, *args, **kwargs):
        """Initialize the population and CMA-ES parameters."""
        # Create a sample model to get the number of parameters
        sample_model = model_fn(*args, **kwargs)
        weights = sample_model.get_weights()
        self.dim = sum(w.size for w in weights)
        
        # Initialize mean (flattened weights of the sample model)
        self.mean = self._flatten_weights(weights)
        
        # Initialize covariance matrix as identity
        self.C = np.eye(self.dim)
        
        # Initialize evolution paths
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        
        # Strategy parameter setting: Selection
        self.mu = self.config.population_size // 2  # Number of parents/points for recombination
        self.lam = self.config.population_size  # Number of offspring
        
        # Recombination weights
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / np.sum(self.weights)
        self.mueff = 1.0 / np.sum(self.weights ** 2)  # Variance effective selection mass
        
        # Adaptation parameters
        self.cc = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mueff)
        cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / 
                 ((self.dim + 2) ** 2 + self.mueff))
        self.cmu = cmu
        
        # Initialize population
        self.population = []
        self.offspring = []
        
        # Damping for sigma
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs
        
        # Call parent's initialize_population to create initial models
        super().initialize_population(model_fn, *args, **kwargs)
    
    def _flatten_weights(self, weights):
        """Flatten model weights into a single vector."""
        return np.concatenate([w.flatten() for w in weights])
    
    def _unflatten_weights(self, flat_weights, template_weights):
        """Unflatten vector back into model weights."""
        weights = []
        idx = 0
        for w in template_weights:
            size = np.prod(w.shape)
            weights.append(flat_weights[idx:idx + size].reshape(w.shape))
            idx += size
        return weights
    
    def evolve(self, evaluate_fn):
        """Perform one generation of CMA-ES."""
        # Generate and evaluate offspring
        self.offspring = []
        fitness_scores = []
        
        # Sample new population
        for _ in range(self.lam):
            # Sample from multivariate normal distribution
            z = np.random.standard_normal(self.dim)
            if self.eigenvectors is not None:
                # Use eigendecomposition for sampling
                D = np.diag(np.sqrt(self.eigenvalues))
                x = self.mean + self.sigma * np.dot(self.eigenvectors, np.dot(D, z))
            else:
                # Simple sampling (first generation)
                x = self.mean + self.sigma * z
            
            # Create model with new weights
            model = tf.keras.models.clone_model(self.population[0])
            weights = model.get_weights()
            new_weights = self._unflatten_weights(x, weights)
            model.set_weights(new_weights)
            
            self.offspring.append((x, model))
        
        # Evaluate offspring
        fitness_scores = [evaluate_fn(ind[1]) for ind in self.offspring]
        
        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        self.offspring = [self.offspring[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]
        
        # Update best solution
        if fitness_scores[0] > self.best_fitness:
            self.best_fitness = fitness_scores[0]
            self.best_solution = self.offspring[0][1]
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        
        # Update mean
        old_mean = self.mean.copy()
        x_weights = np.zeros(self.dim)
        for i in range(self.mu):
            x_weights += self.weights[i] * self.offspring[i][0]
        self.mean = x_weights
        
        # Update evolution paths
        y = (self.mean - old_mean) / self.sigma
        z = np.linalg.solve(self.C, y) if hasattr(np.linalg, 'solve') else np.linalg.lstsq(self.C, y, rcond=None)[0]
        
        # Update evolution path for sigma
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * z
        
        # Update evolution path for C
        hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (self.generation + 1))) < 1.4 + 2 / (self.dim + 1)
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y
        
        # Update covariance matrix
        C1 = np.outer(self.pc, self.pc)
        C2 = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            y = (self.offspring[i][0] - old_mean) / self.sigma
            C2 += self.weights[i] * np.outer(y, y)
        
        self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * C1 + self.cmu * C2
        
        # Update step size
        self.sigma = self.sigma * np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / np.linalg.norm(np.random.standard_normal(self.dim)) - 1))
        
        # Update population for next generation
        self.population = [ind[1] for ind in self.offspring]
        
        # Update generation counter
        self.generation += 1
        self.fitness_history.append(max(fitness_scores))
        
        # Check for stagnation
        if self.stagnation_counter >= self.config.max_stagnation and self.config.reset_on_stagnation:
            self._handle_stagnation()
        
        return max(fitness_scores)


def get_evolution_strategy(config: EvolutionConfig) -> EvolutionStrategyBase:
    """Factory function to get an evolution strategy instance."""
    if config.strategy == EvolutionStrategy.SIMPLE:
        return SimpleEvolution(config)
    elif config.strategy == EvolutionStrategy.COVARIANCE_MATRIX_ADAPTATION:
        return CMAES(config)
    else:
        raise ValueError(f"Unsupported evolution strategy: {config.strategy}")
