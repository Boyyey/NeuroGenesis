"""
Evolution Engine Module

Implements the core evolutionary algorithms for neural architecture search and optimization.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import tensorflow as tf
from dataclasses import dataclass, field
from enum import Enum
import random
from .controller import MetaController, EvolutionAction

class MutationType(Enum):
    """Types of mutations that can be applied to the network."""
    ADD_NEURON = auto()
    REMOVE_NEURON = auto()
    ADD_LAYER = auto()
    REMOVE_LAYER = auto()
    MODIFY_ACTIVATION = auto()
    ADJUST_LEARNING_RATE = auto()
    MODIFY_CONNECTIVITY = auto()

@dataclass
class EvolutionConfig:
    """Configuration parameters for the evolution process."""
    # Population parameters
    population_size: int = 10
    elite_size: int = 2
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    
    # Architecture search space
    min_layers: int = 1
    max_layers: int = 8
    min_neurons: int = 16
    max_neurons: int = 1024
    neuron_step_size: int = 8
    
    # Training parameters
    initial_learning_rate: float = 1e-3
    min_learning_rate: float = 1e-6
    max_learning_rate: float = 1e-2
    
    # Evolution strategy
    max_generations: int = 50
    epochs_per_generation: int = 5
    early_stopping_rounds: int = 10
    
    # Mutation probabilities (should sum to 1.0)
    mutation_weights: Dict[MutationType, float] = field(
        default_factory=lambda: {
            MutationType.ADD_NEURON: 0.25,
            MutationType.REMOVE_NEURON: 0.2,
            MutationType.ADD_LAYER: 0.15,
            MutationType.REMOVE_LAYER: 0.1,
            MutationType.MODIFY_ACTIVATION: 0.1,
            MutationType.ADJUST_LEARNING_RATE: 0.1,
            MutationType.MODIFY_CONNECTIVITY: 0.1
        }
    )
    
    def __post_init__(self):
        # Normalize mutation weights
        total = sum(self.mutation_weights.values())
        self.mutation_weights = {k: v/total for k, v in self.mutation_weights.items()}

class EvolutionEngine:
    """
    Implements neuroevolution algorithms for evolving neural network architectures.
    """
    
    def __init__(self, config: Optional[EvolutionConfig] = None):
        """Initialize the evolution engine with configuration."""
        self.config = config if config is not None else EvolutionConfig()
        self.population = []
        self.generation = 0
        self.best_individual = None
        self.history = []
        
        # Initialize activation functions to choose from
        self.activation_functions = [
            'relu', 'leaky_relu', 'elu', 'selu', 'tanh', 'sigmoid'
        ]
    
    def initialize_population(self, input_shape: Tuple[int, ...], num_classes: int):
        """Initialize a population of neural networks with random architectures."""
        self.population = []
        
        for _ in range(self.config.population_size):
            # Random number of layers
            num_layers = random.randint(
                self.config.min_layers,
                min(self.config.max_layers, 4)  # Start with simpler networks
            )
            
            # Random neurons per layer
            neurons_per_layer = [
                random.randrange(
                    self.config.min_neurons,
                    self.config.max_neurons + 1,
                    self.config.neuron_step_size
                )
                for _ in range(num_layers)
            ]
            
            # Random activation function
            activation = random.choice(self.activation_functions)
            
            # Create and compile model
            model = self._create_model(input_shape, num_classes, neurons_per_layer, activation)
            
            # Add to population
            self.population.append({
                'model': model,
                'fitness': 0.0,
                'neurons_per_layer': neurons_per_layer,
                'num_layers': num_layers,
                'activation': activation,
                'learning_rate': self.config.initial_learning_rate
            })
        
        self.generation = 0
        self._update_best_individual()
    
    def _create_model(self, input_shape, num_classes, neurons_per_layer, activation):
        """Helper to create a Keras model with the given architecture."""
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        
        # Add hidden layers
        for units in neurons_per_layer:
            x = tf.keras.layers.Dense(
                units=units,
                activation=activation,
                kernel_initializer='he_normal'
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.2)(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(
            num_classes,
            activation='softmax'
        )(x)
        
        # Create and compile model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.initial_learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def evaluate_population(self, x_train, y_train, x_val, y_val, epochs=5, batch_size=32):
        """Evaluate the fitness of each individual in the population."""
        for individual in self.population:
            model = individual['model']
            
            # Train the model
            history = model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )
            
            # Fitness is validation accuracy
            val_accuracy = history.history['val_accuracy'][-1]
            individual['fitness'] = val_accuracy
            individual['history'] = history.history
        
        # Sort population by fitness (descending)
        self.population.sort(key=lambda x: x['fitness'], reverse=True)
        self._update_best_individual()
        
        # Record generation statistics
        self._record_generation_stats()
        
        return self.population
    
    def evolve(self):
        """Evolve the population to the next generation."""
        if self.generation >= self.config.max_generations:
            return False
        
        # Keep the elite individuals
        elites = self.population[:self.config.elite_size]
        
        # Create new population starting with elites
        new_population = [elite.copy() for elite in elites]
        
        # Generate offspring through selection, crossover, and mutation
        while len(new_population) < self.config.population_size:
            # Selection (tournament selection)
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if random.random() < self.config.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            if random.random() < self.config.mutation_rate:
                child1 = self._mutate(child1)
            if random.random() < self.config.mutation_rate:
                child2 = self._mutate(child2)
            
            # Add to new population if we have room
            if len(new_population) < self.config.population_size - 1:
                new_population.extend([child1, child2])
            else:
                new_population.append(child1)
        
        # Update population and generation
        self.population = new_population
        self.generation += 1
        
        return True
    
    def _tournament_selection(self, tournament_size=3):
        """Select an individual using tournament selection."""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x['fitness'])
    
    def _crossover(self, parent1, parent2):
        """Perform crossover between two parent networks."""
        # Simple uniform crossover for architecture
        child1_neurons = []
        child2_neurons = []
        
        max_layers = max(len(parent1['neurons_per_layer']), len(parent2['neurons_per_layer']))
        
        for i in range(max_layers):
            if i < len(parent1['neurons_per_layer']) and i < len(parent2['neurons_per_layer']):
                # Both parents have this layer
                if random.random() < 0.5:
                    child1_neurons.append(parent1['neurons_per_layer'][i])
                    child2_neurons.append(parent2['neurons_per_layer'][i])
                else:
                    child1_neurons.append(parent2['neurons_per_layer'][i])
                    child2_neurons.append(parent1['neurons_per_layer'][i])
            elif i < len(parent1['neurons_per_layer']):
                # Only parent1 has this layer
                if random.random() < 0.5:  # 50% chance to keep or discard
                    child1_neurons.append(parent1['neurons_per_layer'][i])
            else:
                # Only parent2 has this layer
                if random.random() < 0.5:  # 50% chance to keep or discard
                    child2_neurons.append(parent2['neurons_per_layer'][i])
        
        # Ensure we have at least min_layers
        if len(child1_neurons) < self.config.min_layers:
            child1_neurons.extend([self.config.min_neurons] * 
                                (self.config.min_layers - len(child1_neurons)))
        if len(child2_neurons) < self.config.min_layers:
            child2_neurons.extend([self.config.min_neurons] * 
                                (self.config.min_layers - len(child2_neurons)))
        
        # Create child models
        input_shape = parent1['model'].input_shape[1:]
        num_classes = parent1['model'].output_shape[-1]
        
        # Choose activation from parents
        activation = random.choice([parent1['activation'], parent2['activation']])
        learning_rate = np.random.uniform(
            self.config.min_learning_rate,
            self.config.max_learning_rate
        )
        
        child1 = {
            'model': self._create_model(input_shape, num_classes, child1_neurons, activation),
            'fitness': 0.0,
            'neurons_per_layer': child1_neurons,
            'num_layers': len(child1_neurons),
            'activation': activation,
            'learning_rate': learning_rate
        }
        
        child2 = {
            'model': self._create_model(input_shape, num_classes, child2_neurons, activation),
            'fitness': 0.0,
            'neurons_per_layer': child2_neurons,
            'num_layers': len(child2_neurons),
            'activation': activation,
            'learning_rate': learning_rate
        }
        
        return child1, child2
    
    def _mutate(self, individual):
        """Apply a random mutation to an individual."""
        mutation_type = np.random.choice(
            list(self.config.mutation_weights.keys()),
            p=list(self.config.mutation_weights.values())
        )
        
        if mutation_type == MutationType.ADD_NEURON:
            # Add neurons to a random layer
            layer_idx = random.randint(0, len(individual['neurons_per_layer']) - 1)
            current_neurons = individual['neurons_per_layer'][layer_idx]
            if current_neurons < self.config.max_neurons:
                individual['neurons_per_layer'][layer_idx] = min(
                    current_neurons + self.config.neuron_step_size,
                    self.config.max_neurons
                )
        
        elif mutation_type == MutationType.REMOVE_NEURON:
            # Remove neurons from a random layer
            layer_idx = random.randint(0, len(individual['neurons_per_layer']) - 1)
            current_neurons = individual['neurons_per_layer'][layer_idx]
            if current_neurons > self.config.min_neurons:
                individual['neurons_per_layer'][layer_idx] = max(
                    current_neurons - self.config.neuron_step_size,
                    self.config.min_neurons
                )
        
        elif mutation_type == MutationType.ADD_LAYER:
            # Add a new layer if under max layers
            if len(individual['neurons_per_layer']) < self.config.max_layers:
                new_neurons = random.randrange(
                    self.config.min_neurons,
                    self.config.max_neurons + 1,
                    self.config.neuron_step_size
                )
                # Insert at random position
                pos = random.randint(0, len(individual['neurons_per_layer']))
                individual['neurons_per_layer'].insert(pos, new_neurons)
                individual['num_layers'] += 1
        
        elif mutation_type == MutationType.REMOVE_LAYER:
            # Remove a random layer if above min layers
            if len(individual['neurons_per_layer']) > self.config.min_layers:
                idx = random.randint(0, len(individual['neurons_per_layer']) - 1)
                individual['neurons_per_layer'].pop(idx)
                individual['num_layers'] -= 1
        
        elif mutation_type == MutationType.MODIFY_ACTIVATION:
            # Change activation function
            current_activation = individual['activation']
            new_activation = random.choice(
                [a for a in self.activation_functions if a != current_activation]
            )
            individual['activation'] = new_activation
        
        elif mutation_type == MutationType.ADJUST_LEARNING_RATE:
            # Adjust learning rate
            individual['learning_rate'] = np.clip(
                individual['learning_rate'] * np.random.uniform(0.5, 2.0),
                self.config.min_learning_rate,
                self.config.max_learning_rate
            )
        
        # Rebuild the model with the new architecture
        input_shape = individual['model'].input_shape[1:]
        num_classes = individual['model'].output_shape[-1]
        individual['model'] = self._create_model(
            input_shape,
            num_classes,
            individual['neurons_per_layer'],
            individual['activation']
        )
        
        return individual
    
    def _update_best_individual(self):
        """Update the best individual based on current population."""
        if self.population:
            current_best = max(self.population, key=lambda x: x['fitness'])
            if (self.best_individual is None or 
                current_best['fitness'] > self.best_individual['fitness']):
                self.best_individual = current_best.copy()
    
    def _record_generation_stats(self):
        """Record statistics for the current generation."""
        if not self.population:
            return
        
        fitnesses = [ind['fitness'] for ind in self.population]
        num_layers = [ind['num_layers'] for ind in self.population]
        num_neurons = [sum(ind['neurons_per_layer']) for ind in self.population]
        
        stats = {
            'generation': self.generation,
            'best_fitness': max(fitnesses),
            'avg_fitness': np.mean(fitnesses),
            'worst_fitness': min(fitnesses),
            'avg_layers': np.mean(num_layers),
            'avg_neurons': np.mean(num_neurons),
            'best_individual': self.best_individual.copy() if self.best_individual else None
        }
        
        self.history.append(stats)
    
    def get_best_model(self):
        """Get the best model found during evolution."""
        return self.best_individual['model'] if self.best_individual else None
    
    def get_evolution_history(self):
        """Get the history of evolution across generations."""
        return self.history
