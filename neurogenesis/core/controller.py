"""
Meta-Controller Module

Implements the controller that monitors and modifies the neural network architecture
during training based on performance metrics and learning dynamics.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import tensorflow as tf
from dataclasses import dataclass, field
from enum import Enum, auto
import math

class EvolutionAction(Enum):
    """Possible actions the meta-controller can take."""
    NONE = auto()
    ADD_NEURONS = auto()
    REMOVE_NEURONS = auto()
    ADD_LAYER = auto()
    REMOVE_LAYER = auto()
    ADJUST_LEARNING_RATE = auto()
    CHANGE_ACTIVATION = auto()

@dataclass
class TrainingState:
    """Tracks the state of training and model performance."""
    epoch: int = 0
    loss: float = float('inf')
    val_loss: float = float('inf')
    accuracy: float = 0.0
    val_accuracy: float = 0.0
    learning_rate: float = 0.001
    last_improvement: int = 0
    best_val_loss: float = float('inf')
    patience: int = 10
    min_delta: float = 1e-4
    history: List[Dict[str, Any]] = field(default_factory=list)
    
    def update(self, metrics: Dict[str, float]) -> None:
        """Update training state with new metrics."""
        self.epoch += 1
        self.loss = metrics.get('loss', self.loss)
        self.val_loss = metrics.get('val_loss', self.val_loss)
        self.accuracy = metrics.get('accuracy', self.accuracy)
        self.val_accuracy = metrics.get('val_accuracy', self.val_accuracy)
        
        # Track best validation loss
        if self.val_loss < (self.best_val_loss - self.min_delta):
            self.best_val_loss = self.val_loss
            self.last_improvement = self.epoch
        
        # Record history
        self.history.append({
            'epoch': self.epoch,
            'loss': self.loss,
            'val_loss': self.val_loss,
            'accuracy': self.accuracy,
            'val_accuracy': self.val_accuracy,
            'learning_rate': self.learning_rate
        })
    
    def should_evolve(self) -> bool:
        """Determine if the model should evolve based on training progress."""
        # If we haven't improved in 'patience' epochs, consider evolving
        return (self.epoch - self.last_improvement) >= self.patience
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        return {
            'loss': self.loss,
            'val_loss': self.val_loss,
            'accuracy': self.accuracy,
            'val_accuracy': self.val_accuracy,
            'learning_rate': self.learning_rate,
            'epoch': self.epoch
        }

class MetaController:
    """
    Meta-controller that monitors training and makes decisions about
    when and how to evolve the neural network architecture.
    """
    
    def __init__(self,
                 min_neurons: int = 32,
                 max_neurons: int = 1024,
                 max_layers: int = 10,
                 min_layers: int = 1,
                 growth_rate: float = 0.1,
                 max_growth_factor: float = 2.0,
                 min_growth_factor: float = 0.5,
                 patience: int = 5,
                 min_delta: float = 1e-4):
        """
        Initialize the meta-controller.
        
        Args:
            min_neurons: Minimum number of neurons in any layer
            max_neurons: Maximum number of neurons in any layer
            max_layers: Maximum number of hidden layers
            min_layers: Minimum number of hidden layers
            growth_rate: Rate at which to grow the network
            max_growth_factor: Maximum factor by which to grow a layer
            min_growth_factor: Minimum factor by which to shrink a layer
            patience: Number of epochs to wait before evolving
            min_delta: Minimum change to qualify as an improvement
        """
        self.min_neurons = min_neurons
        self.max_neurons = max_neurons
        self.max_layers = max_layers
        self.min_layers = min_layers
        self.growth_rate = growth_rate
        self.max_growth_factor = max_growth_factor
        self.min_growth_factor = min_growth_factor
        self.patience = patience
        self.min_delta = min_delta
        self.state = TrainingState(patience=patience, min_delta=min_delta)
        self.evolution_history = []
    
    def analyze_performance(self, model, x_batch: tf.Tensor) -> EvolutionAction:
        """
        Analyze model performance and decide on evolution action.
        
        Args:
            model: The NeuralLearner instance
            x_batch: A batch of input data for analysis
            
        Returns:
            An EvolutionAction indicating what action to take
        """
        # Check if we should evolve based on training progress
        if not self.state.should_evolve():
            return EvolutionAction.NONE
        
        # Get model architecture info
        arch = model.get_architecture()
        num_layers = arch['num_layers']
        neurons_per_layer = arch['neurons_per_layer']
        
        # Calculate layer importance
        importance_scores = model.get_layer_importance(x_batch)
        
        # If we have too few layers, consider adding one
        if num_layers < self.min_layers:
            return EvolutionAction.ADD_LAYER
        
        # If we have too many layers, consider removing the least important one
        if num_layers > self.max_layers:
            least_important = np.argmin(importance_scores)
            return EvolutionAction.REMOVE_LAYER
        
        # Check for layers that might need more capacity
        for i, (neurons, importance) in enumerate(zip(neurons_per_layer, importance_scores)):
            # If layer is important but small, consider growing it
            if (importance > np.median(importance_scores) * 1.5 and 
                neurons < self.max_neurons):
                return EvolutionAction.ADD_NEURONS
            
            # If layer is large but not very important, consider shrinking it
            if (importance < np.median(importance_scores) * 0.5 and 
                neurons > self.min_neurons):
                return EvolutionAction.REMOVE_NEURONS
        
        # If we're stuck but don't need to modify architecture, adjust learning rate
        if self.state.last_improvement > self.patience // 2:
            return EvolutionAction.ADJUST_LEARNING_RATE
        
        return EvolutionAction.NONE
    
    def apply_evolution(self, 
                       model, 
                       action: EvolutionAction, 
                       x_batch: tf.Tensor) -> Dict[str, Any]:
        """
        Apply the specified evolution action to the model.
        
        Args:
            model: The NeuralLearner instance
            action: The evolution action to take
            x_batch: A batch of input data for analysis
            
        Returns:
            A dictionary containing information about the evolution
        """
        evolution_info = {
            'action': action.name,
            'timestamp': tf.timestamp().numpy(),
            'before_arch': model.get_architecture()
        }
        
        if action == EvolutionAction.ADD_NEURONS:
            self._add_neurons(model, x_batch, evolution_info)
        elif action == EvolutionAction.REMOVE_NEURONS:
            self._remove_neurons(model, x_batch, evolution_info)
        elif action == EvolutionAction.ADD_LAYER:
            self._add_layer(model, evolution_info)
        elif action == EvolutionAction.REMOVE_LAYER:
            self._remove_layer(model, evolution_info)
        elif action == EvolutionAction.ADJUST_LEARNING_RATE:
            self._adjust_learning_rate(model, evolution_info)
        
        evolution_info['after_arch'] = model.get_architecture()
        self.evolution_history.append(evolution_info)
        self.state.last_improvement = 0  # Reset patience
        
        return evolution_info
    
    def _add_neurons(self, model, x_batch, evolution_info: Dict[str, Any]) -> None:
        """Add neurons to the most important layer."""
        importance_scores = model.get_layer_importance(x_batch)
        layer_idx = np.argmax(importance_scores)
        current_neurons = model.layers_.layers[layer_idx].layers[0].units
        new_neurons = min(
            int(current_neurons * (1.0 + self.growth_rate)),
            self.max_neurons
        )
        model.grow_neuron(layer_idx, new_neurons - current_neurons)
        evolution_info['details'] = {
            'layer': layer_idx,
            'neurons_added': new_neurons - current_neurons,
            'new_size': new_neurons
        }
    
    def _remove_neurons(self, model, x_batch, evolution_info: Dict[str, Any]) -> None:
        """Remove neurons from the least important layer."""
        importance_scores = model.get_layer_importance(x_batch)
        layer_idx = np.argmin(importance_scores)
        current_neurons = model.layers_.layers[layer_idx].layers[0].units
        neurons_to_remove = max(
            int(current_neurons * (1.0 - self.growth_rate * 0.5)),
            self.min_neurons
        )
        neurons_to_remove = min(neurons_to_remove, current_neurons - self.min_neurons)
        
        if neurons_to_remove > 0:
            # In a real implementation, we would identify which neurons to remove
            # For now, we'll just remove from the end
            remove_indices = list(range(current_neurons - neurons_to_remove, current_neurons))
            model.prune_neurons(layer_idx, remove_indices)
            
            evolution_info['details'] = {
                'layer': layer_idx,
                'neurons_removed': neurons_to_remove,
                'new_size': current_neurons - neurons_to_remove
            }
    
    def _add_layer(self, model, evolution_info: Dict[str, Any]) -> None:
        """Add a new layer to the model."""
        arch = model.get_architecture()
        if arch['num_layers'] >= self.max_layers:
            return
            
        # Add a new layer with the average size of existing layers
        avg_neurons = int(np.mean(arch['neurons_per_layer'])) if arch['neurons_per_layer'] else 64
        new_neurons = max(min(avg_neurons, self.max_neurons), self.min_neurons)
        
        # Add the new layer before the output layer
        model._add_dense_layer(new_neurons, -1)
        
        evolution_info['details'] = {
            'position': -1,
            'neurons': new_neurons
        }
    
    def _remove_layer(self, model, evolution_info: Dict[str, Any]) -> None:
        """Remove the least important layer from the model."""
        arch = model.get_architecture()
        if arch['num_layers'] <= self.min_layers:
            return
            
        # Remove the smallest layer (simplified heuristic)
        min_neurons = min(arch['neurons_per_layer'])
        layer_to_remove = arch['neurons_per_layer'].index(min_neurons)
        model.remove_layer(layer_to_remove)
        
        evolution_info['details'] = {
            'position': layer_to_remove,
            'neurons_removed': min_neurons
        }
    
    def _adjust_learning_rate(self, model, evolution_info: Dict[str, Any]) -> None:
        """Adjust the learning rate of the optimizer."""
        if not hasattr(model.optimizer, 'learning_rate'):
            return
            
        old_lr = float(model.optimizer.learning_rate)
        new_lr = old_lr * 0.5  # Reduce learning rate
        model.optimizer.learning_rate.assign(new_lr)
        self.state.learning_rate = new_lr
        
        evolution_info['details'] = {
            'old_learning_rate': old_lr,
            'new_learning_rate': new_lr
        }
    
    def update_state(self, metrics: Dict[str, float]) -> None:
        """Update the controller's state with new metrics."""
        self.state.update(metrics)
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get a summary of all evolution steps taken."""
        return {
            'total_evolutions': len(self.evolution_history),
            'evolution_history': self.evolution_history,
            'current_architecture': self.evolution_history[-1]['after_arch'] if self.evolution_history else {}
        }
