"""
Neural Learner Module

Implements the base neural network that will be evolved by the meta-controller.
"""

import tensorflow as tf
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

class NeuralLearner(tf.keras.Model):
    """
    A flexible neural network that can be dynamically modified during training.
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, ...],
                 num_classes: int,
                 initial_neurons: List[int] = [128, 64],
                 activation: str = 'relu',
                 dropout_rate: float = 0.2):
        """
        Initialize the neural learner.
        
        Args:
            input_shape: Shape of the input data (excluding batch dimension)
            num_classes: Number of output classes
            initial_neurons: List of integers representing neurons per layer
            activation: Activation function to use in hidden layers
            dropout_rate: Dropout rate for regularization
        """
        super(NeuralLearner, self).__init__()
        
        self.input_shape_ = input_shape
        self.num_classes = num_classes
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.performance_history = []
        self.architecture_history = []
        
        # Build the initial model
        self.layers_ = tf.keras.Sequential()
        self.layers_.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        
        # Add initial hidden layers
        for neurons in initial_neurons:
            self._add_dense_layer(neurons)
        
        # Output layer
        self.output_layer = tf.keras.layers.Dense(
            num_classes, 
            activation='softmax'
        )
    
    def _add_dense_layer(self, units: int, index: Optional[int] = None) -> None:
        """Add a dense layer to the network."""
        new_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(
                units,
                activation=self.activation,
                kernel_initializer='he_normal'
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(self.dropout_rate)
        ])
        
        if index is None:
            self.layers_.add(new_layer)
        else:
            # Insert at specific position
            self.layers_.layers.insert(index, new_layer)
    
    def remove_layer(self, index: int) -> None:
        """Remove a layer from the network."""
        if 0 <= index < len(self.layers_.layers) - 1:  # Don't remove input layer
            self.layers_.layers.pop(index)
    
    def call(self, inputs, training=False):
        """Forward pass through the network."""
        x = self.layers_(inputs, training=training)
        return self.output_layer(x)
    
    def get_architecture(self) -> Dict[str, Any]:
        """Get current network architecture as a dictionary."""
        return {
            'num_layers': len(self.layers_.layers) - 1,  # Exclude input layer
            'neurons_per_layer': [layer.units for layer in self.layers_.layers[1:]],
            'total_parameters': self.count_params(),
            'activation': self.activation
        }
    
    def record_performance(self, metrics: Dict[str, float]) -> None:
        """Record performance metrics for the current architecture."""
        self.performance_history.append({
            'metrics': metrics,
            'architecture': self.get_architecture(),
            'timestamp': tf.timestamp()
        })
    
    def grow_neuron(self, layer_idx: int, num_neurons: int = 1) -> None:
        """Add neurons to a specific layer."""
        if 0 <= layer_idx < len(self.layers_.layers):
            current_units = self.layers_.layers[layer_idx].layers[0].units
            self.layers_.layers[layer_idx].layers[0].units += num_neurons
            # Rebuild layer to apply changes
            self.layers_.layers[layer_idx].layers[0].build(
                self.layers_.layers[layer_idx-1].output_shape
            )
    
    def prune_neurons(self, layer_idx: int, neuron_indices: List[int]) -> None:
        """Prune specific neurons from a layer."""
        if 0 <= layer_idx < len(self.layers_.layers):
            layer = self.layers_.layers[layer_idx].layers[0]
            # Create mask to keep only non-pruned neurons
            mask = np.ones(layer.units, dtype=bool)
            mask[neuron_indices] = False
            
            # Get weights and apply mask
            weights, biases = layer.get_weights()
            new_weights = weights[:, mask]
            new_biases = biases[mask]
            
            # Update layer units and set new weights
            layer.units = len(new_biases)
            layer.set_weights([new_weights, new_biases])
    
    def get_layer_importance(self, x_batch: tf.Tensor) -> List[float]:
        """
        Estimate the importance of each layer using gradient-based method.
        Returns a list of importance scores for each layer.
        """
        if len(x_batch.shape) == 1:
            x_batch = tf.expand_dims(x_batch, 0)
            
        with tf.GradientTape() as tape:
            tape.watch(x_batch)
            outputs = self(x_batch, training=False)
            output_magnitude = tf.reduce_sum(tf.square(outputs))
            
        # Get gradients of output magnitude w.r.t. each layer's output
        layer_outputs = [x_batch]
        for layer in self.layers_.layers[1:]:  # Skip input layer
            layer_outputs.append(layer(layer_outputs[-1]))
            
        # Calculate importance as gradient magnitude
        importance_scores = []
        for i, out in enumerate(layer_outputs[1:], 1):  # Skip input
            with tf.GradientTape() as tape:
                tape.watch(out)
                output_magnitude = tf.reduce_sum(tf.square(self.layers_.layers[i:](out)))
            
            grad = tape.gradient(output_magnitude, out)
            importance = tf.reduce_mean(tf.abs(grad)).numpy()
            importance_scores.append(float(importance))
            
        return importance_scores
