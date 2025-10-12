"""
Neural Network Layers Module

Implements various types of neural network layers that can be used in the NeuroGenesis framework.
"""

import tensorflow as tf
from enum import Enum, auto
from typing import List, Dict, Any, Optional, Union, Callable

class LayerType(Enum):
    """Supported layer types in the NeuroGenesis framework."""
    DENSE = auto()
    CONV2D = auto()
    LSTM = auto()
    GRU = auto()
    ATTENTION = auto()
    BATCH_NORM = auto()
    DROPOUT = auto()
    LAYER_NORM = auto()
    RESIDUAL = auto()

class LayerFactory:
    """Factory class for creating different types of neural network layers."""
    
    @staticmethod
    def create_layer(
        layer_type: LayerType,
        units: Optional[int] = None,
        activation: Optional[Union[str, Callable]] = 'relu',
        **kwargs
    ) -> tf.keras.layers.Layer:
        """
        Create a neural network layer of the specified type.
        
        Args:
            layer_type: Type of layer to create
            units: Number of units/filters for the layer
            activation: Activation function to use
            **kwargs: Additional layer-specific arguments
            
        Returns:
            A Keras Layer instance
        """
        if layer_type == LayerType.DENSE:
            return tf.keras.layers.Dense(units=units, activation=activation, **kwargs)
            
        elif layer_type == LayerType.CONV2D:
            filters = units if units is not None else 32
            return tf.keras.layers.Conv2D(
                filters=filters, 
                kernel_size=kwargs.get('kernel_size', 3),
                activation=activation,
                padding=kwargs.get('padding', 'same'),
                **{k: v for k, v in kwargs.items() if k not in ['kernel_size', 'padding']}
            )
            
        elif layer_type == LayerType.LSTM:
            return tf.keras.layers.LSTM(
                units=units,
                activation=activation,
                return_sequences=kwargs.get('return_sequences', False),
                **{k: v for k, v in kwargs.items() if k != 'return_sequences'}
            )
            
        elif layer_type == LayerType.GRU:
            return tf.keras.layers.GRU(
                units=units,
                activation=activation,
                return_sequences=kwargs.get('return_sequences', False),
                **{k: v for k, v in kwargs.items() if k != 'return_sequences'}
            )
            
        elif layer_type == LayerType.ATTENTION:
            return MultiHeadAttention(
                num_heads=kwargs.get('num_heads', 4),
                key_dim=kwargs.get('key_dim', 64),
                **{k: v for k, v in kwargs.items() if k not in ['num_heads', 'key_dim']}
            )
            
        elif layer_type == LayerType.BATCH_NORM:
            return tf.keras.layers.BatchNormalization(**kwargs)
            
        elif layer_type == LayerType.LAYER_NORM:
            return tf.keras.layers.LayerNormalization(**kwargs)
            
        elif layer_type == LayerType.DROPOUT:
            rate = kwargs.get('rate', 0.2)
            return tf.keras.layers.Dropout(rate=rate)
            
        elif layer_type == LayerType.RESIDUAL:
            return ResidualBlock(
                units=units,
                activation=activation,
                **kwargs
            )
            
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head attention layer with residual connection and layer normalization."""
    
    def __init__(self, num_heads=4, key_dim=64, **kwargs):
        """
        Initialize the multi-head attention layer.
        
        Args:
            num_heads: Number of attention heads
            key_dim: Dimension of the key vectors
            **kwargs: Additional arguments passed to the parent class
        """
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        
    def build(self, input_shape):
        """Build the layer's weights."""
        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            value_dim=self.key_dim
        )
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()
        super().build(input_shape)
    
    def call(self, inputs, training=False):
        """Forward pass for the layer."""
        attn_output = self.multi_head_attention(
            query=inputs,
            value=inputs,
            key=inputs,
            training=training
        )
        # Add residual connection and layer normalization
        x = self.add([inputs, attn_output])
        return self.layer_norm(x, training=training)
    
    def get_config(self):
        """Get the layer configuration."""
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim
        })
        return config


class ResidualBlock(tf.keras.layers.Layer):
    """A residual block with optional projection shortcut."""
    
    def __init__(self, units, activation='relu', use_conv_shortcut=False, **kwargs):
        """
        Initialize the residual block.
        
        Args:
            units: Number of units in the dense layers
            activation: Activation function to use
            use_conv_shortcut: Whether to use a convolutional shortcut
            **kwargs: Additional arguments passed to the parent class
        """
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_conv_shortcut = use_conv_shortcut
        
    def build(self, input_shape):
        """Build the layer's weights."""
        input_dim = input_shape[-1]
        
        # Main branch
        self.dense1 = tf.keras.layers.Dense(self.units, activation=self.activation)
        self.dense2 = tf.keras.layers.Dense(input_dim)
        self.layer_norm1 = tf.keras.layers.LayerNormalization()
        self.layer_norm2 = tf.keras.layers.LayerNormalization()
        
        # Shortcut connection
        if self.use_conv_shortcut or input_dim != self.units:
            self.shortcut = tf.keras.layers.Dense(self.units)
        else:
            self.shortcut = tf.keras.layers.Lambda(lambda x: x)
            
        super().build(input_shape)
    
    def call(self, inputs, training=False):
        """Forward pass for the layer."""
        # Main branch
        x = self.layer_norm1(inputs, training=training)
        x = self.activation(x)
        x = self.dense1(x)
        
        x = self.layer_norm2(x, training=training)
        x = self.activation(x)
        x = self.dense2(x)
        
        # Shortcut connection
        shortcut = self.shortcut(inputs)
        
        return x + shortcut
    
    def get_config(self):
        """Get the layer configuration."""
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation': tf.keras.activations.serialize(self.activation),
            'use_conv_shortcut': self.use_conv_shortcut
        })
        return config
