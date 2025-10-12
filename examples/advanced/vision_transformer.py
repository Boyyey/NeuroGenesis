"""
Vision Transformer with Self-Evolving Architecture and Attention Visualization

This example demonstrates a self-evolving Vision Transformer that can adapt its architecture
during training and provides interactive attention visualization.
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, HTML

from neurogenesis.core import NeuralLearner, NetworkConfig, NetworkType
from neurogenesis.core.layers import LayerType
from neurogenesis.core.strategies import EvolutionConfig, get_evolution_strategy

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

class VisionTransformer(NeuralLearner):
    """Enhanced Vision Transformer with self-evolving capabilities."""
    
    def __init__(self, 
                 input_shape=(224, 224, 3),
                 num_classes=1000,
                 patch_size=16,
                 num_layers=12,
                 num_heads=12,
                 hidden_dim=768,
                 mlp_dim=3072,
                 dropout_rate=0.1):
        """Initialize the Vision Transformer."""
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        
        # Create network configuration
        config = NetworkConfig(
            network_type=NetworkType.TRANSFORMER,
            layer_types=[LayerType.ATTENTION] * num_layers,
            hidden_units=[hidden_dim] * num_layers,
            attention_heads=num_heads,
            dropout_rate=dropout_rate
        )
        
        super().__init__(
            input_shape=input_shape,
            num_classes=num_classes,
            config=config
        )
        
        # Store attention weights for visualization
        self.attention_weights = []
        
    def _create_patches(self, images):
        """Split images into patches."""
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patches = tf.reshape(
            patches,
            [batch_size, -1, self.patch_size * self.patch_size * 3]
        )
        return patches
    
    def call(self, inputs, training=False, return_attention=False):
        """Forward pass with attention visualization."""
        if return_attention:
            self.attention_weights = []
            
        # Create patches
        patches = self._create_patches(inputs)
        
        # Add positional embeddings
        positions = tf.range(start=0, limit=tf.shape(patches)[1], delta=1)
        position_embeddings = self.position_embeddings(positions)
        x = patches + position_embeddings
        
        # Transformer layers
        for i, layer in enumerate(self.transformer_layers):
            x, weights = layer(x, training=training, return_attention_weights=True)
            if return_attention:
                self.attention_weights.append(weights)
        
        # Classification head
        x = self.classification_head(x)
        
        if return_attention:
            return x, self.attention_weights
        return x
    
    def visualize_attention(self, image, layer_idx=0, head_idx=0):
        """Visualize attention maps for a given image and attention head."""
        if not self.attention_weights:
            _, _ = self.predict(np.expand_dims(image, axis=0), return_attention=True)
        
        # Get attention weights for the specified layer and head
        attention = self.attention_weights[layer_idx][0, head_idx].numpy()  # [num_patches, num_patches]
        
        # Create figure with subplots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Original Image', 'Attention Map', 'Attention Overlay'),
            column_widths=[0.3, 0.3, 0.4]
        )
        
        # Original image
        fig.add_trace(
            go.Image(z=image.astype('uint8')),
            row=1, col=1
        )
        
        # Attention map
        fig.add_trace(
            go.Heatmap(z=attention, colorscale='Viridis'),
            row=1, col=2
        )
        
        # Attention overlay
        overlay = self._create_attention_overlay(image, attention)
        fig.add_trace(
            go.Image(z=overlay),
            row=1, col=3
        )
        
        fig.update_layout(
            title=f'Attention Visualization - Layer {layer_idx}, Head {head_idx}',
            height=400,
            showlegend=False
        )
        
        return fig
    
    def _create_attention_overlay(self, image, attention):
        """Create an attention overlay on the original image."""
        # Resize attention to match image dimensions
        h, w = image.shape[:2]
        attention_resized = tf.image.resize(
            tf.expand_dims(attention, -1), 
            [h, w],
            method='bicubic'
        ).numpy()
        
        # Normalize and apply colormap
        attention_norm = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min())
        cmap = cm.get_cmap('jet')
        attention_colored = (cmap(attention_norm[..., 0])[..., :3] * 255).astype('uint8')
        
        # Blend with original image
        alpha = 0.5
        overlay = (image * (1 - alpha) + attention_colored * alpha).astype('uint8')
        return overlay

def train_vision_transformer():
    """Train a self-evolving Vision Transformer."""
    # Load and preprocess data (example with CIFAR-10 for demonstration)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Resize images to 224x224 for ViT
    x_train = tf.image.resize(x_train, [224, 224])
    x_test = tf.image.resize(x_test, [224, 224])
    
    # Normalize pixel values
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # Create model
    model = VisionTransformer(
        input_shape=(224, 224, 3),
        num_classes=10,
        patch_size=32,
        num_layers=6,
        num_heads=8,
        hidden_dim=256,
        mlp_dim=512,
        dropout_rate=0.1
    )
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
    ]
    
    # Train model
    history = model.fit(
        x_train, y_train,
        batch_size=32,
        epochs=50,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f'\nTest accuracy: {test_acc:.4f}')
    
    return model, history

if __name__ == "__main__":
    # Train the model
    model, history = train_vision_transformer()
    
    # Example of attention visualization (in Jupyter notebook)
    if 'get_ipython' in globals():
        # Get a test image
        (x_train, _), _ = tf.keras.datasets.cifar10.load_data()
        test_image = tf.image.resize(x_train[0], [224, 224]) / 255.0
        
        # Visualize attention
        fig = model.visualize_attention(test_image.numpy(), layer_idx=0, head_idx=0)
        fig.show()
