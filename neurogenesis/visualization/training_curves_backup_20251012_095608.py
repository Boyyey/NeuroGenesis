"""
Interactive 3D Training Visualization Module

This module provides tools for visualizing training metrics in 3D,
including loss surfaces, metric trajectories, and hyperparameter optimization results.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

class TrainingVisualizer:
    """Class for creating interactive 3D visualizations of training metrics."""
    
    def __init__(self, history: Optional[Dict] = None):
        """Initialize the TrainingVisualizer with optional training history.
        
        Args:
            history: Dictionary containing training history (e.g., from model.fit())
        """
        self.history = history or {}
        self.figures = {}
    
    def plot_3d_loss_surface(self, model, X: np.ndarray, y: np.ndarray, 
                           weight_directions: Tuple[np.ndarray, np.ndarray],
                           n_points: int = 20, alpha: float = 1.0) -> go.Figure:
        """Create a 3D surface plot of the loss function.
        
        Args:
            model: Trained model to evaluate
            X: Input features
            y: Target values
            weight_directions: Tuple of two weight vectors for the surface axes
            n_points: Number of points along each axis
            alpha: Scale factor for the weight directions
            
        Returns:
            plotly.graph_objects.Figure: 3D surface plot
        """
        # Create grid of points in weight space
        w1, w2 = weight_directions
        w1 = w1.astype(np.float32)
        w2 = w2.astype(np.float32)
        
        # Create grid of coordinates
        coords = np.linspace(-alpha, alpha, n_points)
        w1_grid, w2_grid = np.meshgrid(coords, coords)
        
        # Flatten for vectorized operations
        w1_flat = w1_grid.ravel()
        w2_flat = w2_grid.ravel()
        
        # Compute loss for each point on the grid
        losses = np.zeros_like(w1_flat)
        
        # Get current weights and create a copy of the model
        original_weights = [w.numpy() for w in model.weights]
        temp_weights = [w.numpy().copy() for w in model.weights]
        
        for i, (a, b) in enumerate(zip(w1_flat, w2_flat)):
            # Update weights along the two directions
            for j in range(len(temp_weights)):
                temp_weights[j] = original_weights[j] + a * w1[j] + b * w2[j]
            
            # Set weights and compute loss
            model.set_weights(temp_weights)
            loss = model.evaluate(X, y, verbose=0)
            losses[i] = loss[0] if isinstance(loss, list) else loss
        
        # Restore original weights
        model.set_weights(original_weights)
        
        # Reshape for surface plot
        loss_surface = losses.reshape(w1_grid.shape)
        
        # Create 3D surface plot
        fig = go.Figure(data=[go.Surface(
            z=loss_surface,
            x=w1_grid,
            y=w2_grid,
            colorscale='Viridis',
            opacity=0.8,
            contours={
                "z": {"show": True, "start": loss_surface.min(), "end": loss_surface.max(), 
                      "size": (loss_surface.max() - loss_surface.min()) / 10}
            },
            hovertemplate='<b>Direction 1</b>: %{x:.4f}<br>'
                        '<b>Direction 2</b>: %{y:.4f}<br>'
                        '<b>Loss</b>: %{z:.4f}<extra></extra>'
        )])
        
        fig.update_layout(
            title='3D Loss Surface',
            scene=dict(
                xaxis_title='Weight Direction 1',
                yaxis_title='Weight Direction 2',
                zaxis_title='Loss',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            height=600
        )
        
        self.figures['loss_surface'] = fig
        return fig
    
    def plot_metric_evolution_3d(self, metrics: List[str] = None, title: str = None) -> go.Figure:
        """Create a 3D plot of metric evolution over training.
        
        Args:
            metrics: List of metrics to plot (must be in history). If None, uses all available.
            title: Plot title
            
        Returns:
            plotly.graph_objects.Figure: 3D line plot
        """
        if not self.history:
            raise ValueError("No training history provided. Initialize with history or call add_history().")
        
        # Get available metrics
        available_metrics = [k for k in self.history.keys() if not k.startswith('val_') and k != 'lr']
        
        if not metrics:
            metrics = available_metrics[:3]  # Default to first 3 metrics
        else:
            # Filter to only include metrics that exist in history
            metrics = [m for m in metrics if m in available_metrics]
            if len(metrics) < 2:
                raise ValueError(f"At least 2 metrics are required. Available metrics: {available_metrics}")
        
        # Get data
        epochs = np.arange(1, len(self.history[metrics[0]]) + 1)
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        # Add training metrics trajectory
        fig.add_trace(go.Scatter3d(
            x=epochs,
            y=self.history[metrics[0]],
            z=self.history[metrics[1 % len(metrics)]],
            mode='lines+markers',
            name='Training',
            line=dict(color='blue', width=4),
            marker=dict(size=4, opacity=0.8),
            hovertemplate='<b>Epoch</b>: %{x}<br>'
                        f'<b>{metrics[0]}</b>: %{{y:.4f}<br>'
                        f'<b>{metrics[1 % len(metrics)]}</b>: %{{z:.4f}<extra></extra>'
        ))
        
        # Add validation metrics if available
        val_metrics = [f'val_{m}' for m in metrics if f'val_{m}' in self.history]
        if val_metrics:
            fig.add_trace(go.Scatter3d(
                x=epochs,
                y=self.history.get(f'val_{metrics[0]}', []),
                z=self.history.get(f'val_{metrics[1 % len(metrics)]}', []),
                mode='lines+markers',
                name='Validation',
                line=dict(color='red', width=4, dash='dash'),
                marker=dict(size=4, opacity=0.8),
                hovertemplate='<b>Epoch</b>: %{x}<br>'
                            f'<b>Val {metrics[0]}</b>: %{{y:.4f}<br>'
                            f'<b>Val {metrics[1 % len(metrics)]}</b>: %{{z:.4f}<extra></extra>'
             ))
        
        # Add start and end markers
        for i, (x, y, z) in enumerate(zip(
            [epochs[0], epochs[-1]],
            [self.history[metrics[0]][0], self.history[metrics[0]][-1]],
            [self.history[metrics[1 % len(metrics)]][0], 
             self.history[metrics[1 % len(metrics)]][-1]]
        )):
            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode='markers+text',
                marker=dict(size=8, color='green' if i == 0 else 'red'),
                text=['Start' if i == 0 else 'End'],
                textposition='top center',
                showlegend=False,
                hoverinfo='text',
                hovertext=f'<b>Epoch</b>: {x}<br>'
                         f'<b>{metrics[0]}</b>: {y:.4f}<br>'
                         f'<b>{metrics[1 % len(metrics)]}</b>: {z:.4f}'
            ))
        
        # Update layout
        title = title or f'3D Training Trajectory: {metrics[0]} vs {metrics[1 % len(metrics)]}'
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Epoch',
                yaxis_title=metrics[0],
                zaxis_title=metrics[1 % len(metrics)],
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            height=700,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        self.figures['metric_evolution'] = fig
        return fig
    
    def plot_hyperparameter_sensitivity(self, param_name: str, param_values: List[float], 
                                      metric_values: List[float], metric_name: str = 'Loss',
                                      log_scale: bool = False) -> go.Figure:
        """Create a 3D surface plot showing model sensitivity to hyperparameters.
        
        Args:
            param_name: Name of the hyperparameter
            param_values: List of parameter values
            metric_values: Corresponding metric values
            metric_name: Name of the metric being plotted
            log_scale: Whether to use log scale for the parameter axis
            
        Returns:
            plotly.graph_objects.Figure: 3D surface plot
        """
        if len(param_values) != len(metric_values):
            raise ValueError("param_values and metric_values must have the same length")
        
        # Create a grid for the surface
        x = np.array(param_values)
        y = np.arange(len(metric_values[0]))  # Epochs
        X, Y = np.meshgrid(x, y)
        
        # Convert metric_values to numpy array and transpose to match meshgrid
        Z = np.array(metric_values).T
        
        # Create 3D surface plot
        fig = go.Figure(data=[go.Surface(
            z=Z,
            x=X,
            y=Y,
            colorscale='Viridis',
            opacity=0.8,
            contours={
                "z": {"show": True, "start": Z.min(), "end": Z.max(), 
                      "size": (Z.max() - Z.min()) / 10}
            },
            hovertemplate=f'<b>{param_name}</b>: %{{x:.4f}<br>'
                         f'<b>Epoch</b>: %{{y:.0f}<br>'
                         f'<b>{metric_name}</b>: %{{z:.4f}<extra></extra>'
        )])
        
        # Add a scatter plot on top to highlight the points
        for i, (param_val, metrics) in enumerate(zip(param_values, metric_values)):
            fig.add_trace(go.Scatter3d(
                x=[param_val] * len(metrics),
                y=list(range(len(metrics))),
                z=metrics,
                mode='lines+markers',
                line=dict(color='white', width=2),
                marker=dict(size=3, color='red'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Update layout
        fig.update_layout(
            title=f'Hyperparameter Sensitivity: {param_name} vs {metric_name}',
            scene=dict(
                xaxis_title=param_name + (' (log scale)' if log_scale else ''),
                yaxis_title='Epoch',
                zaxis_title=metric_name,
                xaxis_type='log' if log_scale else None,
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            height=700
        )
        
        self.figures['hyperparameter_sensitivity'] = fig
        return fig
    
    def plot_attention_evolution_3d(self, attention_weights: List[np.ndarray], 
                                  layer_idx: int = 0, head_idx: int = 0,
                                  title: str = None) -> go.Figure:
        """Create a 3D visualization of attention weight evolution over time.
        
        Args:
            attention_weights: List of attention weight matrices over time
            layer_idx: Index of the attention layer to visualize
            head_idx: Index of the attention head to visualize
            title: Plot title
            
        Returns:
            plotly.graph_objects.Figure: 3D surface plot of attention weights
        """
        if not attention_weights:
            raise ValueError("No attention weights provided")
            
        # Get the attention weights for the specified layer and head
        attn_weights = []
        for weight in attention_weights:
            if isinstance(weight, list):
                # Handle multi-layer attention
                layer_weights = weight[layer_idx] if layer_idx < len(weight) else weight[-1]
                # Handle multi-head attention
                if len(layer_weights.shape) == 4:  # [batch, heads, seq_len, seq_len]
                    head_weights = layer_weights[0, head_idx % layer_weights.shape[1]]
                else:  # [batch, seq_len, seq_len]
                    head_weights = layer_weights[0]
            else:
                # Single attention matrix
                head_weights = weight[0]  # Remove batch dimension
            attn_weights.append(head_weights.numpy() if tf.is_tensor(head_weights) else head_weights)
        
        # Convert to numpy array
        attn_weights = np.array(attn_weights)
        
        # Create time dimension
        time = np.arange(attn_weights.shape[0])
        seq_len = attn_weights.shape[-1]
        
        # Create meshgrid for 3D surface
        X, Y = np.meshgrid(np.arange(seq_len), np.arange(seq_len))
        
        # Create figure with frames for animation
        fig = go.Figure(
            data=[go.Surface(
                z=attn_weights[0],
                x=X[0],
                y=Y[0],
                colorscale='Viridis',
                cmin=0,
                cmax=1,
                opacity=0.9,
                colorbar=dict(title='Attention Weight'),
                hovertemplate='<b>Query</b>: %{x}<br>'
                            '<b>Key</b>: %{y}<br>'
                            '<b>Weight</b>: %{z:.4f}<extra></extra>'
            )],
            layout=go.Layout(
                title=title or f'Attention Weights Evolution (Layer {layer_idx}, Head {head_idx})',
                scene=dict(
                    xaxis_title='Query Position',
                    yaxis_title='Key Position',
                    zaxis_title='Attention Weight',
                    zaxis=dict(range=[0, 1]),
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=0.8)
                    )
                ),
                margin=dict(l=0, r=0, b=0, t=40),
                height=700,
                updatemenus=[{
                    'type': 'buttons',
                    'buttons': [
                        {
                            'args': [None, {'frame': {'duration': 100, 'redraw': True}, 
                                          'fromcurrent': True, 'transition': {'duration': 0}}],
                            'label': 'Play',
                            'method': 'animate'
                        },
                        {
                            'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 
                                            'mode': 'immediate',
                                            'transition': {'duration': 0}}],
                            'label': 'Pause',
                            'method': 'animate'
                        }
                    ],
                    'direction': 'left',
                    'pad': {'r': 10, 't': 87},
                    'showactive': False,
                    'type': 'buttons',
                    'x': 0.1,
                    'xanchor': 'right',
                    'y': 0,
                    'yanchor': 'top'
                }]
            ),
            frames=[
                go.Frame(
                    data=[go.Surface(
                        z=attn_weights[i],
                        x=X[i % X.shape[0]],
                        y=Y[i % Y.shape[0]]
                    )],
                    name=f'frame_{i}'
                )
                for i in range(attn_weights.shape[0])
            ]
        )
        
        # Add slider
        fig.update_layout(
            sliders=[{
                'active': 0,
                'yanchor': 'top',
                'xanchor': 'left',
                'currentvalue': {
                    'font': {'size': 20},
                    'prefix': 'Step:',
                    'visible': True,
                    'xanchor': 'right'
                },
                'transition': {'duration': 300, 'easing': 'cubic-in-out'},
                'pad': {'b': 10, 't': 50},
                'len': 0.9,
                'x': 0.1,
                'y': 0,
                'steps': [
                    {
                        'args': [
                            [f'frame_{i}'],
                            {'frame': {'duration': 300, 'redraw': True},
                             'mode': 'immediate',
                             'transition': {'duration': 0}}
                        ],
                        'label': str(i),
                        'method': 'animate'
                    }
                    for i in range(attn_weights.shape[0])
                ]
            }]
        )
        
        self.figures['attention_evolution'] = fig
        return fig
    
    def add_history(self, history: Dict):
        """Add or update training history.
        
        Args:
            history: Dictionary containing training history (e.g., from model.fit())
        """
        self.history = history
    
    def save_figure(self, name: str, filename: str, format: str = 'html', 
                   width: int = None, height: int = None):
        """Save a figure to a file.
        
        Args:
            name: Name of the figure to save (key in self.figures)
            filename: Output filename (without extension)
            format: Output format ('html', 'png', 'jpeg', 'webp', 'svg')
            width: Image width in pixels
            height: Image height in pixels
        """
        if name not in self.figures:
            raise ValueError(f"Figure '{name}' not found. Available figures: {list(self.figures.keys())}")
        
        fig = self.figures[name]
        
        if format == 'html':
            fig.write_html(f"{filename}.html")
        else:
            fig.write_image(f"{filename}.{format}", width=width, height=height)
    
    def show(self, name: str = None):
        """Display a figure in the notebook.
        
        Args:
            name: Name of the figure to display. If None, displays all figures.
        """
        if name:
            if name not in self.figures:
                raise ValueError(f"Figure '{name}' not found. Available figures: {list(self.figures.keys())}")
            return self.figures[name].show()
        else:
            for fig_name, fig in self.figures.items():
                print(f"\n--- {fig_name.upper()} ---")
                fig.show()

# Example usage
if __name__ == "__main__":
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    
    # Create a simple model
    model = Sequential([
        Dense(10, activation='relu', input_shape=(5,)),
        Dense(1, activation='sigmoid')
    ])
    
    # Generate some random data
    X = np.random.randn(100, 5)
    y = (X.sum(axis=1) > 0).astype(int)
    
    # Train the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X, y, epochs=10, validation_split=0.2, verbose=0).history
    
    # Create visualizations
    visualizer = TrainingVisualizer(history)
    
    # Plot 3D metric evolution
    fig1 = visualizer.plot_metric_evolution_3d(['loss', 'accuracy'])
    
    # Create some dummy hyperparameter sensitivity data
    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1]
    losses = [
        [0.9 - 0.08 * i for i in range(10)],  # Loss for lr=1e-4
        [0.8 - 0.07 * i for i in range(10)],  # Loss for lr=1e-3
        [0.7 - 0.06 * i for i in range(10)],  # Loss for lr=1e-2
        [0.6 + 0.1 * i for i in range(10)]    # Loss for lr=1e-1 (diverging)
    ]
    
    # Plot hyperparameter sensitivity
    fig2 = visualizer.plot_hyperparameter_sensitivity(
        'Learning Rate', learning_rates, losses, 'Loss', log_scale=True
    )
    
    # Create a simple attention evolution example
    seq_len = 10
    time_steps = 20
    attention_weights = []
    
    for t in range(time_steps):
        # Create a simple attention pattern that changes over time
        attn = np.eye(seq_len) * 0.1
        focus_pos = t % seq_len
        attn[focus_pos, :] = 0.9
        attn[:, focus_pos] = 0.9
        attn = attn / attn.sum(axis=1, keepdims=True)
        attention_weights.append(attn.reshape(1, seq_len, seq_len))
    
    # Plot attention evolution
    fig3 = visualizer.plot_attention_evolution_3d(attention_weights)
    
    # Show all figures
    visualizer.show()
