"""
Neural Network Architecture Visualization

Provides 3D visualizations of neural network architectures with layer-wise activations.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

class ArchitectureVisualizer:
    """Visualize neural network architectures in 3D with layer activations."""
    
    def __init__(self, model):
        """Initialize with a Keras or PyTorch model."""
        self.model = model
        self.layer_info = self._extract_layer_info()
    
    def _extract_layer_info(self):
        """Extract layer information from the model."""
        layer_info = []
        
        if hasattr(self.model, 'layers'):  # Keras model
            for i, layer in enumerate(self.model.layers):
                layer_type = layer.__class__.__name__
                output_shape = layer.output_shape
                
                # Count parameters
                trainable_params = np.sum([np.prod(w.shape) for w in layer.trainable_weights])
                non_trainable_params = np.sum([np.prod(w.shape) for w in layer.non_trainable_weights])
                
                layer_info.append({
                    'index': i,
                    'name': layer.name,
                    'type': layer_type,
                    'output_shape': output_shape,
                    'trainable_params': trainable_params,
                    'non_trainable_params': non_trainable_params,
                    'activation': getattr(layer, 'activation', None)
                })
        
        return layer_info
    
    def plot_3d_architecture(self, figsize=(12, 8), node_size=10, layer_spacing=2.0):
        """Create a 3D visualization of the neural network architecture."""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Colors for different layer types
        layer_colors = {
            'Dense': 'blue',
            'Conv2D': 'green',
            'MaxPooling2D': 'red',
            'Flatten': 'purple',
            'Dropout': 'orange',
            'BatchNormalization': 'brown',
            'LSTM': 'pink',
            'GRU': 'cyan',
            'Embedding': 'magenta',
            'default': 'gray'
        }
        
        # Plot each layer
        for i, layer in enumerate(self.layer_info):
            layer_type = layer['type']
            color = layer_colors.get(layer_type.split('.')[-1], layer_colors['default'])
            
            # Position of the layer
            x = i * layer_spacing
            
            # For each neuron in the layer
            if len(layer['output_shape']) > 1:  # For layers with spatial dimensions
                if len(layer['output_shape']) == 4:  # Conv layer
                    _, height, width, channels = layer['output_shape']
                    y = np.linspace(-height/2, height/2, height)
                    z = np.linspace(-width/2, width/2, width)
                    Y, Z = np.meshgrid(y, z)
                    
                    # Plot neurons as points
                    for c in range(min(3, channels)):  # Limit to first 3 channels for visualization
                        ax.scatter(
                            np.full_like(Y, x), 
                            Y.flatten(), 
                            Z.flatten(), 
                            s=node_size, 
                            c=color,
                            alpha=0.7 - (c * 0.2),
                            edgecolors='none'
                        )
                else:  # Dense or other layers
                    num_neurons = layer['output_shape'][1] if len(layer['output_shape']) > 1 else layer['output_shape'][0]
                    y = np.linspace(-1, 1, num_neurons)
                    z = np.zeros_like(y)
                    
                    ax.scatter(
                        np.full_like(y, x), 
                        y, 
                        z, 
                        s=node_size, 
                        c=color,
                        alpha=0.7,
                        edgecolors='none',
                        label=layer_type if i == 0 else ""
                    )
            
            # Add layer label
            ax.text(
                x, 0, 1.5, 
                f"{layer_type}\n{layer['output_shape']}", 
                fontsize=8,
                ha='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )
            
            # Connect to previous layer
            if i > 0:
                prev_layer = self.layer_info[i-1]
                if len(prev_layer['output_shape']) > 1 and len(layer['output_shape']) > 1:
                    # Simple connection for visualization
                    prev_neurons = min(5, prev_layer['output_shape'][1] if len(prev_layer['output_shape']) > 1 
                                      else prev_layer['output_shape'][0])
                    curr_neurons = min(5, layer['output_shape'][1] if len(layer['output_shape']) > 1 
                                      else layer['output_shape'][0])
                    
                    for j in range(prev_neurons):
                        for k in range(curr_neurons):
                            ax.plot(
                                [(i-1)*layer_spacing, x],
                                [j/(prev_neurons-1)*2-1 if prev_neurons > 1 else 0, 
                                 k/(curr_neurons-1)*2-1 if curr_neurons > 1 else 0],
                                [0, 0],
                                'gray',
                                alpha=0.2,
                                linewidth=0.5
                            )
        
        # Set labels and title
        ax.set_xlabel('Layers')
        ax.set_ylabel('Neurons')
        ax.set_zlabel('Channels')
        ax.set_title('3D Neural Network Architecture')
        
        # Set the view angle
        ax.view_init(elev=30, azim=45)
        
        # Add legend
        handles = []
        for layer_type, color in layer_colors.items():
            if layer_type != 'default':
                handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, markersize=10, 
                                        label=layer_type))
        
        ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return fig
    
    def plot_interactive_architecture(self):
        """Create an interactive 3D visualization of the architecture."""
        # Create a 3D scatter plot
        fig = go.Figure()
        
        # Colors for different layer types
        layer_colors = {
            'Dense': 'blue',
            'Conv2D': 'green',
            'MaxPooling2D': 'red',
            'Flatten': 'purple',
            'Dropout': 'orange',
            'BatchNormalization': 'brown',
            'LSTM': 'pink',
            'GRU': 'cyan',
            'Embedding': 'magenta',
            'default': 'gray'
        }
        
        # Track layer positions and connections
        layer_positions = []
        connections = []
        
        # Add layers to the plot
        for i, layer in enumerate(self.layer_info):
            layer_type = layer['type'].split('.')[-1]
            color = layer_colors.get(layer_type, layer_colors['default'])
            
            # Position of the layer
            x = i * 2.0
            
            # For each neuron in the layer
            if len(layer['output_shape']) > 1:
                if len(layer['output_shape']) == 4:  # Conv layer
                    _, height, width, channels = layer['output_shape']
                    y = np.linspace(-1, 1, height)
                    z = np.linspace(-1, 1, width)
                    Y, Z = np.meshgrid(y, z)
                    
                    # Limit number of points for performance
                    step = max(1, height // 5)
                    Y = Y[::step, ::step]
                    Z = Z[::step, ::step]
                    
                    # Add neurons as scatter points
                    fig.add_trace(go.Scatter3d(
                        x=np.full_like(Y, x).flatten(),
                        y=Y.flatten(),
                        z=Z.flatten(),
                        mode='markers',
                        marker=dict(
                            size=5,
                            color=color,
                            opacity=0.7,
                            line=dict(width=0.5, color='white')
                        ),
                        name=f"{layer_type} {i}",
                        text=f"{layer_type}<br>Shape: {layer['output_shape']}<br>Params: {layer['trainable_params']}",
                        hoverinfo='text',
                        showlegend=False
                    ))
                    
                    # Store layer position for connections
                    layer_positions.append({
                        'x': x,
                        'y': 0,
                        'z': 0,
                        'type': layer_type,
                        'shape': layer['output_shape']
                    })
                    
                else:  # Dense or other layers
                    num_neurons = layer['output_shape'][1] if len(layer['output_shape']) > 1 else layer['output_shape'][0]
                    y = np.linspace(-1, 1, min(20, num_neurons))  # Limit number of neurons for visualization
                    z = np.zeros_like(y)
                    
                    # Add neurons as scatter points
                    fig.add_trace(go.Scatter3d(
                        x=np.full_like(y, x),
                        y=y,
                        z=z,
                        mode='markers',
                        marker=dict(
                            size=5,
                            color=color,
                            opacity=0.7,
                            line=dict(width=0.5, color='white')
                        ),
                        name=f"{layer_type} {i}",
                        text=f"{layer_type}<br>Shape: {layer['output_shape']}<br>Params: {layer['trainable_params']}",
                        hoverinfo='text',
                        showlegend=True if i == 0 else False
                    ))
                    
                    # Store layer position for connections
                    layer_positions.append({
                        'x': x,
                        'y': 0,
                        'z': 0,
                        'type': layer_type,
                        'shape': layer['output_shape']
                    })
            
            # Add layer label
            fig.add_annotation(
                x=x,
                y=1.2,
                z=0,
                text=f"{layer_type}<br>{layer['output_shape']}",
                showarrow=False,
                font=dict(size=10, color="black"),
                bgcolor="white",
                opacity=0.8
            )
        
        # Add connections between layers
        for i in range(1, len(layer_positions)):
            prev_layer = layer_positions[i-1]
            curr_layer = layer_positions[i]
            
            # Add a few sample connections for visualization
            for j in range(3):
                fig.add_trace(go.Scatter3d(
                    x=[prev_layer['x'], curr_layer['x']],
                    y=[-1 + j, -1 + j],
                    z=[0, 0],
                    mode='lines',
                    line=dict(
                        color='gray',
                        width=1,
                        dash='dot'
                    ),
                    showlegend=False,
                    hoverinfo='none'
                ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='Interactive 3D Neural Network Architecture',
                x=0.5,
                xanchor='center',
                y=0.95,
                yanchor='top',
                font=dict(size=20)
            ),
            scene=dict(
                xaxis_title='Layers',
                yaxis_title='Neurons',
                zaxis_title='Channels',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=0.8)
                ),
                xaxis=dict(showbackground=False),
                yaxis=dict(showbackground=False),
                zaxis=dict(showbackground=False)
            ),
            width=1000,
            height=800,
            margin=dict(l=0, r=0, b=0, t=50),
            showlegend=True,
            legend=dict(
                x=1.05,
                y=0.5,
                traceorder='normal',
                font=dict(size=12)
            )
        )
        
        return fig
    
    def create_architecture_animation(self, n_frames=36):
        """Create a rotating 3D animation of the architecture."""
        # Create a 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Colors for different layer types
        layer_colors = {
            'Dense': 'blue',
            'Conv2D': 'green',
            'MaxPooling2D': 'red',
            'Flatten': 'purple',
            'Dropout': 'orange',
            'BatchNormalization': 'brown',
            'LSTM': 'pink',
            'GRU': 'cyan',
            'Embedding': 'magenta',
            'default': 'gray'
        }
        
        # Plot each layer
        for i, layer in enumerate(self.layer_info):
            layer_type = layer['type'].split('.')[-1]
            color = layer_colors.get(layer_type, layer_colors['default'])
            
            # Position of the layer
            x = i * 2.0
            
            # For each neuron in the layer
            if len(layer['output_shape']) > 1:
                if len(layer['output_shape']) == 4:  # Conv layer
                    _, height, width, channels = layer['output_shape']
                    y = np.linspace(-1, 1, height)
                    z = np.linspace(-1, 1, width)
                    Y, Z = np.meshgrid(y, z)
                    
                    # Limit number of points for performance
                    step = max(1, height // 5)
                    Y = Y[::step, ::step]
                    Z = Z[::step, ::step]
                    
                    # Plot neurons as points
                    ax.scatter(
                        np.full_like(Y, x), 
                        Y, 
                        Z, 
                        s=10, 
                        c=color,
                        alpha=0.7,
                        edgecolors='none',
                        label=layer_type if i == 0 else ""
                    )
                    
                else:  # Dense or other layers
                    num_neurons = layer['output_shape'][1] if len(layer['output_shape']) > 1 else layer['output_shape'][0]
                    y = np.linspace(-1, 1, min(20, num_neurons))  # Limit number of neurons
                    z = np.zeros_like(y)
                    
                    ax.scatter(
                        np.full_like(y, x), 
                        y, 
                        z, 
                        s=20, 
                        c=color,
                        alpha=0.7,
                        edgecolors='none',
                        label=layer_type if i == 0 else ""
                    )
            
            # Add layer label
            ax.text(
                x, 0, 1.5, 
                f"{layer_type}\n{layer['output_shape']}", 
                fontsize=8,
                ha='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )
        
        # Set labels and title
        ax.set_xlabel('Layers')
        ax.set_ylabel('Neurons')
        ax.set_zlabel('Channels')
        ax.set_title('3D Neural Network Architecture Animation')
        
        # Add legend
        handles = []
        for layer_type, color in layer_colors.items():
            if layer_type != 'default':
                handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, markersize=10, 
                                        label=layer_type))
        
        ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Animation function
        def update(frame):
            ax.view_init(elev=20, azim=frame * 10)
            return (ax,)
        
        # Create animation
        anim = FuncAnimation(
            fig, update, frames=n_frames,
            interval=100, blit=True
        )
        
        # Close the figure to prevent it from displaying twice
        plt.close(fig)
        
        # Return HTML5 video
        return HTML(anim.to_html5_video())
