"""
Attention Visualization Module

Provides interactive 3D visualizations of attention mechanisms in transformer models.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

class AttentionVisualizer:
    """Visualize attention weights in 3D and 2D with interactive controls."""
    
    def __init__(self, model, tokenizer=None):
        """Initialize with a trained model and optional tokenizer."""
        self.model = model
        self.tokenizer = tokenizer
    
    def plot_3d_attention(self, attention_weights, tokens=None, layer=0, head=0):
        """Create a 3D surface plot of attention weights."""
        fig = go.Figure(data=[
            go.Surface(
                z=attention_weights[layer][head],
                colorscale='Viridis',
                opacity=0.8,
                contours={
                    "x": {"show": True, "color": "white", "start": 0, "end": 1, "size": 0.1},
                    "y": {"show": True, "color": "white", "start": 0, "end": 1, "size": 0.1},
                    "z": {"show": True, "color": "white", "start": 0, "end": 1, "size": 0.1}
                }
            )
        ])
        
        fig.update_layout(
            title=f'3D Attention Map - Layer {layer}, Head {head}',
            scene={
                'xaxis_title': 'Query Position',
                'yaxis_title': 'Key Position',
                'zaxis_title': 'Attention Weight',
                'camera': {
                    'up': {'x': 0, 'y': 0, 'z': 1},
                    'center': {'x': 0, 'y': 0, 'z': 0},
                    'eye': {'x': 2, 'y': 2, 'z': 1}
                }
            },
            width=1000,
            height=800
        )
        
        return fig
    
    def plot_attention_flow(self, attention_weights, tokens, layer_range=None):
        """Visualize how attention flows through layers."""
        if layer_range is None:
            layer_range = (0, len(attention_weights) - 1)
            
        n_layers = layer_range[1] - layer_range[0] + 1
        fig = make_subplots(
            rows=n_layers, cols=1,
            subplot_titles=[f'Layer {i}' for i in range(layer_range[0], layer_range[1] + 1)],
            vertical_spacing=0.02
        )
        
        for i in range(n_layers):
            layer_idx = layer_range[0] + i
            # Take average attention across heads
            avg_attention = np.mean(attention_weights[layer_idx], axis=0)
            
            fig.add_trace(
                go.Heatmap(
                    z=avg_attention,
                    colorscale='Viridis',
                    showscale=False,
                    text=[[f"{tokens[i]}→{tokens[j]}: {avg_attention[i,j]:.2f}" 
                          for j in range(len(tokens))] 
                         for i in range(len(tokens))],
                    hoverinfo='text'
                ),
                row=i+1, col=1
            )
            
            # Add token labels if available
            if tokens:
                fig.update_xaxes(
                    ticktext=tokens,
                    tickvals=list(range(len(tokens))),
                    row=i+1, col=1
                )
                fig.update_yaxes(
                    ticktext=tokens,
                    tickvals=list(range(len(tokens))),
                    row=i+1, col=1
                )
        
        fig.update_layout(
            title='Attention Flow Through Layers',
            height=300 * n_layers,
            showlegend=False
        )
        
        return fig
    
    def plot_attention_3d_volume(self, attention_weights, tokens=None):
        """Create a 3D volume visualization of attention across all layers and heads."""
        n_layers = len(attention_weights)
        n_heads = len(attention_weights[0])
        
        # Prepare data for 3D volume
        X, Y, Z = np.mgrid[0:n_heads, 0:n_layers, 0:1]
        
        # Create figure
        fig = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=attention_weights[0][0].flatten(),  # Will be updated by frames
            isomin=0,
            isomax=1,
            opacity=0.1,
            surface_count=20,
            colorscale='Viridis',
            caps=dict(x_show=False, y_show=False, z_show=False),
        ))
        
        # Add frames for animation
        frames = []
        for i in range(n_layers):
            for j in range(n_heads):
                frames.append(go.Frame(
                    data=[go.Volume(
                        value=attention_weights[i][j].flatten(),
                        colorscale='Viridis'
                    )],
                    name=f"Layer {i}, Head {j}"
                ))
        
        fig.frames = frames
        
        # Add sliders
        sliders = [{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 20},
                'prefix': 'Attention Head: ',
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
                        [f"Layer {i//n_heads}, Head {i%n_heads}"],
                        {'frame': {'duration': 300, 'redraw': True},
                         'mode': 'immediate',
                         'transition': {'duration': 300}}
                    ],
                    'label': f"L{i//n_heads}H{i%n_heads}",
                    'method': 'animate'
                } for i in range(n_layers * n_heads)
            ]
        }]
        
        fig.update_layout(
            title='3D Attention Volume Across Layers and Heads',
            scene={
                'xaxis_title': 'Heads',
                'yaxis_title': 'Layers',
                'zaxis_title': 'Attention',
                'camera': {
                    'up': {'x': 0, 'y': 0, 'z': 1},
                    'center': {'x': 0, 'y': 0, 'z': 0},
                    'eye': {'x': 1.5, 'y': 1.5, 'z': 1}
                }
            },
            sliders=sliders,
            updatemenus=[{
                'buttons': [
                    {
                        'args': [None, {
                            'frame': {'duration': 500, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 300, 'easing': 'quadratic-in-out'}
                        }],
                        'label': 'Play',
                        'method': 'animate'
                    },
                    {
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }],
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
            }],
            width=1000,
            height=800
        )
        
        return fig
