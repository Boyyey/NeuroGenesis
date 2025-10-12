"""
Interactive 3D Feature Map Visualization

This module provides tools for visualizing feature maps from convolutional
layers in 3D, including depth-wise and channel-wise visualizations.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Tuple, Dict, Optional, Union
import tensorflow as tf
from PIL import Image
import cv2

class FeatureMapVisualizer:
    """Class for visualizing feature maps from convolutional layers in 3D."""
    
    def __init__(self, model=None, layer_names: List[str] = None):
        """Initialize the FeatureMapVisualizer with a model and optional layer names.
        
        Args:
            model: Keras model to visualize
            layer_names: List of layer names to visualize (if None, all conv layers will be used)
        """
        self.model = model
        self.layer_names = layer_names
        self.feature_maps = {}
        self.figures = {}
        
        if model is not None:
            self._extract_layer_outputs()
    
    def _extract_layer_outputs(self):
        """Extract output tensors for each convolutional layer in the model."""
        if self.model is None:
            return
            
        # Get all convolutional layers if no specific layers are provided
        if self.layer_names is None:
            self.layer_names = [
                layer.name for layer in self.model.layers 
                if 'conv' in layer.name.lower() or 'pool' in layer.name.lower()
            ]
        
        # Create a model that will return the outputs of the specified layers
        layer_outputs = [self.model.get_layer(name).output for name in self.layer_names]
        self.feature_model = tf.keras.models.Model(
            inputs=self.model.inputs, 
            outputs=layer_outputs
        )
    
    def get_feature_maps(self, input_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Get feature maps for the input data.
        
        Args:
            input_data: Input data to get feature maps for
            
        Returns:
            Dictionary mapping layer names to their corresponding feature maps
        """
        if not hasattr(self, 'feature_model'):
            self._extract_layer_outputs()
            
        # Get feature maps for all layers
        layer_outputs = self.feature_model(input_data)
        
        # Convert to dictionary
        self.feature_maps = {
            name: output.numpy() 
            for name, output in zip(self.layer_names, layer_outputs)
        }
        
        return self.feature_maps
    
    def plot_feature_maps_3d(self, layer_name: str, sample_idx: int = 0, 
                           channel_indices: List[int] = None, n_cols: int = 4,
                           cmap: str = 'viridis', opacity: float = 0.8) -> go.Figure:
        """Create an interactive 3D visualization of feature maps.
        
        Args:
            layer_name: Name of the layer to visualize
            sample_idx: Index of the sample to visualize
            channel_indices: List of channel indices to visualize (if None, first 4 channels)
            n_cols: Number of columns in the subplot grid
            cmap: Colormap to use for the visualization
            opacity: Opacity of the 3D surface
            
        Returns:
            plotly.graph_objects.Figure: 3D visualization of feature maps
        """
        if not self.feature_maps:
            raise ValueError("No feature maps found. Call get_feature_maps() first.")
            
        if layer_name not in self.feature_maps:
            raise ValueError(f"Layer '{layer_name}' not found. Available layers: {list(self.feature_maps.keys())}")
        
        # Get feature maps for the specified layer and sample
        feature_maps = self.feature_maps[layer_name][sample_idx]
        
        # Handle different input shapes (e.g., (height, width, channels) or (channels, height, width))
        if len(feature_maps.shape) == 3:
            # Assume channels_last format (height, width, channels)
            height, width, num_channels = feature_maps.shape
            channel_axis = -1
        else:
            # Assume channels_first format (channels, height, width)
            num_channels, height, width = feature_maps.shape
            channel_axis = 0
        
        # Select channels to visualize
        if channel_indices is None:
            channel_indices = list(range(min(4, num_channels)))
        
        num_channels_to_plot = len(channel_indices)
        n_rows = (num_channels_to_plot + n_cols - 1) // n_cols
        
        # Create subplots
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[f'Channel {i}' for i in channel_indices],
            specs=[[{'type': 'surface'} for _ in range(n_cols)] for _ in range(n_rows)]
        )
        
        # Create meshgrid for 3D surface
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)
        
        # Plot each channel
        for idx, channel_idx in enumerate(channel_indices):
            row = idx // n_cols + 1
            col = idx % n_cols + 1
            
            # Get the channel data
            if channel_axis == -1:
                Z = feature_maps[:, :, channel_idx]
            else:
                Z = feature_maps[channel_idx, :, :]
            
            # Normalize to [0, 1] for better visualization
            Z_norm = (Z - Z.min()) / (Z.max() - Z.min() + 1e-8)
            
            # Add 3D surface
            fig.add_trace(
                go.Surface(
                    z=Z_norm,
                    x=X,
                    y=Y,
                    colorscale=cmap,
                    showscale=False,
                    opacity=opacity,
                    hoverinfo='z',
                    hovertext=f'Channel {channel_idx}<br>Value: {Z.ravel():.4f}',
                    hoverlabel=dict(namelength=0)
                ),
                row=row, col=col
            )
            
            # Update subplot layout
            fig.update_scenes(
                dict(
                    xaxis_title='Width',
                    yaxis_title='Height',
                    zaxis_title='Activation',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=0.8)
                    ),
                    aspectratio=dict(x=1, y=1, z=0.5)
                ),
                row=row, col=col
            )
        
        # Update layout
        fig.update_layout(
            title=f'3D Feature Maps: {layer_name}',
            height=300 * n_rows,
            margin=dict(l=0, r=0, b=0, t=50),
            showlegend=False
        )
        
        self.figures[f'feature_maps_3d_{layer_name}'] = fig
        return fig
    
    def plot_feature_map_evolution_3d(self, layer_name: str, sample_idx: int = 0, 
                                    channel_idx: int = 0, n_frames: int = 20,
                                    cmap: str = 'viridis', opacity: float = 0.9) -> go.Figure:
        """Create an animated 3D visualization of feature map evolution over time.
        
        Args:
            layer_name: Name of the layer to visualize
            sample_idx: Index of the sample to visualize
            channel_idx: Index of the channel to visualize
            n_frames: Number of frames for the animation
            cmap: Colormap to use for the visualization
            opacity: Opacity of the 3D surface
            
        Returns:
            plotly.graph_objects.Figure: Animated 3D visualization of feature map evolution
        """
        if not self.feature_maps:
            raise ValueError("No feature maps found. Call get_feature_maps() first.")
            
        if layer_name not in self.feature_maps:
            raise ValueError(f"Layer '{layer_name}' not found. Available layers: {list(self.feature_maps.keys())}")
        
        # Get feature maps for the specified layer and sample
        feature_maps = self.feature_maps[layer_name][sample_idx]
        
        # Handle different input shapes
        if len(feature_maps.shape) == 3:
            # channels_last format (height, width, channels)
            height, width, num_channels = feature_maps.shape
            channel_slice = (slice(None), slice(None), channel_idx % num_channels)
        else:
            # channels_first format (channels, height, width)
            num_channels, height, width = feature_maps.shape
            channel_slice = (channel_idx % num_channels, slice(None), slice(None))
        
        # Create meshgrid for 3D surface
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)
        
        # Create figure with frames for animation
        fig = go.Figure(
            data=[go.Surface(
                z=feature_maps[channel_slice],
                x=X,
                y=Y,
                colorscale=cmap,
                cmin=feature_maps[channel_slice].min(),
                cmax=feature_maps[channel_slice].max(),
                opacity=opacity,
                colorbar=dict(title='Activation'),
                hovertemplate='<b>X</b>: %{x}<br>'
                            '<b>Y</b>: %{y}<br>'
                            '<b>Activation</b>: %{z:.4f}<extra></extra>'
            )],
            layout=go.Layout(
                title=f'Feature Map Evolution: {layer_name} (Channel {channel_idx})',
                scene=dict(
                    xaxis_title='Width',
                    yaxis_title='Height',
                    zaxis_title='Activation',
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
                        z=feature_maps[channel_slice] * (i / n_frames),
                        x=X,
                        y=Y
                    )],
                    name=f'frame_{i}'
                )
                for i in range(n_frames)
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
                    'prefix': 'Time:',
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
                    for i in range(n_frames)
                ]
            }]
        )
        
        self.figures[f'feature_map_evolution_{layer_name}_ch{channel_idx}'] = fig
        return fig
    
    def plot_feature_map_channels_3d(self, layer_name: str, sample_idx: int = 0, 
                                   n_channels: int = 8, cmap: str = 'viridis',
                                   opacity: float = 0.8) -> go.Figure:
        """Create a 3D visualization of multiple channels from a feature map.
        
        Args:
            layer_name: Name of the layer to visualize
            sample_idx: Index of the sample to visualize
            n_channels: Number of channels to visualize
            cmap: Colormap to use for the visualization
            opacity: Opacity of the 3D surfaces
            
        Returns:
            plotly.graph_objects.Figure: 3D visualization of feature map channels
        """
        if not self.feature_maps:
            raise ValueError("No feature maps found. Call get_feature_maps() first.")
            
        if layer_name not in self.feature_maps:
            raise ValueError(f"Layer '{layer_name}' not found. Available layers: {list(self.feature_maps.keys())}")
        
        # Get feature maps for the specified layer and sample
        feature_maps = self.feature_maps[layer_name][sample_idx]
        
        # Handle different input shapes
        if len(feature_maps.shape) == 3:
            # channels_last format (height, width, channels)
            height, width, num_channels = feature_maps.shape
            channel_axis = -1
        else:
            # channels_first format (channels, height, width)
            num_channels, height, width = feature_maps.shape
            channel_axis = 0
        
        # Limit number of channels to visualize
        n_channels = min(n_channels, num_channels)
        channel_indices = np.linspace(0, num_channels - 1, n_channels, dtype=int)
        
        # Create figure
        fig = go.Figure()
        
        # Create meshgrid for 3D surface
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)
        
        # Add each channel as a 3D surface
        for i, channel_idx in enumerate(channel_indices):
            # Get the channel data
            if channel_axis == -1:
                Z = feature_maps[:, :, channel_idx]
            else:
                Z = feature_maps[channel_idx, :, :]
            
            # Normalize to [0, 1] for better visualization
            Z_norm = (Z - Z.min()) / (Z.max() - Z.min() + 1e-8)
            
            # Add 3D surface
            fig.add_trace(go.Surface(
                z=Z_norm + i * 1.2,  # Offset each surface for better visibility
                x=X,
                y=Y,
                colorscale=cmap,
                surfacecolor=Z_norm,
                cmin=0,
                cmax=1,
                opacity=opacity,
                name=f'Channel {channel_idx}',
                showscale=True if i == 0 else False,
                colorbar=dict(title='Activation') if i == 0 else None,
                hovertemplate='<b>Channel</b>: ' + str(channel_idx) + 
                            '<br><b>X</b>: %{x}' +
                            '<br><b>Y</b>: %{y}' +
                            '<br><b>Activation</b>: %{z:.4f}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=f'3D Feature Map Channels: {layer_name}',
            scene=dict(
                xaxis_title='Width',
                yaxis_title='Height',
                zaxis_title='Channel Activation',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=0.8)
                ),
                aspectratio=dict(x=1, y=1, z=1)
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            height=700,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        self.figures[f'feature_map_channels_3d_{layer_name}'] = fig
        return fig
    
    def plot_feature_map_activation_paths(self, layer_name: str, sample_idx: int = 0,
                                        n_paths: int = 10, path_length: int = 5,
                                        cmap: str = 'viridis') -> go.Figure:
        """Create a 3D visualization of activation paths through a feature map.
        
        Args:
            layer_name: Name of the layer to visualize
            sample_idx: Index of the sample to visualize
            n_paths: Number of paths to visualize
            path_length: Length of each path
            cmap: Colormap to use for the visualization
            
        Returns:
            plotly.graph_objects.Figure: 3D visualization of activation paths
        """
        if not self.feature_maps:
            raise ValueError("No feature maps found. Call get_feature_maps() first.")
            
        if layer_name not in self.feature_maps:
            raise ValueError(f"Layer '{layer_name}' not found. Available layers: {list(self.feature_maps.keys())}")
        
        # Get feature maps for the specified layer and sample
        feature_maps = self.feature_maps[layer_name][sample_idx]
        
        # Handle different input shapes
        if len(feature_maps.shape) == 3:
            # channels_last format (height, width, channels)
            height, width, num_channels = feature_maps.shape
            channel_axis = -1
        else:
            # channels_first format (channels, height, width)
            num_channels, height, width = feature_maps.shape
            channel_axis = 0
        
        # Create figure
        fig = go.Figure()
        
        # Generate random paths through the feature map
        for path_idx in range(n_paths):
            # Random starting point
            if channel_axis == -1:
                # Random channel, random position
                channel = np.random.randint(0, num_channels)
                x, y = np.random.randint(0, width), np.random.randint(0, height)
            else:
                # Random position, random channel
                x, y = np.random.randint(0, width), np.random.randint(0, height)
                channel = np.random.randint(0, num_channels)
            
            # Generate path
            x_path = [x]
            y_path = [y]
            z_path = [feature_maps[y, x, channel] if channel_axis == -1 else 
                     feature_maps[channel, y, x]]
            
            for _ in range(1, path_length):
                # Random walk
                dx, dy = np.random.randint(-1, 2, 2)
                x = np.clip(x + dx, 0, width - 1)
                y = np.clip(y + dy, 0, height - 1)
                
                x_path.append(x)
                y_path.append(y)
                z_path.append(feature_maps[y, x, channel] if channel_axis == -1 else 
                            feature_maps[channel, y, x])
            
            # Add path to figure
            fig.add_trace(go.Scatter3d(
                x=x_path,
                y=y_path,
                z=z_path,
                mode='lines+markers',
                line=dict(
                    width=4,
                    color=z_path,
                    colorscale=cmap,
                    showscale=False
                ),
                marker=dict(
                    size=4,
                    color=z_path,
                    colorscale=cmap,
                    showscale=False,
                    opacity=0.8
                ),
                name=f'Path {path_idx + 1}',
                hovertemplate='<b>X</b>: %{x}<br>'
                            '<b>Y</b>: %{y}<br>'
                            '<b>Activation</b>: %{z:.4f}<extra></extra>'
            ))
        
        # Add a surface representing the feature map
        if channel_axis == -1:
            # Use first channel for the surface
            Z = feature_maps[:, :, 0]
        else:
            # Use first channel for the surface
            Z = feature_maps[0, :, :]
        
        # Normalize for visualization
        Z_norm = (Z - Z.min()) / (Z.max() - Z.min() + 1e-8)
        
        # Create meshgrid for surface
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)
        
        # Add surface
        fig.add_trace(go.Surface(
            z=Z_norm * 0.5,  # Scale down for better visibility
            x=X,
            y=Y,
            colorscale='Greys',
            opacity=0.3,
            showscale=False,
            hoverinfo='none'
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Activation Paths: {layer_name}',
            scene=dict(
                xaxis_title='Width',
                yaxis_title='Height',
                zaxis_title='Activation',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=0.8)
                ),
                aspectratio=dict(x=1, y=1, z=0.5)
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            height=700,
            showlegend=True
        )
        
        self.figures[f'feature_map_paths_{layer_name}'] = fig
        return fig
    
    def plot_feature_map_volume(self, layer_name: str, sample_idx: int = 0,
                              n_slices: int = 10, cmap: str = 'viridis') -> go.Figure:
        """Create a 3D volume visualization of a feature map.
        
        Args:
            layer_name: Name of the layer to visualize
            sample_idx: Index of the sample to visualize
            n_slices: Number of slices to show in the volume
            cmap: Colormap to use for the visualization
            
        Returns:
            plotly.graph_objects.Figure: 3D volume visualization of the feature map
        """
        if not self.feature_maps:
            raise ValueError("No feature maps found. Call get_feature_maps() first.")
            
        if layer_name not in self.feature_maps:
            raise ValueError(f"Layer '{layer_name}' not found. Available layers: {list(self.feature_maps.keys())}")
        
        # Get feature maps for the specified layer and sample
        feature_maps = self.feature_maps[layer_name][sample_idx]
        
        # Handle different input shapes
        if len(feature_maps.shape) == 3:
            # channels_last format (height, width, channels)
            height, width, num_channels = feature_maps.shape
            channel_axis = -1
        else:
            # channels_first format (channels, height, width)
            num_channels, height, width = feature_maps.shape
            channel_axis = 0
        
        # Prepare data for volume plot
        if channel_axis == -1:
            # Move channels to first dimension for volume visualization
            volume_data = np.moveaxis(feature_maps, -1, 0)
        else:
            volume_data = feature_maps
        
        # Normalize data
        volume_data = (volume_data - volume_data.min()) / (volume_data.max() - volume_data.min() + 1e-8)
        
        # Sample slices if needed
        if n_slices < num_channels:
            step = max(1, num_channels // n_slices)
            volume_data = volume_data[::step]
        
        # Create figure
        fig = go.Figure(data=go.Volume(
            z=np.linspace(0, 1, volume_data.shape[0]),
            y=np.linspace(0, 1, volume_data.shape[1]),
            x=np.linspace(0, 1, volume_data.shape[2]),
            value=volume_data.ravel(),
            isomin=0.1,
            isomax=0.9,
            opacity=0.1,  # needs to be small to see through all surfaces
            surface_count=20,  # needs to be a large number for good volume rendering
            colorscale=cmap,
            showscale=True,
            slices_z=dict(show=True, locations=[0.2, 0.5, 0.8]),
            caps=dict(x_show=False, y_show=False, z_show=False),
            hoverinfo='skip'
        ))
        
        # Add outline
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    nticks=5, range=[0, 1],
                    showbackground=True,
                    backgroundcolor='rgb(240, 240, 240)',
                    gridcolor='white',
                    showspikes=False
                ),
                yaxis=dict(
                    nticks=5, range=[0, 1],
                    showbackground=True,
                    backgroundcolor='rgb(240, 240, 240)',
                    gridcolor='white',
                    showspikes=False
                ),
                zaxis=dict(
                    nticks=5, range=[0, 1],
                    showbackground=True,
                    backgroundcolor='rgb(240, 240, 240)',
                    gridcolor='white',
                    showspikes=False
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            height=700
        )
        
        # Add title
        fig.update_layout(
            title=f'3D Feature Map Volume: {layer_name}'
        )
        
        self.figures[f'feature_map_volume_{layer_name}'] = fig
        return fig
    
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
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.vgg16 import preprocess_input
    
    # Load a pre-trained model
    model = VGG16(weights='imagenet', include_top=True)
    
    # Load and preprocess an example image
    img_path = 'example.jpg'  # Replace with your image path
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # Create visualizer with the model
    visualizer = FeatureMapVisualizer(model)
    
    # Get feature maps for the input image
    visualizer.get_feature_maps(x)
    
    # Create visualizations
    fig1 = visualizer.plot_feature_maps_3d('block1_conv1')
    fig2 = visualizer.plot_feature_map_evolution_3d('block2_conv2')
    fig3 = visualizer.plot_feature_map_channels_3d('block3_conv3', n_channels=6)
    fig4 = visualizer.plot_feature_map_activation_paths('block4_conv3')
    fig5 = visualizer.plot_feature_map_volume('block5_conv3', n_slices=20)
    
    # Show all figures
    visualizer.show()
