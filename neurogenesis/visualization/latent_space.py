"""
Latent Space Visualization

Provides interactive 3D visualizations of latent spaces in deep learning models.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import umap.plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

class LatentSpaceExplorer:
    """Interactive 3D visualization of latent spaces with multiple projection methods."""
    
    def __init__(self, model=None, method='tsne', n_components=3):
        """Initialize the latent space explorer."""
        self.model = model
        self.method = method
        self.n_components = n_components
        self.projection = None
        self.scaler = StandardScaler()
        
    def fit_transform(self, data):
        """Fit and transform data to lower dimensions."""
        # Standardize the data
        data_std = self.scaler.fit_transform(data)
        
        # Apply dimensionality reduction
        if self.method == 'tsne':
            self.projection = TSNE(
                n_components=self.n_components,
                perplexity=30,
                n_iter=1000,
                random_state=42
            ).fit_transform(data_std)
            
        elif self.method == 'pca':
            self.projection = PCA(
                n_components=self.n_components,
                random_state=42
            ).fit_transform(data_std)
            
        elif self.method == 'umap':
            self.projection = umap.UMAP(
                n_components=self.n_components,
                n_neighbors=15,
                min_dist=0.1,
                metric='euclidean',
                random_state=42
            ).fit_transform(data_std)
            
        elif self.method == 'isomap':
            self.projection = Isomap(
                n_components=self.n_components,
                n_neighbors=5
            ).fit_transform(data_std)
            
        return self.projection
    
    def plot_3d_latent_space(self, points, labels=None, images=None, title='3D Latent Space'):
        """Create an interactive 3D plot of the latent space."""
        if self.projection is None:
            self.fit_transform(points)
            
        if self.n_components < 3:
            return self._plot_2d_latent_space(points, labels, images, title)
            
        # Create a 3D scatter plot
        fig = go.Figure()
        
        # Add points to the plot
        if labels is not None:
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = (labels == label)
                fig.add_trace(go.Scatter3d(
                    x=self.projection[mask, 0],
                    y=self.projection[mask, 1],
                    z=self.projection[mask, 2],
                    mode='markers',
                    name=f'Class {label}',
                    marker=dict(
                        size=5,
                        opacity=0.7,
                        line=dict(width=0.5, color='white')
                    ),
                    hoverinfo='text',
                    hovertext=[f'Class: {label}<br>Point: {i}' for i in np.where(mask)[0]]
                ))
        else:
            fig.add_trace(go.Scatter3d(
                x=self.projection[:, 0],
                y=self.projection[:, 1],
                z=self.projection[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    opacity=0.7,
                    colorscale='Viridis',
                    color=np.linspace(0, 1, len(self.projection)),
                    line=dict(width=0.5, color='white')
                ),
                hoverinfo='text',
                hovertext=[f'Point: {i}' for i in range(len(self.projection))]
            ))
        
        # Add images if provided
        if images is not None and len(images) > 0:
            # For 3D, we can't directly add images, so we'll add them as annotations
            # with hover text showing the image
            pass
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                y=0.95,
                yanchor='top',
                font=dict(size=20)
            ),
            scene=dict(
                xaxis_title='Dimension 1',
                yaxis_title='Dimension 2',
                zaxis_title='Dimension 3',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=0.8)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            showlegend=labels is not None,
            width=1000,
            height=800
        )
        
        return fig
    
    def _plot_2d_latent_space(self, points, labels=None, images=None, title='2D Latent Space'):
        """Create a 2D plot of the latent space."""
        if self.projection is None:
            self.fit_transform(points)
            
        # Create a 2D scatter plot
        fig = go.Figure()
        
        # Add points to the plot
        if labels is not None:
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = (labels == label)
                fig.add_trace(go.Scatter(
                    x=self.projection[mask, 0],
                    y=self.projection[mask, 1],
                    mode='markers',
                    name=f'Class {label}',
                    marker=dict(
                        size=8,
                        opacity=0.7,
                        line=dict(width=0.5, color='white')
                    ),
                    hoverinfo='text',
                    hovertext=[f'Class: {label}<br>Point: {i}' for i in np.where(mask)[0]]
                ))
        else:
            fig.add_trace(go.Scatter(
                x=self.projection[:, 0],
                y=self.projection[:, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    opacity=0.7,
                    colorscale='Viridis',
                    color=np.linspace(0, 1, len(self.projection)),
                    line=dict(width=0.5, color='white')
                ),
                hoverinfo='text',
                hovertext=[f'Point: {i}' for i in range(len(self.projection))]
            ))
        
        # Add images if provided
        if images is not None and len(images) > 0:
            # For 2D, we can add images as annotations
            pass
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                y=0.95,
                yanchor='top',
                font=dict(size=20)
            ),
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            showlegend=labels is not None,
            width=900,
            height=800,
            hovermode='closest'
        )
        
        return fig
    
    def plot_interactive_latent_space(self, points, labels=None, images=None, title='Interactive Latent Space'):
        """Create an interactive plot with multiple visualization options."""
        if self.projection is None:
            self.fit_transform(points)
            
        # Create figure with multiple subplots
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]],
            subplot_titles=(
                '3D Projection',
                '2D Projection',
                'Density Plot',
                'Parallel Coordinates'
            )
        )
        
        # 3D Projection
        if self.n_components >= 3:
            fig.add_trace(
                go.Scatter3d(
                    x=self.projection[:, 0],
                    y=self.projection[:, 1],
                    z=self.projection[:, 2],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=labels if labels is not None else 'blue',
                        colorscale='Viridis',
                        opacity=0.7,
                        line=dict(width=0.5, color='white')
                    ),
                    hoverinfo='text',
                    hovertext=[f'Point: {i}' for i in range(len(self.projection))],
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # 2D Projection
        fig.add_trace(
            go.Scatter(
                x=self.projection[:, 0],
                y=self.projection[:, 1],
                mode='markers',
                marker=dict(
                    size=6,
                    color=labels if labels is not None else 'blue',
                    colorscale='Viridis',
                    opacity=0.7,
                    line=dict(width=0.5, color='white')
                ),
                hoverinfo='text',
                hovertext=[f'Point: {i}' for i in range(len(self.projection))],
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Density Plot
        fig.add_trace(
            go.Histogram2dContour(
                x=self.projection[:, 0],
                y=self.projection[:, 1],
                colorscale='Hot',
                reversescale=True,
                xaxis='x3',
                yaxis='y3',
                showscale=False
            ),
            row=2, col=1
        )
        
        # Parallel Coordinates (if we have more than 2 dimensions)
        if self.n_components >= 3:
            fig.add_trace(
                go.Parcoords(
                    line=dict(
                        color=labels if labels is not None else 'blue',
                        colorscale='Viridis',
                        showscale=True,
                        reversescale=True,
                        cmin=min(labels) if labels is not None else 0,
                        cmax=max(labels) if labels is not None else 1
                    ),
                    dimensions=[
                        dict(
                            label=f'Dim {i+1}',
                            values=self.projection[:, i]
                        ) for i in range(min(5, self.n_components))
                    ]
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                y=0.99,
                yanchor='top',
                font=dict(size=24)
            ),
            showlegend=False,
            width=1200,
            height=1000,
            hovermode='closest'
        )
        
        # Update axes
        fig.update_xaxes(title_text='Dimension 1', row=1, col=2)
        fig.update_yaxes(title_text='Dimension 2', row=1, col=2)
        
        if self.n_components >= 3:
            fig.update_scenes(
                xaxis_title='Dimension 1',
                yaxis_title='Dimension 2',
                zaxis_title='Dimension 3',
                row=1, col=1
            )
        
        return fig
    
    def create_animation(self, points, labels=None, n_frames=36):
        """Create a rotating 3D animation of the latent space."""
        if self.projection is None or self.n_components < 3:
            return None
            
        # Create figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points
        scatter = ax.scatter(
            self.projection[:, 0],
            self.projection[:, 1],
            self.projection[:, 2],
            c=labels if labels is not None else 'b',
            cmap='viridis',
            s=20,
            alpha=0.7
        )
        
        # Set labels
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        ax.set_title('3D Latent Space Animation')
        
        # Animation function
        def update(frame):
            ax.view_init(elev=20, azim=frame)
            return (scatter,)
        
        # Create animation
        anim = FuncAnimation(
            fig, update, frames=np.linspace(0, 360, n_frames, endpoint=False),
            interval=100, blit=True
        )
        
        # Close the figure to prevent it from displaying twice
        plt.close(fig)
        
        # Return HTML5 video
        return HTML(anim.to_html5_video())
