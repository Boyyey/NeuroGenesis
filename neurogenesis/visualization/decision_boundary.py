"""
Interactive 3D Decision Boundary Visualization

This module provides tools for visualizing decision boundaries of classification
models in 3D space, including support for multi-class classification and probability
surfaces.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from typing import List, Tuple, Dict, Optional, Union, Callable
import pandas as pd
from tqdm import tqdm

class DecisionBoundaryVisualizer:
    """Class for creating interactive 3D visualizations of decision boundaries."""
    
    def __init__(self, model=None, feature_names: List[str] = None, 
                 class_names: List[str] = None):
        """Initialize the DecisionBoundaryVisualizer.
        
        Args:
            model: Trained classification model with predict() or predict_proba() method
            feature_names: List of feature names (for axis labels)
            class_names: List of class names (for legend)
        """
        self.model = model
        self.feature_names = feature_names or ['Feature 1', 'Feature 2', 'Feature 3']
        self.class_names = class_names
        self.figures = {}
        
    def _get_model_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from the model, handling different model types."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        elif hasattr(self.model, 'predict'):
            return self.model.predict(X)
        elif callable(self.model):
            return self.model(X)
        else:
            raise ValueError("Model must have predict(), predict_proba(), or be callable")
    
    def plot_3d_decision_boundary(self, X: np.ndarray, y: np.ndarray, 
                                resolution: int = 50, opacity: float = 0.5,
                                colorscale: str = 'Viridis', 
                                show_points: bool = True,
                                title: str = None) -> go.Figure:
        """Create a 3D plot of the decision boundary and data points.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target labels (n_samples,)
            resolution: Resolution of the decision surface grid
            opacity: Opacity of the decision surface
            colorscale: Colorscale for the decision surface
            show_points: Whether to show the data points
            title: Plot title
            
        Returns:
            plotly.graph_objects.Figure: 3D plot of decision boundary
        """
        if X.shape[1] < 2:
            raise ValueError("At least 2 features are required for 3D visualization")
            
        # If more than 3 features, reduce to 3D using PCA
        if X.shape[1] > 3:
            pca = PCA(n_components=3)
            X_reduced = pca.fit_transform(X)
            print(f"Reduced {X.shape[1]}D data to 3D using PCA")
        else:
            X_reduced = X
        
        # Create a mesh grid for the decision surface
        x_min, x_max = X_reduced[:, 0].min() - 0.5, X_reduced[:, 0].max() + 0.5
        y_min, y_max = X_reduced[:, 1].min() - 0.5, X_reduced[:, 1].max() + 0.5
        z_min, z_max = X_reduced[:, 2].min() - 0.5, X_reduced[:, 2].max() + 0.5
        
        xx, yy, zz = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution),
            np.linspace(z_min, z_max, resolution)
        )
        
        # Flatten the grid for prediction
        grid_points = np.c_[
            xx.ravel(), 
            yy.ravel(), 
            zz.ravel()
        ]
        
        # If we used PCA, we need to inverse transform to original space for prediction
        if X.shape[1] > 3:
            grid_points = pca.inverse_transform(grid_points)
        
        # Get predictions for each point in the grid
        if self.model is not None:
            try:
                # Try to get class probabilities
                Z = self._get_model_predictions(grid_points)
                
                # For binary classification, use the probability of the positive class
                if len(Z.shape) == 1 or Z.shape[1] == 1:
                    Z = Z.reshape(xx.shape)
                    is_binary = True
                else:
                    # For multi-class, get the most probable class
                    Z = np.argmax(Z, axis=1).reshape(xx.shape)
                    is_binary = False
            except Exception as e:
                print(f"Error getting predictions: {e}. Using random predictions.")
                Z = np.random.randint(0, 2, size=xx.shape)
                is_binary = True
        else:
            # No model provided, just show the data
            Z = np.ones(xx.shape) * y[0]
            is_binary = len(np.unique(y)) <= 2
        
        # Create the figure
        fig = go.Figure()
        
        # Add the decision surface
        if is_binary:
            # For binary classification, show a single surface with opacity based on probability
            fig.add_trace(go.Volume(
                x=xx.ravel(),
                y=yy.ravel(),
                z=zz.ravel(),
                value=Z.ravel(),
                isomin=0.3,
                isomax=0.7,
                opacity=opacity,
                surface_count=2,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(title='Class Probability'),
                name='Decision Boundary',
                hoverinfo='skip'
            ))
        else:
            # For multi-class, show an isosurface for each class
            for class_idx in range(len(np.unique(y))):
                # Create a binary mask for the current class
                class_mask = (Z == class_idx).astype(float)
                
                # Add the isosurface
                fig.add_trace(go.Volume(
                    x=xx.ravel(),
                    y=yy.ravel(),
                    z=zz.ravel(),
                    value=class_mask.ravel(),
                    isomin=0.5,
                    isomax=1.0,
                    opacity=opacity,
                    surface_count=1,
                    colorscale=colorscale,
                    showscale=False,
                    name=f'Class {class_idx} Boundary',
                    hoverinfo='skip',
                    showlegend=True
                ))
        
        # Add the data points
        if show_points:
            unique_classes = np.unique(y)
            colors = px.colors.qualitative.Plotly
            
            for i, class_idx in enumerate(unique_classes):
                class_mask = (y == class_idx)
                class_name = self.class_names[class_idx] if self.class_names else f'Class {class_idx}'
                
                fig.add_trace(go.Scatter3d(
                    x=X_reduced[class_mask, 0],
                    y=X_reduced[class_mask, 1],
                    z=X_reduced[class_mask, 2],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=colors[i % len(colors)],
                        opacity=0.8,
                        line=dict(width=1, color='DarkSlateGrey')
                    ),
                    name=class_name,
                    hovertemplate=(
                        f'<b>{class_name}</b><br>' +
                        f'{self.feature_names[0]}: %{{x:.2f}<br>' +
                        f'{self.feature_names[1]}: %{{y:.2f}<br>' +
                        f'{self.feature_names[2]}: %{{z:.2f}<extra></extra>}'
                    )
                ))
        
        # Update layout
        title = title or '3D Decision Boundary Visualization'
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=self.feature_names[0],
                yaxis_title=self.feature_names[1],
                zaxis_title=self.feature_names[2] if len(self.feature_names) > 2 else 'Feature 3',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            height=700,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        self.figures['decision_boundary_3d'] = fig
        return fig
    
    def plot_probability_surface(self, X: np.ndarray, y: np.ndarray, 
                               class_idx: int = 0, resolution: int = 50,
                               opacity: float = 0.7, colorscale: str = 'Viridis',
                               title: str = None) -> go.Figure:
        """Create a 3D surface plot of class probabilities.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target labels (n_samples,)
            class_idx: Index of the class to visualize probabilities for
            resolution: Resolution of the probability surface grid
            opacity: Opacity of the probability surface
            colorscale: Colorscale for the probability surface
            title: Plot title
            
        Returns:
            plotly.graph_objects.Figure: 3D probability surface plot
        """
        if X.shape[1] < 2:
            raise ValueError("At least 2 features are required for 3D visualization")
            
        # If more than 2 features, reduce to 2D using PCA
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_reduced = pca.fit_transform(X)
            print(f"Reduced {X.shape[1]}D data to 2D using PCA")
        else:
            X_reduced = X
        
        # Create a mesh grid for the probability surface
        x_min, x_max = X_reduced[:, 0].min() - 0.5, X_reduced[:, 0].max() + 0.5
        y_min, y_max = X_reduced[:, 1].min() - 0.5, X_reduced[:, 1].max() + 0.5
        
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )
        
        # Flatten the grid for prediction
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        # If we used PCA, we need to pad with zeros for the other dimensions
        if X.shape[1] > 2:
            # Pad with zeros for the remaining dimensions
            padding = np.zeros((grid_points.shape[0], X.shape[1] - 2))
            grid_points = np.hstack([grid_points, padding])
            
            # Inverse transform to original space for prediction
            grid_points = pca.inverse_transform(grid_points)
        
        # Get predictions for each point in the grid
        if self.model is not None:
            try:
                # Get class probabilities
                Z = self._get_model_predictions(grid_points)
                
                # If binary classification and only one probability is returned
                if len(Z.shape) == 1 or Z.shape[1] == 1:
                    prob = Z.reshape(xx.shape)
                    if class_idx == 1 and len(Z.shape) > 1:
                        prob = 1 - prob  # For binary, class 1 is the complement of class 0
                else:
                    # For multi-class, get the probability of the specified class
                    prob = Z[:, class_idx].reshape(xx.shape)
            except Exception as e:
                print(f"Error getting predictions: {e}. Using random probabilities.")
                prob = np.random.rand(*xx.shape)
        else:
            # No model provided, use random probabilities
            prob = np.random.rand(*xx.shape)
        
        # Create the figure
        fig = go.Figure()
        
        # Add the probability surface
        fig.add_trace(go.Surface(
            x=xx,
            y=yy,
            z=prob,
            colorscale=colorscale,
            opacity=opacity,
            showscale=True,
            colorbar=dict(title='Probability'),
            name=f'P(Class {class_idx})',
            hovertemplate=(
                f'<b>P(Class {class_idx})</b><br>' +
                f'{self.feature_names[0]}: %{{x:.2f}<br>' +
                f'{self.feature_names[1]}: %{{y:.2f}<br>' +
                f'Probability: %{{z:.4f}<extra></extra>}'
            )
        ))
        
        # Add the data points
        unique_classes = np.unique(y)
        colors = px.colors.qualitative.Plotly
        
        for i, cls in enumerate(unique_classes):
            class_mask = (y == cls)
            class_name = self.class_names[cls] if self.class_names else f'Class {cls}'
            
            # Get the probability for each point
            if self.model is not None:
                point_probs = self._get_model_predictions(X[class_mask])
                if len(point_probs.shape) > 1:
                    point_probs = point_probs[:, class_idx]
                z_values = point_probs
            else:
                z_values = np.zeros(np.sum(class_mask))
            
            fig.add_trace(go.Scatter3d(
                x=X_reduced[class_mask, 0],
                y=X_reduced[class_mask, 1],
                z=z_values,
                mode='markers',
                marker=dict(
                    size=5,
                    color=colors[i % len(colors)],
                    opacity=0.8,
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                name=f'{class_name} (Data)',
                hovertemplate=(
                    f'<b>{class_name}</b><br>' +
                    f'{self.feature_names[0]}: %{{x:.2f}<br>' +
                    f'{self.feature_names[1]}: %{{y:.2f}<br>' +
                    f'P(Class {class_idx}): %{{z:.4f}<extra></extra>'
                )
            ))
        
        # Update layout
        class_name = self.class_names[class_idx] if self.class_names else f'Class {class_idx}'
        title = title or f'3D Probability Surface for {class_name}'
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=self.feature_names[0],
                yaxis_title=self.feature_names[1],
                zaxis_title=f'P({class_name})',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.0)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            height=700,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        self.figures[f'probability_surface_class_{class_idx}'] = fig
        return fig
    
    def plot_decision_regions_3d(self, X: np.ndarray, y: np.ndarray, 
                               resolution: int = 100, opacity: float = 0.5,
                               colorscale: str = 'Viridis', 
                               show_points: bool = True,
                               title: str = None) -> go.Figure:
        """Create a 3D plot of decision regions with semi-transparent surfaces.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target labels (n_samples,)
            resolution: Resolution of the decision surface grid
            opacity: Opacity of the decision surfaces
            colorscale: Colorscale for the decision surfaces
            show_points: Whether to show the data points
            title: Plot title
            
        Returns:
            plotly.graph_objects.Figure: 3D plot of decision regions
        """
        if X.shape[1] < 3:
            raise ValueError("At least 3 features are required for 3D decision regions")
            
        # If more than 3 features, reduce to 3D using PCA
        if X.shape[1] > 3:
            pca = PCA(n_components=3)
            X_reduced = pca.fit_transform(X)
            print(f"Reduced {X.shape[1]}D data to 3D using PCA")
        else:
            X_reduced = X
        
        # Create a mesh grid for the decision surface
        x_min, x_max = X_reduced[:, 0].min() - 0.5, X_reduced[:, 0].max() + 0.5
        y_min, y_max = X_reduced[:, 1].min() - 0.5, X_reduced[:, 1].max() + 0.5
        z_min, z_max = X_reduced[:, 2].min() - 0.5, X_reduced[:, 2].max() + 0.5
        
        xx, yy, zz = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution),
            np.linspace(z_min, z_max, resolution)
        )
        
        # Flatten the grid for prediction
        grid_points = np.c_[
            xx.ravel(), 
            yy.ravel(), 
            zz.ravel()
        ]
        
        # If we used PCA, we need to inverse transform to original space for prediction
        if X.shape[1] > 3:
            grid_points = pca.inverse_transform(grid_points)
        
        # Get predictions for each point in the grid
        if self.model is not None:
            try:
                # Get class predictions
                Z = self._get_model_predictions(grid_points)
                
                # For binary classification, convert to 0/1
                if len(Z.shape) == 1 or Z.shape[1] == 1:
                    Z = (Z > 0.5).astype(int).flatten()
                else:
                    # For multi-class, get the most probable class
                    Z = np.argmax(Z, axis=1)
            except Exception as e:
                print(f"Error getting predictions: {e}. Using random predictions.")
                Z = np.random.randint(0, 2, size=grid_points.shape[0])
        else:
            # No model provided, just use the first class
            Z = np.zeros(grid_points.shape[0])
        
        # Reshape the predictions to match the grid
        Z = Z.reshape(xx.shape)
        
        # Create the figure
        fig = go.Figure()
        
        # Get unique classes and colors
        unique_classes = np.unique(y)
        colors = px.colors.qualitative.Plotly
        
        # For each class, create an isosurface
        for i, class_idx in enumerate(unique_classes):
            # Create a binary mask for the current class
            class_mask = (Z == class_idx).astype(float)
            
            # Skip if no points in this class
            if np.sum(class_mask) == 0:
                continue
            
            # Add the isosurface
            fig.add_trace(go.Volume(
                x=xx.ravel(),
                y=yy.ravel(),
                z=zz.ravel(),
                value=class_mask.ravel(),
                isomin=0.5,
                isomax=1.0,
                opacity=opacity,
                surface_count=1,
                colorscale=[colors[i % len(colors)], colors[i % len(colors)]],
                showscale=False,
                name=f'Class {class_idx} Region',
                hoverinfo='skip',
                showlegend=True
            ))
        
        # Add the data points
        if show_points:
            for i, class_idx in enumerate(unique_classes):
                class_mask = (y == class_idx)
                class_name = self.class_names[class_idx] if self.class_names else f'Class {class_idx}'
                
                fig.add_trace(go.Scatter3d(
                    x=X_reduced[class_mask, 0],
                    y=X_reduced[class_mask, 1],
                    z=X_reduced[class_mask, 2],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=colors[i % len(colors)],
                        opacity=0.8,
                        line=dict(width=1, color='DarkSlateGrey')
                    ),
                    name=class_name,
                    hovertemplate=(
                        f'<b>{class_name}</b><br>' +
                        f'{self.feature_names[0]}: %{{x:.2f}<br>' +
                        f'{self.feature_names[1]}: %{{y:.2f}<br>' +
                        f'{self.feature_names[2]}: %{{z:.2f}<extra></extra>}'
                    )
                ))
        
        # Update layout
        title = title or '3D Decision Regions'
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=self.feature_names[0],
                yaxis_title=self.feature_names[1],
                zaxis_title=self.feature_names[2] if len(self.feature_names) > 2 else 'Feature 3',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            height=700,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        self.figures['decision_regions_3d'] = fig
        return fig
    
    def plot_feature_importance_3d(self, feature_importances: np.ndarray,
                                 feature_names: List[str] = None,
                                 top_n: int = 10) -> go.Figure:
        """Create a 3D bar plot of feature importances.
        
        Args:
            feature_importances: Array of feature importances
            feature_names: List of feature names
            top_n: Number of top features to show
            
        Returns:
            plotly.graph_objects.Figure: 3D bar plot of feature importances
        """
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(feature_importances))]
        
        # Sort features by importance
        sorted_idx = np.argsort(feature_importances)[::-1]
        top_indices = sorted_idx[:top_n]
        
        # Get top features and their importances
        top_features = [feature_names[i] for i in top_indices]
        top_importances = feature_importances[top_indices]
        
        # Create 3D bar plot
        fig = go.Figure(data=[
            go.Bar3d(
                x=np.arange(len(top_features)),  # Feature index on x-axis
                y=np.zeros(len(top_features)),   # All bars at y=0
                z=np.zeros(len(top_features)),   # All bars start at z=0
                dx=0.8,                         # Bar width
                dy=0.8,                         # Bar depth
                dz=top_importances,             # Bar height (importance)
                text=top_features,              # Hover text
                hovertemplate='<b>%{text}</b><br>Importance: %{z:.4f}<extra></extra>',
                marker=dict(
                    color=top_importances,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Importance')
                )
            )
        ])
        
        # Update layout
        fig.update_layout(
            title='3D Feature Importance',
            scene=dict(
                xaxis=dict(
                    title='Feature',
                    ticktext=top_features,
                    tickvals=np.arange(len(top_features)),
                    tickangle=-45,
                    tickfont=dict(size=10)
                ),
                yaxis=dict(showticklabels=False, title=''),
                zaxis=dict(title='Importance'),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.0)
                )
            ),
            margin=dict(l=0, r=0, b=100, t=40),
            height=700
        )
        
        self.figures['feature_importance_3d'] = fig
        return fig
    
    def plot_confusion_matrix_3d(self, y_true: np.ndarray, y_pred: np.ndarray,
                               normalize: bool = False,
                               title: str = None) -> go.Figure:
        """Create a 3D surface plot of a confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Whether to normalize the confusion matrix
            title: Plot title
            
        Returns:
            plotly.graph_objects.Figure: 3D surface plot of confusion matrix
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            z_label = 'Normalized Count'
        else:
            z_label = 'Count'
        
        # Get class names
        if self.class_names is not None and len(self.class_names) == cm.shape[0]:
            class_names = self.class_names
        else:
            class_names = [f'Class {i}' for i in range(cm.shape[0])]
        
        # Create meshgrid for 3D surface
        x = np.arange(cm.shape[0] + 1)
        y = np.arange(cm.shape[1] + 1)
        X, Y = np.meshgrid(x, y)
        
        # Pad the confusion matrix for proper surface plotting
        Z = np.zeros((cm.shape[0] + 1, cm.shape[1] + 1))
        Z[:-1, :-1] = cm
        
        # Create the figure
        fig = go.Figure(data=[
            go.Surface(
                z=Z,
                x=X,
                y=Y,
                colorscale='Viridis',
                opacity=0.8,
                hoverinfo='text',
                text=[[
                    f'True: {class_names[int(x)]}<br>Pred: {class_names[int(y)]}<br>Count: {z:.2f}'
                    for x, y, z in zip(x_row, y_row, z_row)
                ] for x_row, y_row, z_row in zip(X, Y, Z)]
            )
        ])
        
        # Update layout
        title = title or ('Normalized Confusion Matrix' if normalize else 'Confusion Matrix')
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(
                    title='Predicted',
                    ticktext=class_names,
                    tickvals=np.arange(len(class_names)) + 0.5,
                    tickangle=-45
                ),
                yaxis=dict(
                    title='True',
                    ticktext=class_names,
                    tickvals=np.arange(len(class_names)) + 0.5,
                    tickangle=0
                ),
                zaxis_title=z_label,
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.0)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            height=700
        )
        
        self.figures['confusion_matrix_3d'] = fig
        return fig
    
    def plot_roc_auc_3d(self, fpr: Dict, tpr: Dict, roc_auc: Dict,
                       title: str = '3D ROC Curves') -> go.Figure:
        """Create a 3D ROC curve visualization.
        
        Args:
            fpr: Dictionary of false positive rates for each class
            tpr: Dictionary of true positive rates for each class
            roc_auc: Dictionary of AUC scores for each class
            title: Plot title
            
        Returns:
            plotly.graph_objects.Figure: 3D ROC curve visualization
        """
        if not fpr or not tpr or not roc_auc:
            raise ValueError("fpr, tpr, and roc_auc must be non-empty dictionaries")
        
        # Create the figure
        fig = go.Figure()
        
        # Add ROC curve for each class
        for i, (class_idx, class_fpr) in enumerate(fpr.items()):
            class_tpr = tpr[class_idx]
            class_auc = roc_auc[class_idx]
            
            # Get class name
            if self.class_names is not None and class_idx < len(self.class_names):
                class_name = self.class_names[class_idx]
            else:
                class_name = f'Class {class_idx}'
            
            # Add ROC curve
            fig.add_trace(go.Scatter3d(
                x=class_fpr,
                y=class_tpr,
                z=np.ones_like(class_fpr) * class_idx,
                mode='lines',
                line=dict(width=3),
                name=f'{class_name} (AUC = {class_auc:.2f})',
                hovertemplate=(
                    f'<b>{class_name}</b><br>' +
                    'FPR: %{x:.3f}<br>' +
                    'TPR: %{y:.3f}<br>' +
                    f'AUC: {class_auc:.3f}<extra></extra>'
                )
            ))
            
            # Add a point at the optimal threshold (Youden's J statistic)
            if len(class_fpr) > 0 and len(class_tpr) > 0:
                j_scores = class_tpr - class_fpr
                optimal_idx = np.argmax(j_scores)
                
                fig.add_trace(go.Scatter3d(
                    x=[class_fpr[optimal_idx]],
                    y=[class_tpr[optimal_idx]],
                    z=[class_idx],
                    mode='markers',
                    marker=dict(size=8, symbol='x'),
                    name=f'Optimal {class_name}',
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Add a diagonal reference line (random classifier)
        x_range = np.linspace(0, 1, 10)
        for class_idx in fpr.keys():
            fig.add_trace(go.Scatter3d(
                x=x_range,
                y=x_range,
                z=np.ones_like(x_range) * class_idx,
                mode='lines',
                line=dict(dash='dash', color='gray', width=1),
                showlegend=False,
                hoverinfo='skip',
                opacity=0.5
            ))
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(
                    title='False Positive Rate',
                    range=[0, 1],
                    tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0]
                ),
                yaxis=dict(
                    title='True Positive Rate',
                    range=[0, 1],
                    tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0]
                ),
                zaxis=dict(
                    title='Class',
                    ticktext=[f'Class {i}' for i in fpr.keys()],
                    tickvals=list(fpr.keys())
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=0.8)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            height=700,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        self.figures['roc_auc_3d'] = fig
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
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_curve, auc
    
    # Generate sample data
    X, y = make_classification(
        n_samples=500, n_features=5, n_informative=3, 
        n_redundant=1, n_classes=3, n_clusters_per_class=1,
        random_state=42
    )
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train a classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Create visualizer
    visualizer = DecisionBoundaryVisualizer(
        model=clf,
        feature_names=[f'Feature {i+1}' for i in range(X.shape[1])],
        class_names=[f'Class {i}' for i in np.unique(y)]
    )
    
    # Plot 3D decision boundary (using first 3 features)
    fig1 = visualizer.plot_3d_decision_boundary(
        X_test[:, :3], y_test, 
        resolution=30, 
        title='3D Decision Boundary (First 3 Features)'
    )
    
    # Plot probability surface for class 0
    fig2 = visualizer.plot_probability_surface(
        X_test, y_test, 
        class_idx=0,
        title='Probability Surface for Class 0 (First 2 Features)'
    )
    
    # Plot 3D decision regions
    fig3 = visualizer.plot_decision_regions_3d(
        X_test, y_test,
        resolution=20,
        title='3D Decision Regions (First 3 Features)'
    )
    
    # Plot feature importance
    if hasattr(clf, 'feature_importances_'):
        fig4 = visualizer.plot_feature_importance_3d(
            clf.feature_importances_,
            top_n=5
        )
    
    # Plot confusion matrix
    y_pred = clf.predict(X_test)
    fig5 = visualizer.plot_confusion_matrix_3d(
        y_test, y_pred,
        normalize=True,
        title='Normalized Confusion Matrix'
    )
    
    # Plot ROC curves (for binary or multi-class)
    if len(np.unique(y)) == 2:
        # Binary classification
        y_proba = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        fig6 = visualizer.plot_roc_auc_3d(
            {0: fpr}, {0: tpr}, {0: roc_auc},
            title='3D ROC Curve (Binary Classification)'
        )
    else:
        # Multi-class classification
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        # Compute ROC curve and ROC area for each class
        y_score = clf.predict_proba(X_test)
        n_classes = len(np.unique(y))
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        fig6 = visualizer.plot_roc_auc_3d(
            fpr, tpr, roc_auc,
            title='3D ROC Curves (Multi-Class)'
        )
    
    # Show all figures
    visualizer.show()
