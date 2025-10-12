#!/usr/bin/env python3
"""
Research Paper Figure Generator
Creates all visualizations for the NeuroGenesis research paper.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tensorflow as tf
from sklearn.decomposition import PCA
import os

# Create figures directory if it doesn't exist
os.makedirs("figures", exist_ok=True)

def create_loss_landscape_figure():
    """Create Figure 1: 3D Loss Landscape"""
    print("📈 Creating 3D Loss Landscape...")

    # Create a synthetic loss landscape
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)

    # Create a complex loss surface with multiple minima
    Z = (X**2 + Y**2) + 0.5 * np.sin(X) * np.cos(Y) + 0.1 * np.random.randn(50, 50)

    fig = go.Figure(data=[
        go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Viridis',
            opacity=0.8,
            showscale=True,
            colorbar=dict(title="Loss Value"),
            hovertemplate=(
                '<b>X</b>: %{x:.2f}<br>'
                '<b>Y</b>: %{y:.2f}<br>'
                '<b>Loss</b>: %{z:.3f}<extra></extra>'
            )
        )
    ])

    # Add optimization path
    path_x = np.linspace(-2, 2, 20)
    path_y = np.linspace(-1.5, 1.5, 20)
    path_z = path_x**2 + path_y**2 + 0.5 * np.sin(path_x) * np.cos(path_y)

    fig.add_trace(go.Scatter3d(
        x=path_x, y=path_y, z=path_z,
        mode='lines+markers',
        line=dict(color='red', width=4),
        marker=dict(size=4, color='red'),
        name='Optimization Path'
    ))

    fig.update_layout(
        title="3D Loss Landscape with Optimization Trajectory",
        scene=dict(
            xaxis_title="Parameter 1",
            yaxis_title="Parameter 2",
            zaxis_title="Loss",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        height=700,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.write_html("figures/loss_landscape_3d.html")
    fig.write_image("figures/loss_landscape_3d.png", width=1200, height=800)
    print("✅ Loss landscape figure created")

def create_metric_correlation_figure():
    """Create Figure 2: Metric Correlation Heatmap"""
    print("📊 Creating Metric Correlation Heatmap...")

    # Simulate realistic training metrics
    np.random.seed(42)
    epochs = 100

    metrics = {
        'loss': np.exp(-np.linspace(0.1, 2, epochs)) + np.random.normal(0, 0.01, epochs),
        'accuracy': 1 - np.exp(-np.linspace(0.05, 1.5, epochs)) + np.random.normal(0, 0.005, epochs),
        'val_loss': np.exp(-np.linspace(0.08, 1.8, epochs)) + np.random.normal(0, 0.015, epochs),
        'val_accuracy': 1 - np.exp(-np.linspace(0.04, 1.3, epochs)) + np.random.normal(0, 0.008, epochs),
        'f1_score': 1 - np.exp(-np.linspace(0.06, 1.4, epochs)) + np.random.normal(0, 0.007, epochs)
    }

    # Create correlation matrix
    data = np.array([metrics[metric] for metric in metrics.keys()])
    correlation_matrix = np.corrcoef(data)

    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=list(metrics.keys()),
        y=list(metrics.keys()),
        colorscale='RdBu',
        zmin=-1, zmax=1,
        text=np.round(correlation_matrix, 3),
        texttemplate="%{text}",
        textfont={"size":12},
        hovertemplate=(
            '<b>%{x}</b> vs <b>%{y}</b><br>'
            '<b>Correlation</b>: %{z:.3f}<extra></extra>'
        )
    ))

    fig.update_layout(
        title="Training Metrics Correlation Analysis",
        height=600,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.write_html("figures/metric_correlation.html")
    fig.write_image("figures/metric_correlation.png", width=800, height=600)
    print("✅ Metric correlation figure created")

def create_training_dynamics_figure():
    """Create Figure 3: Training Dynamics 3D"""
    print("🎯 Creating Training Dynamics 3D...")

    # Create training trajectory data
    epochs = np.arange(1, 51)
    loss = np.exp(-epochs/20) + 0.1 * np.random.randn(50)
    accuracy = 1 - np.exp(-epochs/15) + 0.05 * np.random.randn(50)

    # Compute derivatives (momentum)
    loss_deriv = np.gradient(loss)
    acc_deriv = np.gradient(accuracy)

    fig = go.Figure()

    # Main trajectory
    fig.add_trace(go.Scatter3d(
        x=epochs,
        y=loss,
        z=accuracy,
        mode='lines+markers',
        line=dict(color='blue', width=4),
        marker=dict(size=3, opacity=0.8),
        name='Training Trajectory'
    ))

    # Momentum vectors
    for i in range(0, len(epochs)-1, 5):
        fig.add_trace(go.Scatter3d(
            x=[epochs[i], epochs[i] + loss_deriv[i]*10],
            y=[loss[i], loss[i]],
            z=[accuracy[i], accuracy[i] + acc_deriv[i]*10],
            mode='lines',
            line=dict(color='red', width=2),
            showlegend=False
        ))

    fig.update_layout(
        title="3D Training Dynamics with Momentum Analysis",
        scene=dict(
            xaxis_title='Epoch',
            yaxis_title='Loss',
            zaxis_title='Accuracy',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        height=700,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.write_html("figures/training_dynamics_3d.html")
    fig.write_image("figures/training_dynamics_3d.png", width=1000, height=700)
    print("✅ Training dynamics figure created")

def create_hyperparameter_surface_figure():
    """Create Figure 4: Hyperparameter Optimization Surface"""
    print("🔧 Creating Hyperparameter Surface...")

    # Create grid search results
    learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    batch_sizes = [16, 32, 64, 128, 256]

    # Simulate results (higher LR and smaller batch usually better up to a point)
    results = []
    for lr in learning_rates:
        for bs in batch_sizes:
            # Simulate validation accuracy based on hyperparameters
            base_acc = 0.85
            lr_effect = 0.1 * np.log10(lr) if lr > 1e-4 else 0.05
            bs_effect = 0.05 * (1 - bs/256)  # Smaller batches usually better
            noise = np.random.normal(0, 0.02)

            val_acc = base_acc + lr_effect + bs_effect + noise
            results.append({'lr': lr, 'batch_size': bs, 'val_accuracy': max(0.5, min(0.95, val_acc))})

    # Extract data for surface
    X = [r['lr'] for r in results]
    Y = [r['batch_size'] for r in results]
    Z = [r['val_accuracy'] for r in results]

    # Create surface
    x_grid = np.array(sorted(set(X)))
    y_grid = np.array(sorted(set(Y)))
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    # Interpolate for surface (simplified)
    Z_surface = np.zeros_like(X_grid)
    for i in range(len(x_grid)):
        for j in range(len(y_grid)):
            # Find nearest neighbors and interpolate
            distances = []
            values = []
            for k in range(len(X)):
                dist = (np.log10(X[k]) - np.log10(x_grid[i]))**2 + (Y[k] - y_grid[j])**2
                if dist < 1.0:  # Within reasonable distance
                    distances.append(dist)
                    values.append(Z[k])

            if values:
                weights = 1 / (np.array(distances) + 1e-6)
                Z_surface[j, i] = np.average(values, weights=weights)

    fig = go.Figure(data=[
        go.Surface(
            x=X_grid, y=Y_grid, z=Z_surface,
            colorscale='Viridis',
            opacity=0.7,
            colorbar=dict(title="Validation Accuracy"),
            hovertemplate=(
                '<b>Learning Rate</b>: %{x:.2e}<br>'
                '<b>Batch Size</b>: %{y}<br>'
                '<b>Accuracy</b>: %{z:.3f}<extra></extra>'
            )
        )
    ])

    # Add scatter points
    fig.add_trace(go.Scatter3d(
        x=X, y=Y, z=Z,
        mode='markers',
        marker=dict(
            size=6,
            color=Z,
            colorscale='Viridis',
            showscale=False,
            line=dict(width=1, color='black')
        ),
        name='Experiments',
        hovertemplate=(
            '<b>Learning Rate</b>: %{x:.2e}<br>'
            '<b>Batch Size</b>: %{y}<br>'
            '<b>Accuracy</b>: %{z:.3f}<extra></extra>'
        )
    ))

    fig.update_layout(
        title="Hyperparameter Optimization Surface: Learning Rate vs Batch Size",
        scene=dict(
            xaxis_title="Learning Rate",
            yaxis_title="Batch Size",
            zaxis_title="Validation Accuracy",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        height=700,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.write_html("figures/hyperparameter_surface.html")
    fig.write_image("figures/hyperparameter_surface.png", width=1000, height=700)
    print("✅ Hyperparameter surface figure created")

def create_model_complexity_figure():
    """Create Figure 5: Model Complexity Analysis"""
    print("🎛️ Creating Model Complexity Analysis...")

    # Create sample model complexity data
    models = ['Model A', 'Model B', 'Model C', 'Model D']
    metrics = ['Layer Count', 'Parameter Count', 'Memory Usage (MB)', 'Training Time (s)']

    # Simulated complexity data
    complexity_data = np.array([
        [5, 1000, 4.2, 120],    # Model A
        [8, 2500, 10.5, 180],   # Model B
        [12, 5000, 21.0, 300],  # Model C
        [20, 10000, 42.0, 600]  # Model D
    ])

    # Normalize for radar chart
    normalized_data = complexity_data / complexity_data.max(axis=0)

    fig = go.Figure()

    colors = ['blue', 'red', 'green', 'orange']
    for i, model in enumerate(models):
        fig.add_trace(go.Scatterpolar(
            r=np.concatenate([normalized_data[i], [normalized_data[i, 0]]]),
            theta=metrics + [metrics[0]],
            fill='toself',
            name=model,
            line=dict(color=colors[i], width=3)
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1.2],
                tickangle=0,
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                tickfont=dict(size=12)
            )
        ),
        title="Model Complexity Analysis Across Different Architectures",
        height=600,
        margin=dict(l=80, r=80, t=80, b=80)
    )

    fig.write_html("figures/model_complexity.html")
    fig.write_image("figures/model_complexity.png", width=800, height=600)
    print("✅ Model complexity figure created")

def create_convergence_analysis_figure():
    """Create Figure 6: Training Convergence Analysis"""
    print("📈 Creating Convergence Analysis...")

    # Create subplots for convergence analysis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Training Curves", "Convergence Rate",
            "Gradient Magnitude", "Learning Rate Schedule"
        ),
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}]]
    )

    epochs = np.arange(1, 101)

    # Training curves
    loss = np.exp(-epochs/30) + 0.05 * np.random.randn(100)
    accuracy = 1 - np.exp(-epochs/25) + 0.02 * np.random.randn(100)

    fig.add_trace(
        go.Scatter(x=epochs, y=loss, mode='lines', name='Training Loss', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=accuracy, mode='lines', name='Training Accuracy', line=dict(color='green')),
        row=1, col=1
    )

    # Convergence rate (second derivative)
    second_deriv = np.gradient(np.gradient(loss))
    convergence_rate = np.abs(second_deriv)

    fig.add_trace(
        go.Scatter(x=epochs, y=convergence_rate, mode='lines', name='Convergence Rate'),
        row=1, col=2
    )
    fig.add_hline(y=1e-4, line=dict(color="red", dash="dash"), row=1, col=2)

    # Gradient magnitude
    gradient = np.abs(np.gradient(loss))
    fig.add_trace(
        go.Scatter(x=epochs, y=gradient, mode='lines', name='Gradient Magnitude'),
        row=2, col=1
    )

    # Learning rate schedule
    lr_schedule = np.logspace(-3, -5, 100)
    fig.add_trace(
        go.Scatter(x=epochs, y=lr_schedule, mode='lines', name='Learning Rate', line=dict(color='purple')),
        row=2, col=2
    )

    fig.update_layout(
        title="Comprehensive Training Convergence Analysis",
        height=800,
        showlegend=True
    )

    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=2)

    fig.write_html("figures/convergence_analysis.html")
    fig.write_image("figures/convergence_analysis.png", width=1200, height=800)
    print("✅ Convergence analysis figure created")

def create_attention_evolution_figure():
    """Create Figure 7: Attention Evolution"""
    print("🎭 Creating Attention Evolution...")

    # Create attention evolution data
    seq_len = 8
    time_steps = 12
    attention_weights = []

    for t in range(time_steps):
        # Create evolving attention pattern
        attn = np.eye(seq_len) * 0.1
        focus_pos = (t * 2) % seq_len
        attn[focus_pos, :] = 0.7
        attn[:, focus_pos] = 0.7
        attn = attn / attn.sum(axis=1, keepdims=True)
        attention_weights.append(attn)

    fig = go.Figure()

    # Create frames for animation
    frames = []
    for i, attn in enumerate(attention_weights):
        frames.append(go.Frame(
            data=[go.Heatmap(
                z=attn,
                colorscale='Viridis',
                showscale=i==0,
                colorbar=dict(title="Attention Weight") if i == 0 else None
            )],
            name=f'frame_{i}'
        ))

    # Initial frame
    fig.add_trace(go.Heatmap(
        z=attention_weights[0],
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Attention Weight"),
        hovertemplate='<b>Query</b>: %{x}<br><b>Key</b>: %{y}<br><b>Weight</b>: %{z:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title="Attention Mechanism Evolution Over Training",
        xaxis_title="Key Position",
        yaxis_title="Query Position",
        height=600,
        margin=dict(l=0, r=0, b=0, t=40),
        updatemenus=[{
            'type': 'buttons',
            'buttons': [
                {
                    'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}],
                    'label': '▶️ Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}],
                    'label': '⏸️ Pause',
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
    )

    fig.frames = frames

    fig.write_html("figures/attention_evolution.html")
    fig.write_image("figures/attention_evolution.png", width=800, height=600)
    print("✅ Attention evolution figure created")

def main():
    """Generate all research paper figures."""
    print("🎨 Starting research paper figure generation...")
    print("=" * 60)

    # Create all figures
    create_loss_landscape_figure()
    create_metric_correlation_figure()
    create_training_dynamics_figure()
    create_hyperparameter_surface_figure()
    create_model_complexity_figure()
    create_convergence_analysis_figure()
    create_attention_evolution_figure()

    print("=" * 60)
    print("🎉 All research paper figures generated successfully!")
    print(f"📁 Figures saved in: {os.path.abspath('figures/')}")
    print("📄 Research paper template: research_paper.md")

    # List all created files
    print("\n📋 Generated files:")
    for filename in sorted(os.listdir("figures/")):
        filepath = os.path.join("figures", filename)
        size = os.path.getsize(filepath) / 1024  # KB
        print(f"   • {filename} ({size:.1f} KB)")

if __name__ == "__main__":
    main()
