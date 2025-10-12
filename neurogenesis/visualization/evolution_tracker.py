"""
Evolution Tracker for NeuroGenesis

This module provides tools for tracking and visualizing the evolution of neural networks
over time, including architecture changes, performance metrics, and other statistics.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import json
import os
from datetime import datetime
import tensorflow as tf
from collections import defaultdict

class EvolutionTracker:
    """Class for tracking and visualizing the evolution of neural networks."""
    
    def __init__(self, log_dir: str = None):
        """Initialize the EvolutionTracker.
        
        Args:
            log_dir: Directory to save logs and visualizations. If None, uses 'logs/evolution_<timestamp>'
        """
        if log_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.log_dir = os.path.join("logs", f"evolution_{timestamp}")
        else:
            self.log_dir = log_dir
            
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize data storage
        self.generations = []
        self.population = []
        self.metrics = defaultdict(list)
        self.architectures = []
        self.metadata = {}
        self.current_generation = 0
        
        # Create a log file
        self.log_file = os.path.join(self.log_dir, "evolution_log.json")
        self._init_log_file()
    
    def _init_log_file(self):
        """Initialize the log file with an empty structure."""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                json.dump({
                    'metadata': {},
                    'generations': [],
                    'best_models': []
                }, f, indent=2)
    
    def log_generation(self, models: List[Any], metrics: Dict[str, List[float]],
                      generation: int = None, metadata: Dict = None):
        """Log information about a generation of models.
        
        Args:
            models: List of models in the current generation
            metrics: Dictionary of metrics for each model, with metric names as keys
                     and lists of values (one per model) as values
            generation: Generation number. If None, uses the next generation number
            metadata: Additional metadata to store with the generation
        """
        if generation is None:
            generation = self.current_generation
            self.current_generation += 1
        
        if metadata is None:
            metadata = {}
        
        # Store generation data
        gen_data = {
            'generation': generation,
            'timestamp': datetime.now().isoformat(),
            'num_models': len(models),
            'metrics': metrics,
            'metadata': metadata
        }
        
        # Extract and store architecture information
        architectures = []
        for i, model in enumerate(models):
            arch = self._extract_architecture(model)
            arch['model_id'] = f"gen{generation}_model{i}"
            architectures.append(arch)
        
        gen_data['architectures'] = architectures
        
        # Find the best model in this generation
        if 'fitness' in metrics and metrics['fitness']:
            best_idx = np.argmax(metrics['fitness'])
            best_model_data = {
                'generation': generation,
                'model_id': f"gen{generation}_model{best_idx}",
                'metrics': {k: v[best_idx] for k, v in metrics.items()},
                'architecture': architectures[best_idx]
            }
        else:
            best_model_data = None
        
        # Update in-memory data
        self.generations.append(gen_data)
        if best_model_data:
            self.population.append(best_model_data)
        
        # Update metrics
        for metric_name, values in metrics.items():
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []
            self.metrics[metric_name].append(np.max(values) if values else None)
        
        # Save to log file
        self._save_to_log(gen_data, best_model_data)
        
        return gen_data
    
    def _extract_architecture(self, model) -> Dict:
        """Extract architecture information from a model.
        
        Args:
            model: The model to extract architecture from
            
        Returns:
            Dictionary containing architecture information
        """
        if hasattr(model, 'get_config'):
            # Keras model
            try:
                config = model.get_config()
                return {
                    'type': 'keras',
                    'num_layers': len(model.layers),
                    'num_params': model.count_params(),
                    'config': config
                }
            except:
                pass
        
        # Fallback for non-Keras models
        return {
            'type': str(type(model).__name__),
            'str': str(model)
        }
    
    def _save_to_log(self, gen_data: Dict, best_model_data: Dict = None):
        """Save generation data to the log file.
        
        Args:
            gen_data: Generation data
            best_model_data: Best model data for this generation
        """
        # Read existing data
        try:
            with open(self.log_file, 'r') as f:
                log_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            log_data = {'generations': [], 'best_models': []}
        
        # Update data
        log_data['generations'].append(gen_data)
        if best_model_data:
            log_data['best_models'].append(best_model_data)
        
        # Save back to file
        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def plot_metric_evolution(self, metric_name: str = 'fitness', 
                            title: str = None) -> go.Figure:
        """Plot the evolution of a metric over generations.
        
        Args:
            metric_name: Name of the metric to plot
            title: Plot title. If None, generates a default title
            
        Returns:
            plotly.graph_objects.Figure: Line plot of metric evolution
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            raise ValueError(f"No data for metric '{metric_name}'. Available metrics: {list(self.metrics.keys())}")
        
        # Prepare data
        generations = list(range(len(self.metrics[metric_name])))
        values = self.metrics[metric_name]
        
        # Create figure
        fig = go.Figure()
        
        # Add main line
        fig.add_trace(go.Scatter(
            x=generations,
            y=values,
            mode='lines+markers',
            name=metric_name,
            line=dict(width=2),
            marker=dict(size=8)
        ))
        
        # Add rolling average
        window_size = max(1, len(values) // 10)  # 10% window size
        if len(values) > window_size:
            rolling_avg = pd.Series(values).rolling(window=window_size).mean()
            fig.add_trace(go.Scatter(
                x=generations,
                y=rolling_avg,
                mode='lines',
                name=f'{window_size}-gen avg',
                line=dict(dash='dash', width=2)
            ))
        
        # Update layout
        title = title or f'Evolution of {metric_name.capitalize()} Over Generations'
        fig.update_layout(
            title=title,
            xaxis_title='Generation',
            yaxis_title=metric_name.capitalize(),
            hovermode='x',
            height=500,
            margin=dict(l=50, r=50, b=50, t=50, pad=4),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def plot_architecture_evolution(self, metric_name: str = 'fitness') -> go.Figure:
        """Plot the evolution of model architecture over generations.
        
        Args:
            metric_name: Metric to use for coloring the points
            
        Returns:
            plotly.graph_objects.Figure: Scatter plot of architecture evolution
        """
        if not self.population:
            raise ValueError("No population data available. Call log_generation() first.")
        
        # Prepare data
        generations = []
        num_layers = []
        num_params = []
        metric_values = []
        
        for model_data in self.population:
            gen = model_data['generation']
            arch = model_data['architecture']
            
            generations.append(gen)
            num_layers.append(arch.get('num_layers', 0))
            num_params.append(arch.get('num_params', 0) / 1e6)  # Convert to millions
            
            # Get metric value if available
            if 'metrics' in model_data and metric_name in model_data['metrics']:
                metric_values.append(model_data['metrics'][metric_name])
            else:
                metric_values.append(None)
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=generations,
            y=num_layers,
            mode='markers',
            name='# Layers',
            marker=dict(
                size=10,
                color=num_params,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='# Params (M)')
            ),
            text=[f"Gen {g}<br>Layers: {l}<br>Params: {p:.2f}M" 
                 for g, l, p in zip(generations, num_layers, num_params)],
            hoverinfo='text'
        ))
        
        # Add trend lines
        if len(generations) > 1:
            # Add moving average for number of layers
            window_size = max(1, len(generations) // 5)
            layers_ma = pd.Series(num_layers).rolling(window=window_size).mean()
            
            fig.add_trace(go.Scatter(
                x=generations,
                y=layers_ma,
                mode='lines',
                name=f'{window_size}-gen avg layers',
                line=dict(dash='dash', width=2)
            ))
        
        # Update layout
        fig.update_layout(
            title='Model Architecture Evolution',
            xaxis_title='Generation',
            yaxis_title='Number of Layers',
            hovermode='closest',
            height=600,
            margin=dict(l=50, r=50, b=50, t=50, pad=4),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def plot_3d_evolution(self, x_metric: str = 'generation',
                         y_metric: str = 'num_layers',
                         z_metric: str = 'fitness',
                         color_metric: str = 'num_params',
                         title: str = None) -> go.Figure:
        """Create a 3D plot of the evolution trajectory.
        
        Args:
            x_metric: Metric to plot on the x-axis ('generation', 'num_layers', 'num_params', or a metric name)
            y_metric: Metric to plot on the y-axis
            z_metric: Metric to plot on the z-axis
            color_metric: Metric to use for coloring the points
            title: Plot title. If None, generates a default title
            
        Returns:
            plotly.graph_objects.Figure: 3D scatter plot of the evolution trajectory
        """
        if not self.population:
            raise ValueError("No population data available. Call log_generation() first.")
        
        # Prepare data
        data = []
        
        for model_data in self.population:
            gen = model_data['generation']
            arch = model_data['architecture']
            metrics = model_data.get('metrics', {})
            
            point = {
                'generation': gen,
                'num_layers': arch.get('num_layers', 0),
                'num_params': arch.get('num_params', 0) / 1e6,  # Convert to millions
                **metrics
            }
            
            data.append(point)
        
        df = pd.DataFrame(data)
        
        # Check if requested metrics exist
        for metric in [x_metric, y_metric, z_metric, color_metric]:
            if metric not in df.columns:
                raise ValueError(f"Metric '{metric}' not found in data. Available metrics: {df.columns.tolist()}")
        
        # Create figure
        fig = go.Figure()
        
        # Add 3D scatter plot
        fig.add_trace(go.Scatter3d(
            x=df[x_metric],
            y=df[y_metric],
            z=df[z_metric],
            mode='markers+lines',
            marker=dict(
                size=8,
                color=df[color_metric],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=color_metric.replace('_', ' ').title())
            ),
            line=dict(
                color='gray',
                width=1
            ),
            text=[f"Gen {g}" for g in df['generation']],
            hoverinfo='text',
            hovertext=[f"Gen {row['generation']}<br>" +
                      f"{x_metric}: {row[x_metric]:.2f}<br>" +
                      f"{y_metric}: {row[y_metric]:.2f}<br>" +
                      f"{z_metric}: {row[z_metric]:.2f}<br>" +
                      f"{color_metric}: {row[color_metric]:.2f}" 
                      for _, row in df.iterrows()],
            name='Evolution'
        ))
        
        # Add start and end markers
        if len(df) > 1:
            for i, (x, y, z) in enumerate([(df[x_metric].iloc[0], df[y_metric].iloc[0], df[z_metric].iloc[0]),
                                         (df[x_metric].iloc[-1], df[y_metric].iloc[-1], df[z_metric].iloc[-1])]):
                fig.add_trace(go.Scatter3d(
                    x=[x],
                    y=[y],
                    z=[z],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='red' if i == 0 else 'green',
                        symbol='diamond'
                    ),
                    name='Start' if i == 0 else 'End',
                    hoverinfo='text',
                    hovertext=f"{'Start' if i == 0 else 'End'} (Gen {df['generation'].iloc[0 if i == 0 else -1]})"
                ))
        
        # Update layout
        title = title or f'3D Evolution: {x_metric} vs {y_metric} vs {z_metric}'
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=x_metric.replace('_', ' ').title(),
                yaxis_title=y_metric.replace('_', ' ').title(),
                zaxis_title=z_metric.replace('_', ' ').title(),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.0)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            height=700
        )
        
        return fig
    
    def plot_parallel_coordinates(self, metrics: List[str] = None,
                                title: str = 'Parallel Coordinates Plot') -> go.Figure:
        """Create a parallel coordinates plot of model metrics across generations.
        
        Args:
            metrics: List of metrics to include in the plot. If None, uses all available metrics.
            title: Plot title
            
        Returns:
            plotly.graph_objects.Figure: Parallel coordinates plot
        """
        if not self.population:
            raise ValueError("No population data available. Call log_generation() first.")
        
        # Prepare data
        data = []
        
        for model_data in self.population:
            gen = model_data['generation']
            arch = model_data['architecture']
            metrics_data = model_data.get('metrics', {})
            
            point = {
                'Generation': gen,
                'Num Layers': arch.get('num_layers', 0),
                'Num Params (M)': arch.get('num_params', 0) / 1e6,  # Convert to millions
                **metrics_data
            }
            
            data.append(point)
        
        df = pd.DataFrame(data)
        
        # Use all numeric columns if metrics not specified
        if metrics is None:
            metrics = [col for col in df.columns if col not in ['Generation'] and pd.api.types.is_numeric_dtype(df[col])]
        
        # Check if requested metrics exist
        for metric in metrics:
            if metric not in df.columns:
                raise ValueError(f"Metric '{metric}' not found in data. Available metrics: {df.columns.tolist()}")
        
        # Create dimensions for the parallel coordinates plot
        dimensions = [dict(
            label='Generation',
            values=df['Generation'],
            tickvals=df['Generation'].unique(),
            ticktext=[f'Gen {g}' for g in df['Generation'].unique()]
        )]
        
        for metric in metrics:
            dimensions.append(dict(
                label=metric.replace('_', ' ').title(),
                values=df[metric]
            ))
        
        # Create figure
        fig = go.Figure(data=go.Parcoords(
            line=dict(
                color=df['Generation'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Generation')
            ),
            dimensions=dimensions,
            labelangle=-45,
            labelside='bottom'
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            height=600,
            margin=dict(l=50, r=50, b=100, t=50, pad=4)
        )
        
        return fig
    
    def save_visualization(self, fig: go.Figure, filename: str, 
                          format: str = 'html', width: int = None, 
                          height: int = None):
        """Save a visualization to a file.
        
        Args:
            fig: The figure to save
            filename: Output filename (without extension)
            format: Output format ('html', 'png', 'jpeg', 'webp', 'svg')
            width: Image width in pixels
            height: Image height in pixels
        """
        os.makedirs(os.path.join(self.log_dir, 'visualizations'), exist_ok=True)
        filepath = os.path.join(self.log_dir, 'visualizations', f"{filename}.{format}")
        
        if format == 'html':
            fig.write_html(filepath)
        else:
            fig.write_image(filepath, width=width, height=height)
    
    def generate_report(self, output_file: str = None):
        """Generate an HTML report with all visualizations.
        
        Args:
            output_file: Output HTML file path. If None, uses 'report_<timestamp>.html' in the log directory
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.log_dir, f'report_{timestamp}.html')
        
        # Create visualizations directory if it doesn't exist
        vis_dir = os.path.join(self.log_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Generate all available visualizations
        visualizations = {}
        
        try:
            # Metric evolution
            for metric in self.metrics.keys():
                fig = self.plot_metric_evolution(metric)
                filename = f'metric_evolution_{metric}'
                self.save_visualization(fig, filename, 'html')
                visualizations[f'metric_evolution_{metric}'] = filename
        except Exception as e:
            print(f"Error generating metric evolution plots: {e}")
        
        try:
            # Architecture evolution
            fig = self.plot_architecture_evolution()
            self.save_visualization(fig, 'architecture_evolution', 'html')
            visualizations['architecture_evolution'] = 'architecture_evolution'
        except Exception as e:
            print(f"Error generating architecture evolution plot: {e}")
        
        try:
            # 3D evolution
            fig = self.plot_3d_evolution()
            self.save_visualization(fig, '3d_evolution', 'html')
            visualizations['3d_evolution'] = '3d_evolution'
        except Exception as e:
            print(f"Error generating 3D evolution plot: {e}")
        
        try:
            # Parallel coordinates
            fig = self.plot_parallel_coordinates()
            self.save_visualization(fig, 'parallel_coordinates', 'html')
            visualizations['parallel_coordinates'] = 'parallel_coordinates'
        except Exception as e:
            print(f"Error generating parallel coordinates plot: {e}")
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>NeuroGenesis Evolution Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .plot-container {{ margin: 30px 0; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
                .plot-title {{ margin-top: 0; color: #3498db; }}
                iframe {{ width: 100%; height: 600px; border: none; }}
                .header {{ 
                    background-color: #2c3e50; 
                    color: white; 
                    padding: 20px; 
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .timestamp {{ 
                    color: #bdc3c7; 
                    font-size: 0.9em;
                    margin-top: 10px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>NeuroGenesis Evolution Report</h1>
                    <div class="timestamp">Generated on {timestamp}</div>
                </div>
        """.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Add visualizations to the report
        for vis_name, vis_file in visualizations.items():
            vis_title = ' '.join([word.capitalize() for word in vis_name.split('_')])
            html_content += f"""
                <div class="plot-container">
                    <h2 class="plot-title">{vis_title}</h2>
                    <iframe src="visualizations/{vis_file}.html"></iframe>
                </div>
            """
        
        # Add summary statistics
        if self.metrics:
            html_content += """
                <div class="plot-container">
                    <h2>Summary Statistics</h2>
                    <table border="1" cellpadding="5" cellspacing="0" style="width:100%; border-collapse: collapse;">
                        <tr style="background-color: #f2f2f2;">
                            <th>Metric</th>
                            <th>Min</th>
                            <th>Max</th>
                            <th>Mean</th>
                            <th>Std Dev</th>
                        </tr>
            """
            
            for metric, values in self.metrics.items():
                if values and len(values) > 0:
                    html_content += f"""
                        <tr>
                            <td>{metric}</td>
                            <td>{min(values):.4f}</td>
                            <td>{max(values):.4f}</td>
                            <td>{np.mean(values):.4f}</td>
                            <td>{np.std(values):.4f}</td>
                        </tr>
                    """
            
            html_content += """
                    </table>
                </div>
            """
        
        # Close HTML
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Save the report
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"Report generated: {output_file}")
        return output_file

# Example usage
if __name__ == "__main__":
    import numpy as np
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    import random
    
    # Initialize the tracker
    tracker = EvolutionTracker()
    
    # Simulate evolution for 20 generations
    num_generations = 20
    num_models_per_gen = 10
    
    for gen in range(num_generations):
        print(f"\n--- Generation {gen} ---")
        
        # Create random models for this generation
        models = []
        metrics = {
            'accuracy': [],
            'loss': [],
            'fitness': []
        }
        
        for i in range(num_models_per_gen):
            # Create a simple model with random architecture
            num_layers = random.randint(1, 5)
            model = Sequential()
            model.add(Dense(units=random.choice([32, 64, 128, 256]), 
                           activation='relu', 
                           input_shape=(10,)))
            
            for _ in range(num_layers - 1):
                model.add(Dense(units=random.choice([32, 64, 128, 256]), 
                               activation='relu'))
            
            model.add(Dense(1, activation='sigmoid'))
            
            # Generate random metrics
            accuracy = 0.5 + 0.4 * (gen / num_generations) + 0.1 * random.random()
            loss = 0.5 - 0.4 * (gen / num_generations) - 0.1 * random.random()
            fitness = 0.6 * accuracy + 0.4 * (1 - loss)
            
            models.append(model)
            metrics['accuracy'].append(accuracy)
            metrics['loss'].append(loss)
            metrics['fitness'].append(fitness)
            
            print(f"Model {i}: layers={num_layers}, accuracy={accuracy:.3f}, loss={loss:.3f}, fitness={fitness:.3f}")
        
        # Log the generation
        tracker.log_generation(
            models=models,
            metrics=metrics,
            generation=gen,
            metadata={
                'note': f'Generation {gen} with {num_models_per_gen} models',
                'best_fitness': max(metrics['fitness'])
            }
        )
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Plot metric evolution
    fig1 = tracker.plot_metric_evolution('fitness')
    fig1.show()
    
    # Plot architecture evolution
    fig2 = tracker.plot_architecture_evolution()
    fig2.show()
    
    # 3D evolution plot
    fig3 = tracker.plot_3d_evolution()
    fig3.show()
    
    # Parallel coordinates plot
    fig4 = tracker.plot_parallel_coordinates()
    fig4.show()
    
    # Generate and open the report
    report_path = tracker.generate_report()
    print(f"\nReport generated: {report_path}")
