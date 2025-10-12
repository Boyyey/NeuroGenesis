"""
NeuroGenesis Dashboard

A comprehensive dashboard for visualizing and interacting with the NeuroGenesis framework.
"""
import os
import sys
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_plotly_events import plotly_events
import pandas as pd
from PIL import Image
import base64
from io import BytesIO

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import visualization modules
from neurogenesis.visualization import (
    AttentionVisualizer,
    LatentSpaceExplorer,
    ArchitectureVisualizer,
    TrainingVisualizer,
    FeatureMapVisualizer,
    DecisionBoundaryVisualizer,
    EvolutionTracker
)

# Import example modules
from examples.advanced.vision_transformer import VisionTransformer, train_vision_transformer
from examples.advanced.evolving_gan import EvolvingGAN, train_evolving_gan
from examples.advanced.rl_agent import EvolvingRLAgent, train_cartpole

# Set page config
st.set_page_config(
    page_title="NeuroGenesis Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    h1, h2, h3 {
        color: #1E88E5;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🧠 NeuroGenesis Dashboard")
st.sidebar.markdown("---")

# Main content
def main():
    """Main dashboard function."""
    st.title("NeuroGenesis: Self-Evolving Neural Architectures")
    st.markdown("---")
    
    # Dashboard sections
    sections = [
        "🏠 Home",
        "🔍 Model Explorer",
        "🎨 Latent Space Explorer",
        "🧬 Architecture Visualizer",
        "📊 Training Monitor",
        "🎮 Interactive Demos",
        "📈 Performance Analytics"
    ]
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", sections)
    
    # Home section
    if selection == "🏠 Home":
        st.header("Welcome to NeuroGenesis")
        st.markdown("""
        ### A Framework for Self-Evolving Neural Networks
        
        NeuroGenesis enables the creation, visualization, and analysis of self-evolving 
        neural network architectures with advanced visualization capabilities.
        """)
        
        # Quick start guide
        with st.expander("🚀 Quick Start"):
            st.markdown("""
            1. **Explore Pre-trained Models**: Navigate to the Model Explorer to load and analyze existing models.
            2. **Visualize Latent Spaces**: Use the Latent Space Explorer to understand your model's learned representations.
            3. **Analyze Architectures**: The Architecture Visualizer provides 3D views of network structures.
            4. **Run Interactive Demos**: Try out our pre-built examples in the Interactive Demos section.
            """)
        
        # Features grid
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("**3D Architecture Visualization**")
            st.markdown("Interactive 3D visualization of neural network architectures with layer-wise activations.")
        
        with col2:
            st.success("**Latent Space Exploration**")
            st.markdown("Explore high-dimensional latent spaces with dimensionality reduction techniques.")
        
        with col3:
            st.warning("**Training Monitoring**")
            st.markdown("Real-time monitoring of training progress and model performance.")
        
        # Add a sample visualization
        st.markdown("---")
        st.header("Sample Visualization")
        
        # Create a sample 3D scatter plot
        fig = go.Figure(data=[
            go.Scatter3d(
                x=np.random.randn(100),
                y=np.random.randn(100),
                z=np.random.randn(100),
                mode='markers',
                marker=dict(
                    size=5,
                    color=np.random.randn(100),
                    colorscale='Viridis',
                    opacity=0.8
                )
            )
        ])
        
        fig.update_layout(
            title="Sample 3D Scatter Plot",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            width=1000,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Model Explorer section
    elif selection == "🔍 Model Explorer":
        st.header("Model Explorer")
        st.markdown("Load and analyze pre-trained models or train new ones.")
        
        # Model selection
        model_type = st.selectbox(
            "Select Model Type",
            ["Vision Transformer", "Evolving GAN", "Reinforcement Learning Agent"]
        )
        
        if model_type == "Vision Transformer":
            with st.spinner("Loading Vision Transformer model..."):
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
                st.success("Vision Transformer model loaded successfully!")
                
                # Display model summary
                st.subheader("Model Summary")
                model.model.summary(print_fn=lambda x: st.text(x))
                
                # Visualize architecture
                st.subheader("Architecture Visualization")
                if st.button("Show 3D Architecture"):
                    visualizer = ArchitectureVisualizer(model.model)
                    fig = visualizer.plot_interactive_architecture()
                    st.plotly_chart(fig, use_container_width=True)
        
        # Add similar sections for other model types...
    
    # Latent Space Explorer section
    elif selection == "🎨 Latent Space Explorer":
        st.header("Latent Space Explorer")
        st.markdown("Visualize and explore high-dimensional latent spaces.")
        
        # Generate sample data
        np.random.seed(42)
        n_samples = 1000
        n_features = 50
        X = np.random.randn(n_samples, n_features)
        labels = np.random.randint(0, 5, n_samples)
        
        # Method selection
        method = st.selectbox(
            "Dimensionality Reduction Method",
            ["t-SNE", "UMAP", "PCA", "Isomap"]
        )
        
        # Create explorer
        explorer = LatentSpaceExplorer(method=method.lower(), n_components=3)
        
        # Fit and transform
        with st.spinner(f"Fitting {method}..."):
            projection = explorer.fit_transform(X)
        
        # Plot
        st.subheader("3D Latent Space Visualization")
        fig = explorer.plot_3d_latent_space(X, labels=labels)
        st.plotly_chart(fig, use_container_width=True)
        
        # Interactive controls
        st.sidebar.markdown("### Visualization Controls")
        show_labels = st.sidebar.checkbox("Show Labels", value=True)
        point_size = st.sidebar.slider("Point Size", 1, 10, 5)
        
        # Update plot with new parameters
        if st.sidebar.button("Update Visualization"):
            fig.update_traces(marker_size=point_size)
            st.plotly_chart(fig, use_container_width=True)
    
    # Architecture Visualizer section
    elif selection == "🧬 Architecture Visualizer":
        st.header("Neural Network Architecture Visualizer")
        st.markdown("Visualize neural network architectures in 3D with interactive controls.")
        
        # Load a sample model
        with st.spinner("Loading sample model..."):
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
            
            visualizer = ArchitectureVisualizer(model)
            
            # Display model summary
            st.subheader("Model Summary")
            model.summary(print_fn=lambda x: st.text(x))
            
            # Show 3D visualization
            st.subheader("3D Architecture Visualization")
            fig = visualizer.plot_interactive_architecture()
            st.plotly_chart(fig, use_container_width=True)
            
            # Show animation
            st.subheader("Architecture Animation")
            if st.button("Generate Animation"):
                with st.spinner("Generating animation..."):
                    animation = visualizer.create_architecture_animation()
                    st.components.v1.html(animation.data, height=600)
    
    # Add other sections...
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About NeuroGenesis
    Version 1.0.0  
    [GitHub Repository](https://github.com/boyyey/neurogenesis)  
    
    Developed with ❤️ AmirHosseinRasti
    """)

if __name__ == "__main__":
    main()
