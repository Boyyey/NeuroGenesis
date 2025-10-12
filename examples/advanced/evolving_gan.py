"""
Self-Evolving GAN with Interactive Latent Space Visualization

This example demonstrates a Generative Adversarial Network that can evolve its architecture
during training and provides interactive visualization of the latent space.
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import HTML, display
import ipywidgets as widgets
from tqdm import tqdm

from neurogenesis.core import NeuralLearner, NetworkConfig, NetworkType
from neurogenesis.core.layers import LayerType
from neurogenesis.core.strategies import EvolutionConfig, get_evolution_strategy

class EvolvingGAN:
    """Self-Evolving Generative Adversarial Network with interactive visualization."""
    
    def __init__(self, 
                 latent_dim=100,
                 img_shape=(64, 64, 3),
                 initial_filters=64):
        """Initialize the EvolvingGAN."""
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.initial_filters = initial_filters
        
        # Build models
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        
        # Combined model for training generator
        self.combined = self._build_combined()
        
        # Evolution configuration
        self.evolution_config = EvolutionConfig(
            population_size=5,
            elite_size=2,
            mutation_rate=0.2,
            crossover_rate=0.8
        )
        
        # Visualization
        self.latent_points = None
        self.generated_images = None
    
    def _build_generator(self):
        """Build the generator model."""
        noise = tf.keras.layers.Input(shape=(self.latent_dim,))
        
        # Initial dense layer
        x = tf.keras.layers.Dense(8 * 8 * 256, use_bias=False)(noise)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Reshape((8, 8, 256))(x)
        
        # Upsample to 16x16
        x = tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        
        # Upsample to 32x32
        x = tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        
        # Upsample to 64x64 if needed
        if self.img_shape[0] >= 64:
            x = tf.keras.layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(0.2)(x)
        
        # Output layer
        x = tf.keras.layers.Conv2DTranspose(
            self.img_shape[2], (5, 5), strides=(1, 1), 
            padding='same', activation='tanh')(x)
        
        return tf.keras.Model(noise, x, name='generator')
    
    def _build_discriminator(self):
        """Build the discriminator model."""
        img = tf.keras.layers.Input(shape=self.img_shape)
        
        # Downsample to 32x32 if needed
        if self.img_shape[0] > 32:
            x = tf.keras.layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same')(img)
            x = tf.keras.layers.LeakyReLU(0.2)(x)
        else:
            x = img
        
        # Downsample to 16x16
        x = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Downsample to 8x8
        x = tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Output layer
        x = tf.keras.layers.Flatten()(x)
        validity = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        return tf.keras.Model(img, validity, name='discriminator')
    
    def _build_combined(self):
        """Build the combined GAN model."""
        self.discriminator.trainable = False
        
        # Generator takes noise as input and generates images
        noise = tf.keras.layers.Input(shape=(self.latent_dim,))
        img = self.generator(noise)
        
        # Discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)
        
        # Combined model (stacked generator and discriminator)
        return tf.keras.Model(noise, valid, name='gan')
    
    def train(self, dataset, epochs=10000, batch_size=32, sample_interval=50):
        """Train the GAN with evolution."""
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        # Training history
        d_losses = []
        g_losses = []
        
        # Create output directory
        os.makedirs('gan_images', exist_ok=True)
        
        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            # Select a random batch of images
            idx = np.random.randint(0, dataset.shape[0], batch_size)
            imgs = dataset[idx]
            
            # Generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)
            
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # ---------------------
            #  Train Generator
            # ---------------------
            
            # Train the generator to fool the discriminator
            g_loss = self.combined.train_on_batch(noise, valid)
            
            # Store losses
            d_losses.append(d_loss[0])
            g_losses.append(g_loss)
            
            # Print progress
            print(f"{epoch} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")
            
            # If at sample interval, save generated images and evolve
            if epoch % sample_interval == 0:
                self._sample_images(epoch)
                self._evolve()
    
    def _evolve(self):
        """Evolve the GAN architecture."""
        # Create a population of generators
        population = [self._mutate_generator() for _ in range(self.evolution_config.population_size)]
        
        # Evaluate each generator
        fitness_scores = []
        for gen in population:
            # Generate images with the generator
            noise = np.random.normal(0, 1, (10, self.latent_dim))
            gen_imgs = gen.predict(noise)
            
            # Evaluate quality using discriminator
            validity = self.discriminator.predict(gen_imgs)
            fitness = np.mean(validity)  # Higher is better
            fitness_scores.append(fitness)
        
        # Select the best generator
        best_idx = np.argmax(fitness_scores)
        if fitness_scores[best_idx] > 0.7:  # Only update if significantly better
            self.generator = population[best_idx]
            self.combined = self._build_combined()  # Rebuild combined model
    
    def _mutate_generator(self):
        """Create a mutated version of the generator."""
        # Clone the current generator
        new_gen = tf.keras.models.clone_model(self.generator)
        new_gen.set_weights(self.generator.get_weights())
        
        # Randomly mutate some weights
        for layer in new_gen.layers:
            if isinstance(layer, tf.keras.layers.Dense) or isinstance(layer, tf.keras.layers.Conv2DTranspose):
                weights = layer.get_weights()
                if len(weights) > 0:
                    # Add small random noise to weights
                    weights = [w + np.random.normal(0, 0.1, w.shape) for w in weights]
                    layer.set_weights(weights)
        
        return new_gen
    
    def _sample_images(self, epoch):
        """Save generated images for visualization."""
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(f"gan_images/{epoch}.png")
        plt.close()
    
    def visualize_latent_space(self, n_samples=100):
        """Visualize the latent space in 2D using t-SNE or PCA."""
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        
        # Generate random points in latent space
        self.latent_points = np.random.normal(0, 1, (n_samples, self.latent_dim))
        
        # Generate images from these points
        self.generated_images = self.generator.predict(self.latent_points)
        
        # Reduce dimensionality to 2D
        reducer = TSNE(n_components=2, random_state=42)
        # Or use PCA for faster computation:
        # reducer = PCA(n_components=2)
        
        latent_2d = reducer.fit_transform(self.latent_points)
        
        # Create interactive plot
        fig = go.Figure()
        
        # Add scatter points
        fig.add_trace(go.Scatter(
            x=latent_2d[:, 0],
            y=latent_2d[:, 1],
            mode='markers',
            marker=dict(size=10, opacity=0.7),
            hoverinfo='none',
            showlegend=False
        ))
        
        # Add images as annotations
        for i in range(n_samples):
            # Convert image to base64
            img = self.generated_images[i]
            img = (img * 255).astype('uint8')
            
            # Create hover text with image
            fig.add_annotation(
                x=latent_2d[i, 0],
                y=latent_2d[i, 1],
                xref="x",
                yref="y",
                text="",
                showarrow=False,
                xanchor="center",
                yanchor="middle",
                # This would work in a Jupyter notebook with plotly
                # but needs adjustment for other environments
                # ax=0, ay=0,
                # xshift=0, yshift=0,
                # xanchor='left',
                # yanchor='bottom',
                # xref="x",
                # yref="y",
                # text=f"<img src='data:image/png;base64,{img_base64}'>",
                # showarrow=False,
                # xshift=0, yshift=0,
                # xanchor='left',
                # yanchor='bottom',
                # xref="x",
                # yref="y",
                # align='left',
                # bordercolor='#c7c7c7',
                # borderwidth=2,
                # borderpad=4,
                # bgcolor='#ffffff',
                # opacity=0.8
            )
        
        # Update layout
        fig.update_layout(
            title='Latent Space Visualization',
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            hovermode='closest',
            showlegend=False,
            width=800,
            height=600
        )
        
        # In a Jupyter notebook, you would do:
        # fig.show()
        
        # For non-notebook environments, save to HTML
        fig.write_html("latent_space.html")
        return fig

def load_celeb_a():
    """Load and preprocess the CelebA dataset."""
    # This is a placeholder - in practice, you would load the CelebA dataset
    # or any other dataset of your choice
    (x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()
    
    # Resize to 64x64 and normalize to [-1, 1]
    x_train = tf.image.resize(x_train, [64, 64]) / 127.5 - 1.0
    
    return x_train.numpy()

def train_evolving_gan():
    """Train the evolving GAN."""
    # Load dataset
    dataset = load_celeb_a()
    
    # Create and train GAN
    gan = EvolvingGAN(
        latent_dim=100,
        img_shape=(64, 64, 3),
        initial_filters=64
    )
    
    # Compile models
    gan.discriminator.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
        metrics=['accuracy']
    )
    
    gan.combined.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.0002, 0.5)
    )
    
    # Train the GAN
    gan.train(dataset, epochs=10000, batch_size=32, sample_interval=200)
    
    return gan

if __name__ == "__main__":
    # Train the GAN
    gan = train_evolving_gan()
    
    # Visualize latent space (in Jupyter notebook)
    if 'get_ipython' in globals():
        gan.visualize_latent_space(n_samples=50)
