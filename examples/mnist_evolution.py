"""
MNIST Evolution Example

This script demonstrates how to use the NeuroGenesis framework to evolve
neural network architectures on the MNIST dataset.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neurogenesis.core.evolution import EvolutionEngine, EvolutionConfig
from neurogenesis.core.controller import MetaController
from neurogenesis.core.learner import NeuralLearner

def load_mnist():
    """Load and preprocess the MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Add channel dimension (for CNN compatibility)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    # Flatten the images for fully connected networks
    x_train_flat = x_train.reshape((len(x_train), -1))
    x_test_flat = x_test.reshape((len(x_test), -1))
    
    return (x_train_flat, y_train), (x_test_flat, y_test), (x_train.shape[1:], x_test.shape[1:])

def plot_training_history(history, save_path=None):
    """Plot training and validation metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_evolution(history, save_path=None):
    """Plot evolution progress across generations."""
    generations = [h['generation'] for h in history]
    best_fitness = [h['best_fitness'] for h in history]
    avg_fitness = [h['avg_fitness'] for h in history]
    avg_layers = [h['avg_layers'] for h in history]
    avg_neurons = [h['avg_neurons'] for h in history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot fitness
    ax1.plot(generations, best_fitness, 'b-', label='Best Fitness')
    ax1.plot(generations, avg_fitness, 'r-', label='Average Fitness')
    ax1.set_title('Fitness Over Generations')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness (Validation Accuracy)')
    ax1.legend()
    
    # Plot architecture
    ax2.plot(generations, avg_layers, 'g-', label='Average Layers')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Number of Layers', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    
    ax2_2 = ax2.twinx()
    ax2_2.plot(generations, avg_neurons, 'm-', label='Average Neurons')
    ax2_2.set_ylabel('Number of Neurons', color='m')
    ax2_2.tick_params(axis='y', labelcolor='m')
    
    ax2.set_title('Architecture Evolution')
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'output/mnist_evolution_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare data
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test), (input_shape, _) = load_mnist()
    num_classes = 10
    
    # Split training data into training and validation sets
    val_split = 0.1
    num_val = int(len(x_train) * val_split)
    x_val, y_val = x_train[-num_val:], y_train[-num_val:]
    x_train, y_train = x_train[:-num_val], y_train[:-num_val]
    
    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Test samples: {len(x_test)}")
    
    # Configure evolution
    config = EvolutionConfig(
        population_size=10,
        elite_size=2,
        max_generations=20,
        epochs_per_generation=3,
        min_layers=1,
        max_layers=5,
        min_neurons=32,
        max_neurons=512,
        neuron_step_size=16,
        initial_learning_rate=1e-3,
        min_learning_rate=1e-5,
        max_learning_rate=1e-2,
        mutation_rate=0.4,
        crossover_rate=0.7
    )
    
    # Initialize evolution engine
    print("Initializing evolution engine...")
    evolution = EvolutionEngine(config)
    evolution.initialize_population(input_shape=(np.prod(input_shape),), num_classes=num_classes)
    
    # Evolution loop
    print("Starting evolution...")
    for gen in range(config.max_generations):
        print(f"\n--- Generation {gen + 1}/{config.max_generations} ---")
        
        # Evaluate population
        population = evolution.evaluate_population(
            x_train, y_train, x_val, y_val, 
            epochs=config.epochs_per_generation,
            batch_size=128
        )
        
        # Print generation statistics
        best_fitness = max(ind['fitness'] for ind in population)
        avg_fitness = np.mean([ind['fitness'] for ind in population])
        avg_layers = np.mean([ind['num_layers'] for ind in population])
        avg_neurons = np.mean([sum(ind['neurons_per_layer']) for ind in population])
        
        print(f"Best fitness: {best_fitness:.4f}")
        print(f"Average fitness: {avg_fitness:.4f}")
        print(f"Average layers: {avg_layers:.1f}")
        print(f"Average neurons: {avg_neurons:.1f}")
        
        # Print best individual's architecture
        best = max(population, key=lambda x: x['fitness'])
        print("\nBest architecture:")
        print(f"- Layers: {best['num_layers']}")
        print(f"- Neurons per layer: {best['neurons_per_layer']}")
        print(f"- Activation: {best['activation']}")
        print(f"- Learning rate: {best['learning_rate']:.6f}")
        
        # Save model and plots
        if gen % 5 == 0 or gen == config.max_generations - 1:
            # Save best model
            model_path = os.path.join(output_dir, f'best_model_gen_{gen:03d}.h5')
            best['model'].save(model_path)
            
            # Plot training history for best individual
            if 'history' in best:
                plot_path = os.path.join(output_dir, f'training_gen_{gen:03d}.png')
                plot_training_history(best['history'], save_path=plot_path)
        
        # Evolve to next generation (unless it's the last generation)
        if gen < config.max_generations - 1:
            evolution.evolve()
    
    # Final evaluation on test set
    print("\n--- Final Evaluation on Test Set ---")
    best_model = evolution.get_best_model()
    if best_model:
        test_loss, test_accuracy = best_model.evaluate(x_test, y_test, verbose=0)
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        # Save final model
        final_model_path = os.path.join(output_dir, 'final_model.h5')
        best_model.save(final_model_path)
        print(f"Saved final model to {final_model_path}")
    
    # Plot evolution history
    plot_path = os.path.join(output_dir, 'evolution_history.png')
    plot_evolution(evolution.history, save_path=plot_path)
    
    print(f"\nAll results saved to: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()
