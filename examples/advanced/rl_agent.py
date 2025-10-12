"""
Self-Evolving Reinforcement Learning Agent with Interactive Training Visualization

This example demonstrates a deep reinforcement learning agent that can evolve its architecture
during training and provides interactive visualization of the training process.
"""
import os
import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, clear_output
from collections import deque
import random

from neurogenesis.core import NeuralLearner, NetworkConfig, NetworkType
from neurogenesis.core.layers import LayerType
from neurogenesis.core.strategies import EvolutionConfig, get_evolution_strategy

class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add an experience to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of experiences."""
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones
    
    def __len__(self):
        return len(self.buffer)

class EvolvingRLAgent:
    """Self-Evolving Deep Q-Network agent."""
    
    def __init__(self, 
                 state_dim,
                 action_dim,
                 learning_rate=0.001,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.995,
                 batch_size=64,
                 memory_size=100000,
                 target_update_freq=1000):
        """Initialize the DQN agent."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.training_step = 0
        
        # Create replay buffer
        self.memory = ReplayBuffer(memory_size)
        
        # Build models
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())
        
        # Training history
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_epsilon = []
        self.losses = []
        
        # Evolution configuration
        self.evolution_config = EvolutionConfig(
            population_size=5,
            elite_size=2,
            mutation_rate=0.2,
            crossover_rate=0.8
        )
    
    def _build_model(self):
        """Build the DQN model."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_dim=self.state_dim),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='linear')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.add(state, action, reward, next_state, done)
    
    def act(self, state):
        """Select an action using epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        q_values = self.model.predict(state[None, :], verbose=0)[0]
        return np.argmax(q_values)
    
    def replay(self):
        """Train the model on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Predict Q-values for current and next states
        current_q = self.model.predict(states, verbose=0)
        next_q = self.target_model.predict(next_states, verbose=0)
        
        # Compute target Q-values
        targets = np.copy(current_q)
        for i in range(len(dones)):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])
        
        # Train the model
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        self.losses.append(loss)
        
        # Update target network
        if self.training_step % self.target_update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())
        
        self.training_step += 1
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss
    
    def _evolve_architecture(self):
        """Evolve the network architecture."""
        # Create a population of models
        population = [self._mutate_model() for _ in range(self.evolution_config.population_size)]
        
        # Evaluate each model
        fitness_scores = []
        for model in population:
            # Evaluate on a sample from replay buffer
            if len(self.memory) > self.batch_size:
                states, _, _, _, _ = self.memory.sample(self.batch_size)
                q_values = model.predict(states, verbose=0)
                target_q = self.target_model.predict(states, verbose=0)
                
                # Fitness is negative MSE (higher is better)
                mse = np.mean(np.square(q_values - target_q))
                fitness = -mse
                fitness_scores.append(fitness)
            else:
                fitness_scores.append(-float('inf'))
        
        # Select the best model
        best_idx = np.argmax(fitness_scores)
        if fitness_scores[best_idx] > -np.inf:  # If any model had valid fitness
            self.model = population[best_idx]
    
    def _mutate_model(self):
        """Create a mutated version of the current model."""
        # Clone the current model
        new_model = tf.keras.models.clone_model(self.model)
        new_model.set_weights(self.model.get_weights())
        
        # Randomly mutate some layers
        for layer in new_model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                # Randomly add or remove neurons
                if np.random.random() < 0.3:  # 30% chance to modify layer
                    weights = layer.get_weights()
                    if len(weights) > 0:
                        # Add small random noise to weights
                        weights = [w + np.random.normal(0, 0.1, w.shape) for w in weights]
                        layer.set_weights(weights)
        
        # Recompile the model
        new_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return new_model
    
    def train(self, env, episodes=1000, render_every=50):
        """Train the agent in the given environment."""
        for e in range(episodes):
            state, _ = env.reset()
            state = np.reshape(state, [1, self.state_dim])
            total_reward = 0
            done = False
            step = 0
            
            while not done and step < 1000:  # Prevent infinite episodes
                if e % render_every == 0:
                    env.render()
                
                # Select and perform an action
                action = self.act(state)
                next_state, reward, done, _, _ = env.step(action)
                next_state = np.reshape(next_state, [1, self.state_dim])
                
                # Store experience in replay memory
                self.remember(state[0], action, reward, next_state[0], done)
                
                # Train the model
                loss = self.replay()
                
                # Update state and track rewards
                state = next_state
                total_reward += reward
                step += 1
            
            # Store episode statistics
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(step)
            self.episode_epsilon.append(self.epsilon)
            
            # Evolve architecture every 10 episodes
            if e > 0 and e % 10 == 0:
                self._evolve_architecture()
            
            # Print progress
            avg_reward = np.mean(self.episode_rewards[-100:])  # Last 100 episodes
            print(f"Episode: {e+1}/{episodes}, "
                  f"Score: {total_reward:.2f}, "
                  f"Avg Score (last 100): {avg_reward:.2f}, "
                  f"Epsilon: {self.epsilon:.3f}")
            
            # Early stopping if solved
            if len(self.episode_rewards) >= 100 and avg_reward >= 195.0:
                print(f"Environment solved in {e+1} episodes!")
                break
        
        return self.episode_rewards
    
    def plot_training(self):
        """Plot training statistics."""
        # Create subplots
        fig = make_subplots(rows=2, cols=2, 
                           subplot_titles=('Episode Rewards', 
                                         'Episode Lengths',
                                         'Loss',
                                         'Epsilon'))
        
        # Episode Rewards
        fig.add_trace(
            go.Scatter(
                y=self.episode_rewards,
                mode='lines+markers',
                name='Reward',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Moving average of rewards
        window = max(1, len(self.episode_rewards) // 20)  # 5% window
        if len(self.episode_rewards) > window:
            moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            fig.add_trace(
                go.Scatter(
                    x=np.arange(window-1, len(self.episode_rewards)),
                    y=moving_avg,
                    mode='lines',
                    name=f'Moving Avg ({window} eps)',
                    line=dict(color='red', width=2)
                ),
                row=1, col=1
            )
        
        # Episode Lengths
        fig.add_trace(
            go.Scatter(
                y=self.episode_lengths,
                mode='lines',
                name='Length',
                line=dict(color='green')
            ),
            row=1, col=2
        )
        
        # Loss
        if self.losses:
            fig.add_trace(
                go.Scatter(
                    y=self.losses,
                    mode='lines',
                    name='Loss',
                    line=dict(color='purple')
                ),
                row=2, col=1
            )
        
        # Epsilon
        fig.add_trace(
            go.Scatter(
                y=self.episode_epsilon,
                mode='lines',
                name='Epsilon',
                line=dict(color='orange')
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Training Statistics',
            showlegend=True,
            height=800,
            width=1200
        )
        
        # Show the plot
        fig.show()

def train_cartpole():
    """Train the agent on the CartPole-v1 environment."""
    # Create environment
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Create and train agent
    agent = EvolvingRLAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        memory_size=100000,
        target_update_freq=100
    )
    
    # Train the agent
    rewards = agent.train(env, episodes=500)
    
    # Plot training statistics
    agent.plot_training()
    
    # Close the environment
    env.close()
    
    return agent

if __name__ == "__main__":
    # Train the agent
    agent = train_cartpole()
    
    # Save the trained model
    agent.model.save('cartpole_agent.h5')
    
    # Load and test the model
    # loaded_model = tf.keras.models.load_model('cartpole_agent.h5')
    # agent.model = loaded_model
    # test_agent(agent, env)
