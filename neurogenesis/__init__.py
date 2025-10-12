"""
NeuroGenesis - Self-Evolving Neural Architectures

A framework for creating self-evolving neural networks inspired by biological neurogenesis.
"""

__version__ = "0.1.0"
__author__ = "Amir Hossein Rasti"
__license__ = "MIT"

from .core.learner import NeuralLearner
from .core.controller import MetaController
from .core.evolution import EvolutionEngine

__all__ = ['NeuralLearner', 'MetaController', 'EvolutionEngine']
