"""
EvoOptimizer: An Evolved Deep Learning Optimizer

Discovered via genetic algorithm search over optimizer update rules.
Based on "Evolving Deep Learning Optimizers" (arXiv:2512.11853).
"""

from evo_optimizer.optimizer import EvoOptimizer, EvoOptimizerSimplified
from evo_optimizer.functional import evo_optimizer_step

__version__ = "0.1.0"
__author__ = "Mitchell Marfinetz"
__all__ = ["EvoOptimizer", "EvoOptimizerSimplified", "evo_optimizer_step"]
