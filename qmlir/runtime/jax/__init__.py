"""JAX Runtime Backend

This module provides the JAX runtime backend for high-performance quantum circuit
simulation using JAX with automatic JIT compilation and optimization.
"""

from .simulator import JaxSimulator

__all__ = ["JaxSimulator"]
