"""JAX-based quantum circuit runtime for QMLIR.

This module provides high-performance quantum circuit simulation using JAX
for automatic differentiation, JIT compilation, and GPU acceleration.
"""

from .jax_backend import simulate_circuit, simulate_from_mlir, GateID

__all__ = ["simulate_circuit", "simulate_from_mlir", "GateID"]
