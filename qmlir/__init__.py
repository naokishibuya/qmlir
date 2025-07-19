"""QMLIR - Quantum Computing with MLIR

A simple, fast quantum computing library that compiles circuits to MLIR
and simulates them using JAX for high performance.
"""

from .circuit import QuantumCircuit
from .parameter import Parameter
from .runtime import JaxSimulator

# Main API - Clean, backend-explicit interface
__all__ = [
    "QuantumCircuit",
    "Parameter",
    "JaxSimulator",
]

# No global simulate() function - users explicitly choose their backend
