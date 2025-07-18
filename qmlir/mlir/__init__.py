"""MLIR Integration for QMLIR

This module contains all MLIR-related functionality including
circuit compilation to MLIR and optimization tools.
"""

from .compiler import circuit_to_mlir, optimize
from .config import get_quantum_opt_path

__all__ = ["circuit_to_mlir", "optimize", "get_quantum_opt_path"]
