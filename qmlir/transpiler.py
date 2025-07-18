"""High-level Quantum Circuit Transpiler Interface

This module provides the main transpilation interface for quantum circuits,
including the transpile function that converts circuits to optimized MLIR.
"""

from .mlir import circuit_to_mlir, optimize
from .circuit import QuantumCircuit


def transpile(circuit: QuantumCircuit, optimization_level: int = 1, function_name: str = "main") -> str:
    """Transpile quantum circuit to optimized MLIR.

    Args:
        circuit: QuantumCircuit to transpile
        optimization_level: 0=none, 1=basic optimizations, 2=aggressive (future)
        function_name: Name of the generated MLIR function

    Returns:
        str: Optimized MLIR code

    Raises:
        RuntimeError: If optimization fails
    """
    # Generate base MLIR
    mlir_code = circuit_to_mlir(circuit, function_name)

    # Apply optimizations based on level
    if optimization_level == 0:
        return mlir_code
    elif optimization_level >= 1:
        result = optimize(mlir_code, "--quantum-cancel-self-inverse")
        if result.returncode == 0:
            return result.stdout
        else:
            raise RuntimeError(f"Optimization failed: {result.stderr}")

    return mlir_code
