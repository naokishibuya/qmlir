"""High-level quantum circuit simulation interface for QMLIR.

This module provides a clean, user-friendly API for simulating quantum circuits
that handles the complete build → transpile → simulate pipeline automatically.
"""

from typing import Dict, Optional, Any
from .circuit import QuantumCircuit
from .mlir import circuit_to_mlir, optimize
from .runtime import simulate_from_mlir


def simulate(circuit: QuantumCircuit, optimize_circuit: bool = True, initial_state: Optional[Any] = None) -> Dict:
    """
    High-level function to simulate a quantum circuit.

    This function handles the complete pipeline:
    1. Build circuit → 2. Transpile to MLIR → 3. Optimize → 4. Simulate with JAX

    Args:
        circuit: The quantum circuit to simulate
        optimize_circuit: Whether to apply MLIR optimization passes (default: True)
        initial_state: Optional initial quantum state vector

    Returns:
        Dictionary with simulation results:
        - 'final_state': Final quantum state vector (JAX array)
        - 'probabilities': Measurement probabilities for each basis state
        - 'num_qubits': Number of qubits in the circuit
        - 'num_operations': Number of operations executed

    Example:
        >>> from qmlir import QuantumCircuit
        >>> from qmlir.simulator import simulate
        >>>
        >>> # Create Bell state
        >>> circuit = QuantumCircuit(2)
        >>> circuit.h(0).cx(0, 1)
        >>>
        >>> # Simulate
        >>> results = simulate(circuit)
        >>> print(results['probabilities'])  # [0.5, 0.0, 0.0, 0.5]
        >>>
        >>> # Without optimization
        >>> results = simulate(circuit, optimize_circuit=False)

    Note:
        The simulation uses JAX for high-performance computation with automatic
        vectorization and potential GPU acceleration.
    """
    # Step 1: Transpile circuit to MLIR
    mlir_code = circuit_to_mlir(circuit)

    # Step 2: Optimize if requested
    if optimize_circuit:
        mlir_code = optimize(mlir_code)

    # Step 3: Simulate with JAX runtime
    results = simulate_from_mlir(mlir_code, num_qubits=circuit.num_qubits)

    return results


def simulate_statevector(circuit: QuantumCircuit, optimize_circuit: bool = True) -> Any:
    """
    Simulate quantum circuit and return only the final state vector.

    Args:
        circuit: The quantum circuit to simulate
        optimize_circuit: Whether to apply MLIR optimization passes

    Returns:
        Final quantum state vector as JAX array

    Example:
        >>> circuit = QuantumCircuit(1)
        >>> circuit.h(0)  # |+⟩ = (|0⟩ + |1⟩)/√2
        >>> state = simulate_statevector(circuit)
        >>> print(state)  # [0.7071+0j, 0.7071+0j]
    """
    results = simulate(circuit, optimize_circuit=optimize_circuit)
    return results["final_state"]


def simulate_probabilities(circuit: QuantumCircuit, optimize_circuit: bool = True) -> Any:
    """
    Simulate quantum circuit and return only the measurement probabilities.

    Args:
        circuit: The quantum circuit to simulate
        optimize_circuit: Whether to apply MLIR optimization passes

    Returns:
        Measurement probabilities as JAX array

    Example:
        >>> circuit = QuantumCircuit(2)
        >>> circuit.h(0).cx(0, 1)  # Bell state
        >>> probs = simulate_probabilities(circuit)
        >>> print(probs)  # [0.5, 0.0, 0.0, 0.5]
    """
    results = simulate(circuit, optimize_circuit=optimize_circuit)
    return results["probabilities"]
