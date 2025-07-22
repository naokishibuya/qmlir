"""JAX-based quantum circuit simulator.

This module provides the JaxSimulator class for high-performance quantum circuit
simulation using JAX with automatic JIT compilation and optimization.
"""

import jax
import jax.numpy as jnp
from collections import Counter
from typing import Dict
from ...circuit import QuantumCircuit
from ...observable import Observable
from ...operator import Z
from ...mlir import circuit_to_mlir, apply_passes
from .engine import simulate_from_mlir
from .observable import evaluate_observable


class JaxSimulator:
    def __init__(self, optimize_circuit: bool = True, seed: int = 0):
        self.optimize_circuit = optimize_circuit
        self.rng_key = jax.random.PRNGKey(seed)  # Random key for reproducibility

    def measure(self, circuit: QuantumCircuit, shots: int) -> Dict[str, int]:
        """Sample measurements from the circuit by running it multiple times."""
        self.rng_key, subkey = jax.random.split(self.rng_key)

        # Calculate probabilities of measurement outcomes
        probs = self.probabilities(circuit)
        num_states = probs.shape[0]

        # Sample bitstrings based on probabilities
        samples = jax.random.choice(subkey, num_states, shape=(shots,), p=probs)
        num_qubits = circuit.num_qubits
        bitstrings = [format(i, f"0{num_qubits}b") for i in samples]
        return dict(Counter(bitstrings))

    def expectation(self, circuit: QuantumCircuit, observable: Observable) -> float:
        """Calculate expectation value of an observable."""
        state_vector = self.statevector(circuit)
        num_qubits = circuit.num_qubits
        if observable is None:
            qubits = list(range(num_qubits))
            observable = Z(*qubits)
        elif not isinstance(observable, Observable):
            raise TypeError(f"Expected Observable, got {type(observable).__name__}")
        little_endian = circuit.little_endian
        return float(jnp.real(evaluate_observable(state_vector, num_qubits, observable, little_endian)))

    def statevector(self, circuit: QuantumCircuit) -> jnp.ndarray:
        """Get the final state vector of the circuit."""
        results = self._simulate_circuit(circuit)
        return results["final_state"]

    def probabilities(self, circuit: QuantumCircuit) -> jnp.ndarray:
        """Calculate measurement probabilities."""
        results = self._simulate_circuit(circuit)
        return results["probabilities"]

    def _simulate_circuit(self, circuit: QuantumCircuit) -> Dict:
        """Internal method to simulate a circuit and return results."""
        # Step 1: Transpile circuit to MLIR
        mlir_code = circuit_to_mlir(circuit)

        # Step 2: Apply optimization passes if requested
        if self.optimize_circuit:
            mlir_code = apply_passes(mlir_code)
        circuit.compiled_mlir = mlir_code.strip()  # Store compiled MLIR in circuit for reference

        # Step 3: Collect parameter values from circuit
        param_values = []
        param_ids_seen = set()
        for gate in circuit.gates:
            for param in gate.parameters:
                if param.id not in param_ids_seen:
                    param_values.append(param.value)
                    param_ids_seen.add(param.id)

        # Step 4: Simulate with JAX runtime
        return simulate_from_mlir(mlir_code, circuit.num_qubits, param_values, circuit.little_endian)
