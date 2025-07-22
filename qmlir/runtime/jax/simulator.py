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
from .circuit import simulate_circuit
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
        results = simulate_circuit(circuit, self.optimize_circuit)
        return results["statevector"]

    def probabilities(self, circuit: QuantumCircuit) -> jnp.ndarray:
        """Calculate measurement probabilities."""
        results = simulate_circuit(circuit, self.optimize_circuit)
        return results["probabilities"]
