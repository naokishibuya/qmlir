"""JAX-based quantum circuit simulator.

This module provides the JaxSimulator class for high-performance quantum circuit
simulation using JAX with automatic JIT compilation and optimization.
"""

import jax
import jax.numpy as jnp
from collections import Counter
from typing import Dict
from ...circuit import QuantumCircuit
from ...mlir import circuit_to_mlir, apply_passes
from .engine import simulate_from_mlir


class JaxSimulator:
    """JAX-based quantum circuit simulator.

    Provides high-performance quantum circuit simulation using JAX with automatic
    JIT compilation and MLIR optimization.
    """

    def __init__(self, optimize_circuit: bool = True, seed: int = 0):
        """Initialize the JAX simulator.

        Args:
            optimize_circuit: Whether to apply MLIR optimization passes
        """
        self.optimize_circuit = optimize_circuit
        self.rng_key = jax.random.PRNGKey(seed)  # Random key for reproducibility

    def get_counts(self, circuit: QuantumCircuit, shots: int) -> Dict[str, int]:
        """Sample measurements from the circuit by running it multiple times.

        Args:
            circuit: The quantum circuit to sample from
            shots: Number of shots to run

        Returns:
            Dictionary mapping bitstrings to counts
        """
        self.rng_key, subkey = jax.random.split(self.rng_key)

        # Calculate probabilities of measurement outcomes
        probs = self.calc_probs(circuit)
        num_states = probs.shape[0]

        # Sample bitstrings based on probabilities
        samples = jax.random.choice(subkey, num_states, shape=(shots,), p=probs)
        bitstrings = [format(i, f"0{int(jnp.log2(num_states))}b") for i in samples]

        return dict(Counter(bitstrings))

    def calc_expval(self, circuit: QuantumCircuit, observable: str) -> float:
        """Calculate expectation value of an observable.

        Args:
            circuit: The quantum circuit
            observable: Observable as string (e.g., "ZZ", "X", "Y")

        Returns:
            Expectation value as float
        """
        # For now, implement basic Pauli observables
        state_vector = self.state_vector(circuit)

        if observable == "Z":
            # Single qubit Z expectation
            return float(
                jnp.real(jnp.conj(state_vector[0]) * state_vector[0] - jnp.conj(state_vector[1]) * state_vector[1])
            )
        elif observable == "ZZ":
            # Two qubit ZZ expectation
            return float(
                jnp.real(
                    jnp.conj(state_vector[0]) * state_vector[0]
                    + jnp.conj(state_vector[1]) * state_vector[1]
                    - jnp.conj(state_vector[2]) * state_vector[2]
                    - jnp.conj(state_vector[3]) * state_vector[3]
                )
            )
        else:
            raise ValueError(f"Observable {observable} not yet implemented")

    def state_vector(self, circuit: QuantumCircuit) -> jnp.ndarray:
        """Get the final state vector of the circuit.

        Args:
            circuit: The quantum circuit

        Returns:
            Final state vector as JAX array
        """
        results = self._simulate_circuit(circuit)
        return results["final_state"]

    def calc_probs(self, circuit: QuantumCircuit) -> jnp.ndarray:
        """Calculate measurement probabilities.

        Args:
            circuit: The quantum circuit

        Returns:
            Measurement probabilities as JAX array
        """
        results = self._simulate_circuit(circuit)
        return results["probabilities"]

    def _simulate_circuit(self, circuit: QuantumCircuit) -> Dict:
        """Internal method to simulate a circuit and return results.

        This method handles the complete pipeline:
        1. Transpile circuit to MLIR
        2. Optimize if requested
        3. Simulate with JAX runtime

        Args:
            circuit: The quantum circuit to simulate

        Returns:
            Dictionary with simulation results
        """
        # Step 1: Transpile circuit to MLIR
        mlir_code = circuit_to_mlir(circuit)

        # Step 2: Apply optimization passes if requested
        if self.optimize_circuit:
            mlir_code = apply_passes(mlir_code)

        # Step 3: Collect parameter values from circuit
        param_values = []
        param_ids_seen = set()
        for gate in circuit.gates:
            for param in gate.parameters:
                if param.id not in param_ids_seen:
                    param_values.append(param.initial_value)
                    param_ids_seen.add(param.id)

        # Step 4: Simulate with JAX runtime
        results = simulate_from_mlir(mlir_code, num_qubits=circuit.num_qubits, param_values=param_values)

        return results
