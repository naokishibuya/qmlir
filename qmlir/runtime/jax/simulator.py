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
        bitstrings = [format(i, f"0{int(jnp.log2(num_states))}b") for i in samples]

        return dict(Counter(bitstrings))

    def expectation(self, circuit: QuantumCircuit, observable: str) -> float:
        """Calculate expectation value of an observable."""
        # For now, implement basic Pauli observables
        state_vector = self.statevector(circuit)

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

    def statevector(self, circuit: QuantumCircuit) -> jnp.ndarray:
        """Get the final state vector of the circuit."""
        results = self._simulate_circuit(circuit)
        return results["final_state"]

    def probabilities(self, circuit: QuantumCircuit) -> jnp.ndarray:
        """Calculate measurement probabilities."""
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
        for gate in circuit.operators:
            for param in gate.parameters:
                if param.id not in param_ids_seen:
                    param_values.append(param.value)
                    param_ids_seen.add(param.id)

        # Step 4: Simulate with JAX runtime
        results = simulate_from_mlir(mlir_code, num_qubits=circuit.num_qubits, param_values=param_values)

        return results
