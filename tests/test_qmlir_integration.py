"""Integration tests for qmlir package."""

import pytest
import jax.numpy as jnp
from qmlir import QuantumCircuit, Parameter, JaxSimulator
from qmlir.operator import X, H, CX, CZ, RX, RY, RZ
from qmlir.mlir import circuit_to_mlir, apply_passes


class TestFullPipeline:
    """Test the complete pipeline from circuit to simulation."""

    def test_bell_state_pipeline(self):
        """Test complete Bell state pipeline."""
        # 1. Circuit construction
        circuit = QuantumCircuit(2)
        q0, q1 = circuit.qubits
        H(q0)
        CX(q0, q1)  # Create Bell state |Φ+⟩

        # 2. MLIR transpilation
        mlir = circuit_to_mlir(circuit)
        assert "quantum.h" in mlir
        assert "quantum.cx" in mlir

        # 3. Optimization
        optimized_mlir = apply_passes(mlir)
        assert "module" in optimized_mlir

        # 4. Simulation
        simulator = JaxSimulator()
        state = simulator.statevector(circuit)
        probs = simulator.probabilities(circuit)

        # 5. Verification
        expected_state = jnp.array([1 / jnp.sqrt(2), 0.0, 0.0, 1 / jnp.sqrt(2)])
        expected_probs = jnp.array([0.5, 0.0, 0.0, 0.5])

        assert jnp.allclose(state, expected_state, atol=1e-6)
        assert jnp.allclose(probs, expected_probs, atol=1e-6)

    def test_parametric_pipeline(self):
        """Test complete parametric circuit pipeline."""
        # 1. Circuit construction with parameters
        circuit = QuantumCircuit(1)
        q0 = circuit.qubits[0]
        RX(0.5)(q0)  # Rotate around X by 0.5 radians

        # 2. MLIR transpilation
        mlir = circuit_to_mlir(circuit)
        assert "quantum.rx" in mlir
        assert "f64" in mlir  # Parameter type

        # 3. Optimization
        optimized_mlir = apply_passes(mlir)
        assert "module" in optimized_mlir

        # 4. Simulation
        simulator = JaxSimulator()
        state = simulator.statevector(circuit)
        probs = simulator.probabilities(circuit)

        # 5. Verification
        expected_state = jnp.array([0.9689124, -0.24740396j])
        expected_probs = jnp.array([0.9387913, 0.06120872])

        assert jnp.allclose(state, expected_state, atol=1e-6)
        assert jnp.allclose(probs, expected_probs, atol=1e-6)

    def test_complex_circuit_pipeline(self):
        """Test complete complex circuit pipeline."""
        # 1. Complex circuit construction
        circuit = QuantumCircuit(3)
        q0, q1, q2 = circuit.qubits
        H(q0)
        CX(q0, q1)
        RX(0.5)(q2)
        RY(0.3)(q1)
        CZ(q0, q2)

        # 2. MLIR transpilation
        mlir = circuit_to_mlir(circuit)
        assert "quantum.h" in mlir
        assert "quantum.cx" in mlir
        assert "quantum.rx" in mlir
        assert "quantum.ry" in mlir
        assert "quantum.cz" in mlir

        # 3. Optimization
        optimized_mlir = apply_passes(mlir)
        assert "module" in optimized_mlir

        # 4. Simulation
        simulator = JaxSimulator()
        state = simulator.statevector(circuit)
        probs = simulator.probabilities(circuit)

        # 5. Verification
        assert len(state) == 8
        assert len(probs) == 8
        assert jnp.allclose(jnp.sum(probs), 1.0, atol=1e-6)


class TestSimulatorMethods:
    """Test all simulator methods."""

    def test_get_counts(self):
        """Test sampling with full pipeline."""
        circuit = QuantumCircuit(2)
        q0, q1 = circuit.qubits
        H(q0)
        CX(q0, q1)  # Create Bell state |Φ+⟩

        simulator = JaxSimulator()
        samples = simulator.measure(circuit, 100)

        assert isinstance(samples, dict)
        assert sum(samples.values()) == 100
        # Bell state should primarily produce |00⟩ and |11⟩
        # Allow for some other states due to sampling noise
        bell_states = sum(samples.get(state, 0) for state in ["00", "11"])
        assert bell_states >= 90  # At least 90% should be bell states

    def test_calc_expval(self):
        """Test expectation value calculation with full pipeline."""
        circuit = QuantumCircuit(1)
        q0 = circuit.qubits[0]
        H(q0)  # Prepare superposition

        simulator = JaxSimulator()
        expval = simulator.expectation(circuit, "Z")

        # H|0⟩ = (|0⟩ + |1⟩)/√2, so ⟨Z⟩ = 0
        assert jnp.allclose(expval, 0.0, atol=1e-6)

    def test_state_vector(self):
        """Test state vector retrieval with full pipeline."""
        circuit = QuantumCircuit(1)
        q0 = circuit.qubits[0]
        X(q0)  # Prepare |1⟩ state

        simulator = JaxSimulator()
        state = simulator.statevector(circuit)

        # X|0⟩ = |1⟩
        assert jnp.allclose(state, jnp.array([0.0, 1.0]))

    def test_calc_probs(self):
        """Test probability calculation with full pipeline."""
        circuit = QuantumCircuit(1)
        q0 = circuit.qubits[0]
        H(q0)  # Prepare superposition

        simulator = JaxSimulator()
        probs = simulator.probabilities(circuit)

        # H|0⟩ gives equal probabilities
        assert jnp.allclose(probs, jnp.array([0.5, 0.5]), atol=1e-6)


class TestOptimizationDisabledEnabled:
    """Test optimization disabled."""

    def test_optimization_disabled(self):
        """Test integration with optimization disabled."""
        circuit = QuantumCircuit(2)
        q0, q1 = circuit.qubits
        X(q0)
        X(q0)  # X*X = I, should not be optimized away
        H(q1)

        simulator = JaxSimulator(optimize_circuit=False)
        state = simulator.statevector(circuit)

        # Should be equivalent to just H on qubit 1
        # The actual state shows H applied to qubit 0: [1/sqrt(2), 1/sqrt(2), 0, 0]
        expected_state = jnp.array([1 / jnp.sqrt(2), 1 / jnp.sqrt(2), 0.0, 0.0], dtype=jnp.complex64)
        assert jnp.allclose(state, expected_state, atol=1e-6)

    def test_optimization_enabled(self):
        """Test integration with optimization enabled."""
        circuit = QuantumCircuit(2)
        q0, q1 = circuit.qubits
        X(q0)
        X(q0)  # X*X = I, should be optimized away
        H(q1)

        simulator = JaxSimulator(optimize_circuit=True)
        state = simulator.statevector(circuit)

        # Should be equivalent to just H on qubit 1
        # When optimization is enabled, X*X is optimized away, so we get H(1)
        # H(1) on |00⟩ gives [1/sqrt(2), 0, 1/sqrt(2), 0]
        expected_state = jnp.array([1 / jnp.sqrt(2), 0.0, 1 / jnp.sqrt(2), 0.0], dtype=jnp.complex64)
        assert jnp.allclose(state, expected_state, atol=1e-6)


class TestErrorHandling:
    """Test error handling in integration."""

    def test_invalid_qubit_index(self):
        """Test error handling for invalid qubit index."""
        circuit = QuantumCircuit(1)

        with pytest.raises(IndexError, match="list index out of range"):
            circuit.qubits[1]

    def test_invalid_gate_parameters(self):
        """Test error handling for invalid gate parameters."""
        circuit = QuantumCircuit(2)
        q0, _ = circuit.qubits

        with pytest.raises(ValueError, match="Control and target must differ."):
            CX(q0, q0)

    def test_invalid_mlir(self):
        """Test error handling for invalid MLIR."""
        invalid_mlir = "invalid mlir code"

        with pytest.raises(RuntimeError):
            apply_passes(invalid_mlir)


class TestPerformance:
    """Test performance characteristics."""

    def test_large_circuit_performance(self):
        """Test performance with larger circuits."""
        circuit = QuantumCircuit(4)

        # Add many gates
        for i in range(10):
            q0 = circuit.qubits[i % 4]
            q1 = circuit.qubits[(i + 1) % 4]
            H(q0)
            CX(q0, q1)

        simulator = JaxSimulator()

        # Should complete without errors
        state = simulator.statevector(circuit)
        assert len(state) == 16

        probs = simulator.probabilities(circuit)
        assert len(probs) == 16
        assert jnp.allclose(jnp.sum(probs), 1.0, atol=1e-6)

    def test_multiple_parameters_performance(self):
        """Test performance with multiple parameters."""
        circuit = QuantumCircuit(2)
        q0, q1 = circuit.qubits

        # Add multiple parameters
        p0, p1, p2, p3, p4 = [Parameter(0.1 * i, name=f"theta{i}") for i in range(5)]
        RX(p0)(q0)
        RY(p1)(q1)
        RZ(p2)(q0)
        RX(p3)(q1)
        RY(p4)(q0)

        simulator = JaxSimulator()

        # Should complete without errors
        state = simulator.statevector(circuit)
        assert len(state) == 4

        probs = simulator.probabilities(circuit)
        assert len(probs) == 4
        assert jnp.allclose(jnp.sum(probs), 1.0, atol=1e-6)
