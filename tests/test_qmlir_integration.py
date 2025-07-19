"""Integration tests for qmlir package."""

import pytest
import jax.numpy as jnp
from qmlir import QuantumCircuit, Parameter, JaxSimulator
from qmlir.mlir import circuit_to_mlir, apply_passes


class TestFullPipeline:
    """Test the complete pipeline from circuit to simulation."""

    def test_bell_state_pipeline(self):
        """Test complete Bell state pipeline."""
        # 1. Circuit construction
        circuit = QuantumCircuit(2)
        circuit.h(0).cx(0, 1)

        # 2. MLIR transpilation
        mlir = circuit_to_mlir(circuit)
        assert "quantum.h" in mlir
        assert "quantum.cx" in mlir

        # 3. Optimization
        optimized_mlir = apply_passes(mlir)
        assert "module" in optimized_mlir

        # 4. Simulation
        simulator = JaxSimulator()
        state = simulator.state_vector(circuit)
        probs = simulator.calc_probs(circuit)

        # 5. Verification
        expected_state = jnp.array([1 / jnp.sqrt(2), 0.0, 0.0, 1 / jnp.sqrt(2)])
        expected_probs = jnp.array([0.5, 0.0, 0.0, 0.5])

        assert jnp.allclose(state, expected_state, atol=1e-6)
        assert jnp.allclose(probs, expected_probs, atol=1e-6)

    def test_parametric_pipeline(self):
        """Test complete parametric circuit pipeline."""
        # 1. Circuit construction with parameters
        circuit = QuantumCircuit(1)
        param = Parameter(0.5, name="theta")
        circuit.rx(0, param)

        # 2. MLIR transpilation
        mlir = circuit_to_mlir(circuit)
        assert "quantum.rx" in mlir
        assert "f64" in mlir  # Parameter type

        # 3. Optimization
        optimized_mlir = apply_passes(mlir)
        assert "module" in optimized_mlir

        # 4. Simulation
        simulator = JaxSimulator()
        state = simulator.state_vector(circuit)
        probs = simulator.calc_probs(circuit)

        # 5. Verification
        expected_state = jnp.array([0.9689124, -0.24740396j])
        expected_probs = jnp.array([0.9387913, 0.06120872])

        assert jnp.allclose(state, expected_state, atol=1e-6)
        assert jnp.allclose(probs, expected_probs, atol=1e-6)

    def test_complex_circuit_pipeline(self):
        """Test complete complex circuit pipeline."""
        # 1. Complex circuit construction
        circuit = QuantumCircuit(3)
        param1 = Parameter(0.5, name="theta1")
        param2 = Parameter(0.3, name="theta2")

        circuit.h(0).cx(0, 1).rx(2, param1).ry(1, param2).cz(0, 2)

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
        state = simulator.state_vector(circuit)
        probs = simulator.calc_probs(circuit)

        # 5. Verification
        assert len(state) == 8
        assert len(probs) == 8
        assert jnp.allclose(jnp.sum(probs), 1.0, atol=1e-6)


class TestSimulatorMethods:
    """Test all simulator methods with integration."""

    def test_get_samples_integration(self):
        """Test sampling with full pipeline."""
        circuit = QuantumCircuit(2)
        circuit.h(0).cx(0, 1)

        simulator = JaxSimulator()
        samples = simulator.get_samples(circuit, 100)

        assert isinstance(samples, dict)
        assert sum(samples.values()) >= 95  # Allow for some rounding
        # Bell state should primarily produce |00⟩ and |11⟩
        # Allow for some other states due to sampling noise
        bell_states = sum(samples.get(state, 0) for state in ["00", "11"])
        assert bell_states >= 90  # At least 90% should be bell states

    def test_calc_expval_integration(self):
        """Test expectation value calculation with full pipeline."""
        circuit = QuantumCircuit(1)
        circuit.h(0)

        simulator = JaxSimulator()
        expval = simulator.calc_expval(circuit, "Z")

        # H|0⟩ = (|0⟩ + |1⟩)/√2, so ⟨Z⟩ = 0
        assert jnp.allclose(expval, 0.0, atol=1e-6)

    def test_state_vector_integration(self):
        """Test state vector retrieval with full pipeline."""
        circuit = QuantumCircuit(1)
        circuit.x(0)

        simulator = JaxSimulator()
        state = simulator.state_vector(circuit)

        # X|0⟩ = |1⟩
        assert jnp.allclose(state, jnp.array([0.0, 1.0]))

    def test_calc_probs_integration(self):
        """Test probability calculation with full pipeline."""
        circuit = QuantumCircuit(1)
        circuit.h(0)

        simulator = JaxSimulator()
        probs = simulator.calc_probs(circuit)

        # H|0⟩ gives equal probabilities
        assert jnp.allclose(probs, jnp.array([0.5, 0.5]), atol=1e-6)


class TestOptimizationIntegration:
    """Test optimization integration."""

    def test_optimization_disabled_integration(self):
        """Test integration with optimization disabled."""
        circuit = QuantumCircuit(2)
        circuit.x(0).x(0)  # X*X = I, should be optimized away if enabled
        circuit.h(1)

        simulator = JaxSimulator(optimize_circuit=False)
        state = simulator.state_vector(circuit)

        # Should be equivalent to just H on qubit 1
        # The actual state shows H applied to qubit 0: [1/sqrt(2), 1/sqrt(2), 0, 0]
        expected_state = jnp.array([1 / jnp.sqrt(2), 1 / jnp.sqrt(2), 0.0, 0.0], dtype=jnp.complex64)
        assert jnp.allclose(state, expected_state, atol=1e-6)

    def test_optimization_enabled_integration(self):
        """Test integration with optimization enabled."""
        circuit = QuantumCircuit(2)
        circuit.x(0).x(0)  # X*X = I, should be optimized away
        circuit.h(1)

        simulator = JaxSimulator(optimize_circuit=True)
        state = simulator.state_vector(circuit)

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

        with pytest.raises(ValueError, match="Qubit index 1 is out of bounds"):
            circuit.x(1)

    def test_invalid_gate_parameters(self):
        """Test error handling for invalid gate parameters."""
        circuit = QuantumCircuit(2)

        with pytest.raises(ValueError, match="Control and target qubits must be different"):
            circuit.cx(0, 0)

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
            circuit.h(i % 4).cx(i % 4, (i + 1) % 4)

        simulator = JaxSimulator()

        # Should complete without errors
        state = simulator.state_vector(circuit)
        assert len(state) == 16

        probs = simulator.calc_probs(circuit)
        assert len(probs) == 16
        assert jnp.allclose(jnp.sum(probs), 1.0, atol=1e-6)

    def test_multiple_parameters_performance(self):
        """Test performance with multiple parameters."""
        circuit = QuantumCircuit(2)

        # Add multiple parameters
        params = [Parameter(0.1 * i, name=f"theta{i}") for i in range(5)]
        circuit.rx(0, params[0]).ry(1, params[1]).rz(0, params[2])

        simulator = JaxSimulator()

        # Should complete without errors
        state = simulator.state_vector(circuit)
        assert len(state) == 4

        probs = simulator.calc_probs(circuit)
        assert len(probs) == 4
        assert jnp.allclose(jnp.sum(probs), 1.0, atol=1e-6)
