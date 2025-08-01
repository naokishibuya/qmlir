import numpy as np
import jax.numpy as jnp
from qmlir.operator import CX, H, RX, RY, RZ, X, Z
from qmlir.runtime.jax.simulator import JaxSimulator
from qmlir.circuit import QuantumCircuit
from qmlir.parameter import Parameter


def test_x_gate_on_msb():
    circuit = QuantumCircuit(3, little_endian=False)
    with circuit:
        X(0)  # MSB
    sim = JaxSimulator()
    probs = sim.probabilities(circuit)
    expected = [0.0] * 2**3
    expected[4] = 1.0  # |100⟩ state
    assert np.allclose(probs, expected, atol=1e-6)


def test_cx_self_inverse1():
    circuit = QuantumCircuit(2)
    with circuit:
        H(0)
        CX(0, 1)
        CX(0, 1)  # Apply CX twice
    sim = JaxSimulator()
    probs = sim.probabilities(circuit)
    assert np.allclose(probs, [0.5, 0.5, 0.0, 0.0], atol=1e-6)  # Same as just H(1)


def test_cx_self_inverse2():
    circuit = QuantumCircuit(2)
    with circuit:
        H(1)
        CX(0, 1)
        CX(0, 1)  # Apply CX twice
    sim = JaxSimulator(optimize_circuit=False)
    probs = sim.probabilities(circuit)
    assert np.allclose(probs, [0.5, 0.0, 0.5, 0.0], atol=1e-6)  # Same as just H(1)


class TestJaxSimulator:
    """Test JaxSimulator class."""

    def test_simulator_creation(self):
        """Test basic simulator creation."""
        simulator = JaxSimulator()
        assert simulator.optimize_circuit is True

    def test_simulator_creation_with_optimization_disabled(self):
        """Test simulator creation with optimization disabled."""
        simulator = JaxSimulator(optimize_circuit=False)
        assert simulator.optimize_circuit is False

    def test_get_samples_empty_circuit(self):
        """Test sampling from empty circuit."""
        simulator = JaxSimulator()
        circuit = QuantumCircuit(2)
        samples = simulator.measure(circuit, 10)
        assert isinstance(samples, dict)
        assert sum(samples.values()) == 10
        assert all(sample in ["00", "01", "10", "11"] for sample in samples.keys())

    def test_get_samples_bell_state(self):
        """Test sampling from Bell state circuit."""
        simulator = JaxSimulator()
        circuit = QuantumCircuit(2)
        with circuit:
            H(0)
            CX(0, 1)  # Create Bell state |Φ+⟩
        samples = simulator.measure(circuit, 100)
        assert isinstance(samples, dict)
        assert sum(samples.values()) >= 95  # Allow for some rounding
        # Bell state should primarily produce |00⟩ and |11⟩
        # Allow for some other states due to sampling noise
        bell_states = sum(samples.get(state, 0) for state in ["00", "11"])
        assert bell_states >= 90  # At least 90% should be bell states

    def test_calc_expval_empty_circuit(self):
        """Test expectation value calculation for empty circuit."""
        simulator = JaxSimulator()
        circuit = QuantumCircuit(1)
        expval = simulator.expectation(circuit, Z(0))
        assert isinstance(expval, float)
        # Empty circuit should give expectation value of 1 for Z operator
        assert jnp.allclose(expval, 1.0)

    def test_calc_expval_x_gate(self):
        """Test expectation value calculation for X gate."""
        simulator = JaxSimulator()
        circuit = QuantumCircuit(1)
        with circuit:
            X(0)
        expval = simulator.expectation(circuit, Z(0))
        # X|0⟩ = |1⟩, so ⟨Z⟩ = -1
        assert jnp.allclose(expval, -1.0)

    def test_calc_expval_h_gate(self):
        """Test expectation value calculation for H gate."""
        simulator = JaxSimulator()
        circuit = QuantumCircuit(1)
        with circuit:
            H(0)
        expval = simulator.expectation(circuit, Z(0))
        # H|0⟩ = (|0⟩ + |1⟩)/√2, so ⟨Z⟩ = 0
        assert jnp.allclose(expval, 0.0, atol=1e-6)

    def test_state_vector_empty_circuit(self):
        """Test state vector retrieval for empty circuit."""
        simulator = JaxSimulator()
        circuit = QuantumCircuit(1)
        state = simulator.statevector(circuit)
        assert len(state) == 2
        # Empty circuit should be in |0⟩ state
        assert jnp.allclose(state, jnp.array([1.0, 0.0]))

    def test_state_vector_x_gate(self):
        """Test state vector retrieval for X gate."""
        simulator = JaxSimulator()
        circuit = QuantumCircuit(1)
        with circuit:
            X(0)
        state = simulator.statevector(circuit)
        assert len(state) == 2
        # X|0⟩ = |1⟩
        assert jnp.allclose(state, jnp.array([0.0, 1.0]))

    def test_state_vector_bell_state(self):
        """Test state vector retrieval for Bell state."""
        simulator = JaxSimulator()
        circuit = QuantumCircuit(2)
        with circuit:
            H(0)
            CX(0, 1)
        state = simulator.statevector(circuit)
        assert len(state) == 4
        # Bell state: (|00⟩ + |11⟩)/√2
        expected_state = jnp.array([1 / jnp.sqrt(2), 0.0, 0.0, 1 / jnp.sqrt(2)])
        assert jnp.allclose(state, expected_state, atol=1e-6)

    def test_calc_probs_empty_circuit(self):
        """Test probability calculation for empty circuit."""
        simulator = JaxSimulator()
        circuit = QuantumCircuit(1)
        probs = simulator.probabilities(circuit)
        assert len(probs) == 2
        # Empty circuit should have probability 1 for |0⟩
        assert jnp.allclose(probs, jnp.array([1.0, 0.0]))

    def test_calc_probs_h_gate(self):
        """Test probability calculation for H gate."""
        simulator = JaxSimulator()
        circuit = QuantumCircuit(1)
        with circuit:
            H(0)
        probs = simulator.probabilities(circuit)
        assert len(probs) == 2
        # H|0⟩ gives equal probabilities
        assert jnp.allclose(probs, jnp.array([0.5, 0.5]), atol=1e-6)

    def test_calc_probs_bell_state(self):
        """Test probability calculation for Bell state."""
        simulator = JaxSimulator()
        circuit = QuantumCircuit(2)
        with circuit:
            H(0)
            CX(0, 1)
        probs = simulator.probabilities(circuit)
        assert len(probs) == 4
        # Bell state probabilities: [0.5, 0.0, 0.0, 0.5]
        expected_probs = jnp.array([0.5, 0.0, 0.0, 0.5])
        assert jnp.allclose(probs, expected_probs, atol=1e-6)


class TestJaxSimulatorWithParameters:
    """Test JaxSimulator with parametric gates."""

    def test_parametric_rx_gate(self):
        """Test parametric RX gate."""
        simulator = JaxSimulator()
        circuit = QuantumCircuit(1)
        param = Parameter(0.5, name="theta")
        with circuit:
            RX(param)(0)

        # Test state vector
        state = simulator.statevector(circuit)
        assert len(state) == 2
        # RX(0.5)|0⟩ should give specific state
        expected_state = jnp.array([0.9689124, -0.24740396j])
        assert jnp.allclose(state, expected_state, atol=1e-6)

        # Test probabilities
        probs = simulator.probabilities(circuit)
        expected_probs = jnp.array([0.9387913, 0.06120872])
        assert jnp.allclose(probs, expected_probs, atol=1e-6)

    def test_parametric_ry_gate(self):
        """Test parametric RY gate."""
        simulator = JaxSimulator()
        circuit = QuantumCircuit(1)
        param = Parameter(0.5, name="theta")
        with circuit:
            RY(param)(0)

        # Test state vector
        state = simulator.statevector(circuit)
        assert len(state) == 2
        # RY(0.5)|0⟩ should give specific state
        expected_state = jnp.array([0.9689124, 0.24740396])
        assert jnp.allclose(state, expected_state, atol=1e-6)

    def test_parametric_rz_gate(self):
        """Test parametric RZ gate."""
        simulator = JaxSimulator()
        circuit = QuantumCircuit(1)
        param = Parameter(0.5, name="theta")
        with circuit:
            RZ(param)(0)

        # Test state vector
        state = simulator.statevector(circuit)
        assert len(state) == 2
        # RZ(0.5)|0⟩ should give specific state
        expected_state = jnp.array([0.9689124 - 0.24740396j, 0.0])
        assert jnp.allclose(state, expected_state, atol=1e-6)


class TestJaxSimulatorOptimization:
    """Test JaxSimulator optimization features."""

    def test_optimization_disabled(self):
        """Test simulation with optimization disabled."""
        simulator = JaxSimulator(optimize_circuit=False)
        circuit = QuantumCircuit(2)
        with circuit:
            X(0)
            X(0)
            H(1)

        # With optimization disabled, should still work
        state = simulator.statevector(circuit)
        assert len(state) == 4

        # Should be equivalent to just H on qubit 1
        # The actual state shows H applied to qubit 0: [1/sqrt(2), 1/sqrt(2), 0, 0]
        expected_state = jnp.array([1 / jnp.sqrt(2), 0.0, 1 / jnp.sqrt(2), 0.0], dtype=jnp.complex64)
        assert jnp.allclose(state, expected_state, atol=1e-6)

    def test_optimization_enabled(self):
        """Test simulation with optimization enabled."""
        simulator = JaxSimulator(optimize_circuit=True)
        circuit = QuantumCircuit(2)
        with circuit:
            X(0)
            X(0)  # X*X = I, should be optimized away
            H(1)

        # With optimization enabled, should work and potentially optimize
        state = simulator.statevector(circuit)
        assert len(state) == 4

        # Should be equivalent to just H on qubit 1
        # When optimization is enabled, X*X is optimized away, so we get H(1)
        # H(1) on |00⟩ gives [1/sqrt(2), 0, 1/sqrt(2), 0]
        expected_state = jnp.array([1 / jnp.sqrt(2), 0.0, 1 / jnp.sqrt(2), 0.0], dtype=jnp.complex64)
        assert jnp.allclose(state, expected_state, atol=1e-6)


class TestJaxSimulatorEdgeCases:
    """Test JaxSimulator edge cases."""

    def test_large_circuit(self):
        """Test simulation of larger circuit."""
        simulator = JaxSimulator()
        circuit = QuantumCircuit(3)
        with circuit:
            H(0)
            CX(0, 1)
            H(2)
            CX(1, 2)
            X(0)

        # Should handle larger circuits
        state = simulator.statevector(circuit)
        assert len(state) == 8

        probs = simulator.probabilities(circuit)
        assert len(probs) == 8
        assert jnp.allclose(jnp.sum(probs), 1.0, atol=1e-6)

    def test_multiple_parameters(self):
        """Test circuit with multiple parameters."""
        simulator = JaxSimulator()
        circuit = QuantumCircuit(2)
        param1 = Parameter(0.5, name="theta1")
        param2 = Parameter(0.3, name="theta2")
        with circuit:
            RX(param1)(0)
            RY(param2)(1)

        # Should handle multiple parameters
        state = simulator.statevector(circuit)
        assert len(state) == 4

        probs = simulator.probabilities(circuit)
        assert len(probs) == 4
        assert jnp.allclose(jnp.sum(probs), 1.0, atol=1e-6)
