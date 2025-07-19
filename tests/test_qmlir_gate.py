"""Tests for qmlir.gate module."""

import pytest
from qmlir.gate import QuantumGate, AVAILABLE_QUANTUM_GATES
from qmlir.parameter import Parameter


class TestQuantumGate:
    """Test QuantumGate class."""

    def test_gate_creation_basic(self):
        """Test basic gate creation."""
        gate = QuantumGate("x", 0)
        assert gate.name == "x"
        assert gate.q == (0,)
        assert gate.parameters == []

    def test_gate_creation_with_parameters(self):
        """Test gate creation with parameters."""
        param = Parameter(0.5, name="theta")
        gate = QuantumGate("rx", 0, parameters=[param])
        assert gate.name == "rx"
        assert gate.q == (0,)
        assert gate.parameters == [param]

    def test_gate_creation_two_qubit(self):
        """Test two-qubit gate creation."""
        gate = QuantumGate("cx", 0, 1)
        assert gate.name == "cx"
        assert gate.q == (0, 1)
        assert gate.parameters == []

    def test_invalid_gate_name(self):
        """Test that invalid gate names raise an error."""
        with pytest.raises(AssertionError, match="Unknown gate: invalid"):
            QuantumGate("invalid", 0)

    def test_gate_description(self):
        """Test gate description property."""
        gate = QuantumGate("h", 0)
        assert gate.description == "Hadamard"

    def test_gate_repr_basic(self):
        """Test gate string representation for basic gates."""
        gate = QuantumGate("x", 0)
        assert repr(gate) == "X|0⟩"

    def test_gate_repr_two_qubit(self):
        """Test gate string representation for two-qubit gates."""
        gate = QuantumGate("cx", 0, 1)
        assert repr(gate) == "CX|0, 1⟩"

    def test_gate_repr_with_parameters(self):
        """Test gate string representation with parameters."""
        param = Parameter(0.5, name="theta")
        gate = QuantumGate("rx", 0, parameters=[param])
        assert "RX(theta=0.5)|0⟩" in repr(gate)


class TestAvailableQuantumGates:
    """Test AVAILABLE_QUANTUM_GATES constant."""

    def test_all_gates_present(self):
        """Test that all expected gates are present."""
        expected_gates = {"i", "x", "y", "z", "h", "s", "t", "sdg", "tdg", "cx", "cy", "cz", "rx", "ry", "rz"}
        assert set(AVAILABLE_QUANTUM_GATES.keys()) == expected_gates

    def test_gate_descriptions(self):
        """Test that gate descriptions are meaningful."""
        assert AVAILABLE_QUANTUM_GATES["x"] == "Pauli-X"
        assert AVAILABLE_QUANTUM_GATES["h"] == "Hadamard"
        assert AVAILABLE_QUANTUM_GATES["cx"] == "CNOT"
        assert AVAILABLE_QUANTUM_GATES["rx"] == "Rotation-X"
