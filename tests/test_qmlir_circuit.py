"""Tests for qmlir.circuit module."""

import pytest
from qmlir.circuit import QuantumCircuit
from qmlir.parameter import Parameter


class TestQuantumCircuit:
    """Test QuantumCircuit class."""

    def test_circuit_creation(self):
        """Test basic circuit creation."""
        circuit = QuantumCircuit(2)
        assert circuit.num_qubits == 2
        assert len(circuit.gates) == 0

    def test_single_qubit_gates(self):
        """Test all single-qubit gates."""
        circuit = QuantumCircuit(1)

        # Test all single-qubit gates
        gates = ["i", "x", "y", "z", "h", "s", "t", "sdg", "tdg"]
        for gate_name in gates:
            circuit = QuantumCircuit(1)
            gate_method = getattr(circuit, gate_name)
            result = gate_method(0)
            assert result is circuit  # Method chaining
            assert len(circuit.gates) == 1
            assert circuit.gates[0].name == gate_name
            assert circuit.gates[0].q == (0,)

    def test_two_qubit_gates(self):
        """Test all two-qubit gates."""
        circuit = QuantumCircuit(2)

        # Test all two-qubit gates
        gates = ["cx", "cy", "cz"]
        for gate_name in gates:
            circuit = QuantumCircuit(2)
            gate_method = getattr(circuit, gate_name)
            result = gate_method(0, 1)
            assert result is circuit  # Method chaining
            assert len(circuit.gates) == 1
            assert circuit.gates[0].name == gate_name
            assert circuit.gates[0].q == (0, 1)

    def test_rotation_gates(self):
        """Test rotation gates with parameters."""
        circuit = QuantumCircuit(1)
        param = Parameter(0.5, name="theta")

        # Test all rotation gates
        gates = ["rx", "ry", "rz"]
        for gate_name in gates:
            circuit = QuantumCircuit(1)
            gate_method = getattr(circuit, gate_name)
            result = gate_method(0, param)
            assert result is circuit  # Method chaining
            assert len(circuit.gates) == 1
            assert circuit.gates[0].name == gate_name
            assert circuit.gates[0].q == (0,)
            assert circuit.gates[0].parameters == [param]

    def test_method_chaining(self):
        """Test method chaining for multiple gates."""
        circuit = QuantumCircuit(2)
        result = circuit.h(0).cx(0, 1).x(1)
        assert result is circuit
        assert len(circuit.gates) == 3
        assert circuit.gates[0].name == "h"
        assert circuit.gates[1].name == "cx"
        assert circuit.gates[2].name == "x"

    def test_qubit_validation(self):
        """Test qubit index validation."""
        circuit = QuantumCircuit(2)

        # Test negative qubit index
        with pytest.raises(ValueError, match="Qubit index -1 is out of bounds"):
            circuit.x(-1)

        # Test qubit index too large
        with pytest.raises(ValueError, match="Qubit index 2 is out of bounds"):
            circuit.x(2)

    def test_controlled_gate_validation(self):
        """Test controlled gate validation."""
        circuit = QuantumCircuit(2)

        # Test same control and target
        with pytest.raises(ValueError, match="Control and target qubits must be different"):
            circuit.cx(0, 0)

    def test_circuit_repr(self):
        """Test circuit string representation."""
        circuit = QuantumCircuit(2)
        circuit.h(0).cx(0, 1)
        repr_str = repr(circuit)
        assert "QuantumCircuit(2 qubits, 2 gates)" in repr_str

    def test_circuit_str(self):
        """Test circuit string conversion."""
        circuit = QuantumCircuit(2)
        circuit.h(0).cx(0, 1)
        str_result = str(circuit)
        assert "QuantumCircuit(2 qubits):" in str_result
        assert "H|0⟩" in str_result
        assert "CX|0, 1⟩" in str_result

    def test_empty_circuit(self):
        """Test empty circuit representation."""
        circuit = QuantumCircuit(3)
        assert repr(circuit) == "QuantumCircuit(3 qubits, 0 gates)"
        assert str(circuit).strip() == "QuantumCircuit(3 qubits):"

    def test_complex_circuit(self):
        """Test complex circuit with many gates."""
        circuit = QuantumCircuit(3)
        param = Parameter(0.5, name="theta")

        circuit.h(0).cx(0, 1).rx(2, param).y(1).cz(0, 2)
        assert len(circuit.gates) == 5
        assert circuit.gates[0].name == "h"
        assert circuit.gates[1].name == "cx"
        assert circuit.gates[2].name == "rx"
        assert circuit.gates[3].name == "y"
        assert circuit.gates[4].name == "cz"
