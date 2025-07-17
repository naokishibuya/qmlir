"""Test basic circuit construction and AST functionality."""

# Import configuration to set up paths
from qmlir import QuantumCircuit, QuantumGate


class TestQuantumCircuit:
    """Test the QuantumCircuit class."""

    def test_empty_circuit(self):
        """Test creating an empty circuit."""
        circuit = QuantumCircuit(1)
        assert len(circuit.gates) == 0
        assert str(circuit) == "QuantumCircuit(1 qubits):"

    def test_single_gate(self):
        """Test adding a single gate."""
        circuit = QuantumCircuit(1)
        circuit.h(0)

        assert len(circuit.gates) == 1
        assert circuit.gates[0].name == "h"
        assert circuit.gates[0].q == (0,)

    def test_method_chaining(self):
        """Test that gate methods support chaining."""
        circuit = QuantumCircuit(2)
        result = circuit.h(0).x(1).cx(0, 1)

        assert result is circuit  # Should return self
        assert len(circuit.gates) == 3

        # Verify gate sequence
        assert circuit.gates[0].name == "h"
        assert circuit.gates[0].q == (0,)

        assert circuit.gates[1].name == "x"
        assert circuit.gates[1].q == (1,)

        assert circuit.gates[2].name == "cx"
        assert circuit.gates[2].q == (0, 1)

    def test_bell_state_circuit(self):
        """Test creating a Bell state circuit."""
        circuit = QuantumCircuit(2)
        circuit.h(0).cx(0, 1)

        assert len(circuit.gates) == 2
        assert circuit.gates[0].name == "h"
        assert circuit.gates[0].q == (0,)
        assert circuit.gates[1].name == "cx"
        assert circuit.gates[1].q == (0, 1)

    def test_double_x_circuit(self):
        """Test creating a double X circuit."""
        circuit = QuantumCircuit(1)
        circuit.x(0).x(0)

        assert len(circuit.gates) == 2
        assert all(gate.name == "x" for gate in circuit.gates)
        assert all(gate.q == (0,) for gate in circuit.gates)

    def test_multi_qubit_circuit(self):
        """Test creating a multi-qubit circuit."""
        circuit = QuantumCircuit(3)
        circuit.h(0).h(1).x(2).cx(0, 1).cx(1, 2)

        assert len(circuit.gates) == 5

        # Check gate types and qubits
        expected_gates = [("h", (0,)), ("h", (1,)), ("x", (2,)), ("cx", (0, 1)), ("cx", (1, 2))]

        for gate, (expected_name, expected_qubits) in zip(circuit.gates, expected_gates):
            assert gate.name == expected_name
            assert gate.q == expected_qubits


class TestGate:
    """Test the Gate class."""

    def test_gate_creation(self):
        """Test creating different types of gates."""
        # Single qubit gates
        h_gate = QuantumGate("h", 0)
        assert h_gate.name == "h"
        assert h_gate.q == (0,)

        x_gate = QuantumGate("x", 1)
        assert x_gate.name == "x"
        assert x_gate.q == (1,)

        # Two qubit gate
        cx_gate = QuantumGate("cx", 0, 1)
        assert cx_gate.name == "cx"
        assert cx_gate.q == (0, 1)

    def test_gate_repr(self):
        """Test gate string representation."""
        h_gate = QuantumGate("h", 0)
        assert repr(h_gate) == "Gate(h, 0)"

        cx_gate = QuantumGate("cx", 0, 1)
        assert repr(cx_gate) == "Gate(cx, 0, 1)"
