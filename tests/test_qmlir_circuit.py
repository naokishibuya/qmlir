import pytest
from qmlir.circuit import QuantumCircuit
from qmlir.operator import X, CX


class TestQuantumCircuit:
    def test_circuit_initialization(self):
        circuit = QuantumCircuit(3)
        assert circuit.num_qubits == 3
        assert len(circuit.operators) == 0

    def test_invalid_qubit_number(self):
        with pytest.raises(ValueError, match="must have at least one qubit"):
            QuantumCircuit(0)

    def test_operator_automatic_append(self):
        with QuantumCircuit(2) as circuit:
            op1 = X(0)
            op2 = CX(0, 1)
        assert circuit.num_qubits == 2
        assert len(circuit.operators) == 2
        assert circuit.operators == [op1, op2]

    def test_len_and_iteration(self):
        circuit = QuantumCircuit(1)
        with circuit:
            X(0)
        assert circuit.num_qubits == 1
        assert len(circuit.operators) == 1

    def test_invalid_qubit(self):
        with QuantumCircuit(2):
            with pytest.raises(ValueError, match="out of range for circuit with 2 qubits."):
                X(2)

    def test_str_representation(self):
        circuit = QuantumCircuit(1)
        with circuit:
            X(0)
        result = str(circuit)
        assert "QuantumCircuit(1 qubits):" in result
        assert "X|" in result

    def test_repr_representation(self):
        circuit = QuantumCircuit(2)
        with circuit:
            X(0)
            CX(0, 1)
        text = repr(circuit)
        assert text.startswith("QuantumCircuit(num_qubits=2")

    def test_nested_circuit(self):
        with QuantumCircuit(2):
            with pytest.raises(RuntimeError, match="Only one active quantum circuit can exist at a time."):
                with QuantumCircuit(1):
                    pass
