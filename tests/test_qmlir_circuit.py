import pytest
from qmlir.circuit import QuantumCircuit
from qmlir.bit import QuantumBit
from qmlir.operator import X, CX


class TestQuantumCircuit:
    def test_circuit_initialization(self):
        circuit = QuantumCircuit(3)
        assert circuit.num_qubits == 3
        assert len(circuit.qubits) == 3
        for i, qubit in enumerate(circuit.qubits):
            assert isinstance(qubit, QuantumBit)
            assert qubit.index == i
            assert qubit.circuit is circuit

    def test_invalid_qubit_number(self):
        with pytest.raises(ValueError, match="must have at least one qubit"):
            QuantumCircuit(0)

    def test_operator_automatic_append(self):
        circuit = QuantumCircuit(2)
        q0, q1 = circuit.qubits[0], circuit.qubits[1]
        op1 = X(q0)
        op2 = CX(q0, q1)
        assert len(circuit.operators) == 2
        assert circuit.operators == [op1, op2]

    def test_len_and_iteration(self):
        circuit = QuantumCircuit(1)
        q0 = circuit.qubits[0]
        X(q0)
        assert len(circuit.qubits) == 1
        assert len(circuit.operators) == 1
        for operator in circuit.operators:
            assert isinstance(str(operator), str)

    def test_getitem_valid(self):
        circuit = QuantumCircuit(2)
        assert isinstance(circuit.qubits[0], QuantumBit)
        assert circuit.qubits[1].index == 1

    def test_getitem_invalid(self):
        circuit = QuantumCircuit(2)
        with pytest.raises(IndexError, match="list index out of range"):
            _ = circuit.qubits[2]

    def test_str_representation(self):
        circuit = QuantumCircuit(1)
        q0 = circuit.qubits[0]
        X(q0)
        result = str(circuit)
        assert "QuantumCircuit(1 qubits):" in result
        assert "X|" in result

    def test_repr_representation(self):
        circuit = QuantumCircuit(2)
        q0 = circuit.qubits[0]
        X(q0)
        text = repr(circuit)
        assert text.startswith("QuantumCircuit(num_qubits=2")
