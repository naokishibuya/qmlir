from qmlir.circuit import QuantumCircuit


class TestQuantumBit:
    def setup_method(self):
        self.circuit = QuantumCircuit(3)
        self.q0 = self.circuit.qubits[0]
        self.q1 = self.circuit.qubits[1]

    def test_index_and_circuit(self):
        assert self.q0.index == 0
        assert self.q0.circuit is self.circuit
        assert self.q1.index == 1
        assert self.q1.circuit is self.circuit

    def test_equality_same_bit(self):
        q0_alias = self.circuit.qubits[0]
        assert self.q0 == q0_alias

    def test_inequality_different_index(self):
        assert self.q0 != self.q1

    def test_inequality_different_circuit(self):
        other_circuit = QuantumCircuit(3)
        other_q0 = other_circuit.qubits[0]
        assert self.q0 != other_q0

    def test_hashing(self):
        bit_set = {self.q0, self.q1}
        assert self.q0 in bit_set
        assert self.q1 in bit_set
        assert self.circuit.qubits[0] in bit_set

    def test_str_and_repr(self):
        assert str(self.q0) == "q[0]"
        assert repr(self.q1) == "QuantumBit(index=1)"
