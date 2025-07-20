from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .circuit import QuantumCircuit


class QuantumBit:
    def __init__(self, index: int, circuit: "QuantumCircuit"):
        self.index = index
        self._circuit = circuit

    @property
    def circuit(self) -> "QuantumCircuit":
        return self._circuit

    def __eq__(self, other):
        if isinstance(other, QuantumBit):
            return self.index == other.index and self.circuit == other.circuit
        return False

    def __hash__(self):
        return hash((self.index, id(self.circuit)))

    def __repr__(self):
        return f"QuantumBit(index={self.index})"

    def __str__(self):
        return f"q[{self.index}]"
