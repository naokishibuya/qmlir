from typing import Tuple
from .parameter import Parameter


class Gate:
    def __init__(
        self,
        name: str,
        qubits: Tuple[int, ...],
        parameters: Tuple[Parameter, ...] = (),
    ):
        self.name = name
        self.qubits = qubits
        self.parameters = parameters

        # Register the gate in the current quantum circuit
        from .circuit import QuantumCircuit

        current_circuit = QuantumCircuit.current()
        if current_circuit is None:
            raise RuntimeError("No active quantum circuit to append the gate to.")
        current_circuit.append(self)

    def __repr__(self):
        qstr = ", ".join(f"{q}" for q in self.qubits)
        pstr = f"({', '.join(map(str, self.parameters))})" if self.parameters else ""
        return f"{self.name}{pstr}|{qstr}⟩"
