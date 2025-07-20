"""Quantum Circuit Representation

This module defines the data structures for representing quantum circuits
before they are compiled to MLIR.
"""

from typing import List
from .bit import QuantumBit
from .operator import Operator


class QuantumCircuit:
    def __init__(self, num_qubits: int):
        if num_qubits <= 0:
            raise ValueError("Quantum circuit must have at least one qubit.")

        self.num_qubits = num_qubits
        self.operators: List[Operator] = []
        self.qubits = [QuantumBit(i, self) for i in range(num_qubits)]

    def __str__(self):
        lines = [f"QuantumCircuit({self.num_qubits} qubits):"]
        lines += [f"  {op}" for op in self.operators]
        return "\n".join(lines)

    def __repr__(self):
        return f"QuantumCircuit(num_qubits={self.num_qubits}, ops={self.operators})"
