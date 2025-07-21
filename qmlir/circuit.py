"""Quantum Circuit Representation

This module defines the data structures for representing quantum circuits
before they are compiled to MLIR.
"""

from typing import List
from .operator import Operator


class QuantumCircuit:
    _active_circuit = None

    def __init__(self, num_qubits: int, little_endian: bool = True):
        if num_qubits <= 0:
            raise ValueError("Quantum circuit must have at least one qubit.")

        self.num_qubits = num_qubits
        self.little_endian = little_endian
        self.operators: List[Operator] = []

    def __enter__(self):
        if QuantumCircuit._active_circuit is not None:
            raise RuntimeError("Only one active quantum circuit can exist at a time.")
        QuantumCircuit._active_circuit = self
        return self

    def __exit__(self, *ignored):
        QuantumCircuit._active_circuit = None

    @classmethod
    def active_circuit(cls):
        return cls._active_circuit

    def __str__(self):
        lines = [f"QuantumCircuit({self.num_qubits} qubits):"]
        lines += [f"  {op}" for op in self.operators]
        return "\n".join(lines)

    def __repr__(self):
        return f"QuantumCircuit(num_qubits={self.num_qubits}, ops={self.operators})"
