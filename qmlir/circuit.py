"""Quantum Circuit Representation

This module defines the data structures for representing quantum circuits
before they are compiled to MLIR.
"""

from typing import List
from contextvars import ContextVar
from .operator import Operator


class QuantumCircuit:
    _current_circuit: ContextVar["QuantumCircuit"] = ContextVar("_current_circuit", default=None)

    def __init__(self, num_qubits: int, little_endian: bool = True):
        if num_qubits <= 0:
            raise ValueError("Quantum circuit must have at least one qubit.")

        self._num_qubits = num_qubits
        self.little_endian = little_endian
        self._operators: List[Operator] = []

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def operators(self) -> List[Operator]:
        return self._operators

    def append(self, operator: Operator):
        """Append an operator to the circuit."""
        if not isinstance(operator, Operator):
            raise TypeError(f"Expected Operator, got {type(operator).__name__}")
        if not all(0 <= q < self._num_qubits for q in operator.qubits):
            # find the first qubit that is out of range
            out_of_range_qubits = [q for q in operator.qubits if not (0 <= q < self._num_qubits)]
            raise ValueError(f"Qubits {out_of_range_qubits} out of range [0, {self._num_qubits - 1}].")
        self._operators.append(operator)

    def __enter__(self):
        if QuantumCircuit._current_circuit.get():
            raise RuntimeError("Only one active quantum circuit can exist at a time.")
        self._token = QuantumCircuit._current_circuit.set(self)
        return self

    def __exit__(self, *ignored):
        QuantumCircuit._current_circuit.reset(self._token)

    @classmethod
    def current(cls) -> "QuantumCircuit":
        return cls._current_circuit.get()

    def __str__(self):
        lines = [f"QuantumCircuit({self._num_qubits} qubits):"]
        lines += [f"  {op}" for op in self.operators]
        return "\n".join(lines)

    def __repr__(self):
        return f"QuantumCircuit(num_qubits={self._num_qubits}, ops={self.operators})"
