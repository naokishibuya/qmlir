"""Quantum Circuit Representation

This module defines the data structures for representing quantum circuits
before they are compiled to MLIR.
"""

from typing import List
from contextvars import ContextVar
from .gate import Gate


class QuantumCircuit:
    _current_circuit: ContextVar["QuantumCircuit"] = ContextVar("_current_circuit", default=None)

    def __init__(self, num_qubits: int, little_endian: bool = True):
        if num_qubits <= 0:
            raise ValueError("Quantum circuit must have at least one qubit.")

        self._num_qubits = num_qubits
        self.little_endian = little_endian
        self._gates: List[Gate] = []

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def gates(self) -> List[Gate]:
        return self._gates

    def append(self, gate: Gate):
        """Append a gate to the circuit."""
        if not isinstance(gate, Gate):
            raise TypeError(f"Expected Gate, got {type(gate).__name__}")
        if not all(0 <= q < self._num_qubits for q in gate.qubits):
            # find the first qubit that is out of range
            invalid_qubit = [q for q in gate.qubits if not (0 <= q < self._num_qubits)][0]
            raise ValueError(f"{invalid_qubit} out of range (num_qubits={self._num_qubits}).")
        self._gates.append(gate)

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
        lines += [f"  {op}" for op in self.gates]
        return "\n".join(lines)

    def __repr__(self):
        return f"QuantumCircuit(num_qubits={self._num_qubits}, ops={self.gates})"
