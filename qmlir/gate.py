"""Quantum Gate Representation

This module defines the QuantumGate class and related constants for representing
quantum gate operations in circuits.
"""

from typing import List
from .parameter import Parameter


AVAILABLE_QUANTUM_GATES = {
    "i": "Identity",
    "x": "Pauli-X",
    "y": "Pauli-Y",
    "z": "Pauli-Z",
    "h": "Hadamard",
    "s": "S gate (Phase)",
    "t": "T gate (π/8)",
    "sdg": "S-dagger",
    "tdg": "T-dagger",
    "cx": "CNOT",
    "cy": "Controlled-Y",
    "cz": "Controlled-Z",
    "rx": "Rotation-X",
    "ry": "Rotation-Y",
    "rz": "Rotation-Z",
}


class QuantumGate:
    """Represents a quantum gate operation."""

    def __init__(self, name: str, *qubits: int, parameters: List[Parameter] = None):
        """Initialize a quantum gate.

        Args:
            name: The name of the gate (e.g., 'x', 'h', 'cx', 'rx')
            *qubits: The qubit indices this gate operates on
            parameters: List of Parameter objects for parametric gates
        """
        assert name in AVAILABLE_QUANTUM_GATES, f"Unknown gate: {name}"

        self.name = name
        self.q = qubits
        self.parameters = parameters or []

    @property
    def description(self) -> str:
        """Return the name of the gate."""
        return AVAILABLE_QUANTUM_GATES[self.name]

    def __repr__(self) -> str:
        if self.parameters:
            param_str = ", ".join(str(p) for p in self.parameters)
            return f"{self.name.upper()}({', '.join(map(str, self.q))}, [{param_str}])"
        else:
            return f"{self.name.upper()}({', '.join(map(str, self.q))})"
