"""Abstract Syntax Tree for Quantum Circuits

This module defines the data structures for representing quantum circuits
before they are compiled to MLIR.
"""

from typing import List


class Gate:
    """Represents a quantum gate operation."""

    def __init__(self, name: str, *qubits: int):
        """Initialize a quantum gate.

        Args:
            name: The name of the gate (e.g., 'x', 'h', 'cx')
            *qubits: The qubit indices this gate operates on
        """
        self.name = name
        self.q = qubits

    def __repr__(self) -> str:
        return f"Gate({self.name}, {', '.join(map(str, self.q))})"


class Circuit:
    """Represents a quantum circuit as a sequence of gates."""

    def __init__(self):
        """Initialize an empty quantum circuit."""
        self.gates: List[Gate] = []

    def x(self, qubit: int) -> "Circuit":
        """Add a Pauli-X gate to the circuit.

        Args:
            qubit: The qubit index to apply the gate to

        Returns:
            Self for method chaining
        """
        self.gates.append(Gate("x", qubit))
        return self

    def h(self, qubit: int) -> "Circuit":
        """Add a Hadamard gate to the circuit.

        Args:
            qubit: The qubit index to apply the gate to

        Returns:
            Self for method chaining
        """
        self.gates.append(Gate("h", qubit))
        return self

    def cx(self, control: int, target: int) -> "Circuit":
        """Add a CNOT gate to the circuit.

        Args:
            control: The control qubit index
            target: The target qubit index

        Returns:
            Self for method chaining
        """
        self.gates.append(Gate("cx", control, target))
        return self

    def __repr__(self) -> str:
        return f"Circuit({len(self.gates)} gates)"

    def __str__(self) -> str:
        lines = ["Circuit:"]
        for i, gate in enumerate(self.gates):
            lines.append(f"  {i}: {gate}")
        return "\n".join(lines)
