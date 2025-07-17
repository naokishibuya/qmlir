"""Quantum Circuit Representation

This module defines the data structures for representing quantum circuits
before they are compiled to MLIR.
"""

from typing import List


AVAILABLE_QUANTUM_GATES = {
    "i": "Identity",
    "x": "Pauli-X",
    "y": "Pauli-Y",
    "z": "Pauli-Z",
    "h": "Hadamard",
    "cx": "CNOT",
}


class QuantumGate:
    """Represents a quantum gate operation."""

    def __init__(self, name: str, *qubits: int):
        """Initialize a quantum gate.

        Args:
            name: The name of the gate (e.g., 'x', 'h', 'cx')
            *qubits: The qubit indices this gate operates on
        """
        assert name in AVAILABLE_QUANTUM_GATES, f"Unknown gate: {name}"

        self.name = name
        self.q = qubits

    def __repr__(self) -> str:
        return f"Gate({self.name}, {', '.join(map(str, self.q))})"


class QuantumCircuit:
    """Represents a quantum circuit as a sequence of gates."""

    def __init__(self, num_qubits: int):
        """Initialize a quantum circuit with the specified number of qubits.

        Args:
            num_qubits: Number of qubits in the circuit
        """
        self.num_qubits = num_qubits
        self.gates: List[QuantumGate] = []

    def _validate_qubit(self, qubit: int) -> None:
        """Validate that qubit index is within bounds.

        Args:
            qubit: The qubit index to validate

        Raises:
            ValueError: If qubit index is out of bounds
        """
        if qubit < 0 or qubit >= self.num_qubits:
            raise ValueError(f"Qubit index {qubit} is out of bounds for {self.num_qubits}-qubit circuit")

    def i(self, qubit: int) -> "QuantumCircuit":
        """Add an identity gate to the circuit.

        Args:
            qubit: The qubit index to apply the gate to

        Returns:
            Self for method chaining
        """
        self._validate_qubit(qubit)
        self.gates.append(QuantumGate("i", qubit))
        return self

    def x(self, qubit: int) -> "QuantumCircuit":
        """Add a Pauli-X gate to the circuit.

        Args:
            qubit: The qubit index to apply the gate to

        Returns:
            Self for method chaining
        """
        self._validate_qubit(qubit)
        self.gates.append(QuantumGate("x", qubit))
        return self

    def y(self, qubit: int) -> "QuantumCircuit":
        """Add a Pauli-Y gate to the circuit.

        Args:
            qubit: The qubit index to apply the gate to

        Returns:
            Self for method chaining
        """
        self._validate_qubit(qubit)
        self.gates.append(QuantumGate("y", qubit))
        return self

    def z(self, qubit: int) -> "QuantumCircuit":
        """Add a Pauli-Z gate to the circuit.

        Args:
            qubit: The qubit index to apply the gate to

        Returns:
            Self for method chaining
        """
        self._validate_qubit(qubit)
        self.gates.append(QuantumGate("z", qubit))
        return self

    def h(self, qubit: int) -> "QuantumCircuit":
        """Add a Hadamard gate to the circuit.

        Args:
            qubit: The qubit index to apply the gate to

        Returns:
            Self for method chaining
        """
        self._validate_qubit(qubit)
        self.gates.append(QuantumGate("h", qubit))
        return self

    def cx(self, control: int, target: int) -> "QuantumCircuit":
        """Add a CNOT gate to the circuit.

        Args:
            control: The control qubit index
            target: The target qubit index

        Returns:
            Self for method chaining
        """
        self._validate_qubit(control)
        self._validate_qubit(target)
        if control == target:
            raise ValueError("Control and target qubits cannot be the same")
        self.gates.append(QuantumGate("cx", control, target))
        return self

    def __repr__(self) -> str:
        return f"QuantumCircuit({self.num_qubits} qubits, {len(self.gates)} gates)"

    def __str__(self) -> str:
        lines = [f"QuantumCircuit({self.num_qubits} qubits):"]
        for i, gate in enumerate(self.gates):
            lines.append(f"  {i}: {gate}")
        return "\n".join(lines)
