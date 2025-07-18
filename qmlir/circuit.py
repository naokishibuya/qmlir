"""Quantum Circuit Representation

This module defines the data structures for representing quantum circuits
before they are compiled to MLIR.
"""

from typing import List
from .parameter import Parameter
from .gate import QuantumGate


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

    def s(self, qubit: int) -> "QuantumCircuit":
        """Add an S gate (phase gate) to the circuit.

        Args:
            qubit: The qubit index to apply the gate to

        Returns:
            Self for method chaining
        """
        self._validate_qubit(qubit)
        self.gates.append(QuantumGate("s", qubit))
        return self

    def t(self, qubit: int) -> "QuantumCircuit":
        """Add a T gate (π/8 gate) to the circuit.

        Args:
            qubit: The qubit index to apply the gate to

        Returns:
            Self for method chaining
        """
        self._validate_qubit(qubit)
        self.gates.append(QuantumGate("t", qubit))
        return self

    def sdg(self, qubit: int) -> "QuantumCircuit":
        """Add an S-dagger gate to the circuit.

        Args:
            qubit: The qubit index to apply the gate to

        Returns:
            Self for method chaining
        """
        self._validate_qubit(qubit)
        self.gates.append(QuantumGate("sdg", qubit))
        return self

    def tdg(self, qubit: int) -> "QuantumCircuit":
        """Add a T-dagger gate to the circuit.

        Args:
            qubit: The qubit index to apply the gate to

        Returns:
            Self for method chaining
        """
        self._validate_qubit(qubit)
        self.gates.append(QuantumGate("tdg", qubit))
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
            raise ValueError("Control and target qubits must be different")
        self.gates.append(QuantumGate("cx", control, target))
        return self

    def cy(self, control: int, target: int) -> "QuantumCircuit":
        """Add a controlled-Y gate to the circuit.

        Args:
            control: The control qubit index
            target: The target qubit index

        Returns:
            Self for method chaining
        """
        self._validate_qubit(control)
        self._validate_qubit(target)
        if control == target:
            raise ValueError("Control and target qubits must be different")
        self.gates.append(QuantumGate("cy", control, target))
        return self

    def cz(self, control: int, target: int) -> "QuantumCircuit":
        """Add a controlled-Z gate to the circuit.

        Args:
            control: The control qubit index
            target: The target qubit index

        Returns:
            Self for method chaining
        """
        self._validate_qubit(control)
        self._validate_qubit(target)
        if control == target:
            raise ValueError("Control and target qubits must be different")
        self.gates.append(QuantumGate("cz", control, target))
        return self

    def rx(self, qubit: int, parameter: Parameter) -> "QuantumCircuit":
        """Add a rotation-X gate to the circuit.

        Args:
            qubit: The qubit index to apply the gate to
            parameter: The rotation parameter

        Returns:
            Self for method chaining
        """
        self._validate_qubit(qubit)
        self.gates.append(QuantumGate("rx", qubit, parameters=[parameter]))
        return self

    def ry(self, qubit: int, parameter: Parameter) -> "QuantumCircuit":
        """Add a rotation-Y gate to the circuit.

        Args:
            qubit: The qubit index to apply the gate to
            parameter: The rotation parameter

        Returns:
            Self for method chaining
        """
        self._validate_qubit(qubit)
        self.gates.append(QuantumGate("ry", qubit, parameters=[parameter]))
        return self

    def rz(self, qubit: int, parameter: Parameter) -> "QuantumCircuit":
        """Add a rotation-Z gate to the circuit.

        Args:
            qubit: The qubit index to apply the gate to
            parameter: The rotation parameter

        Returns:
            Self for method chaining
        """
        self._validate_qubit(qubit)
        self.gates.append(QuantumGate("rz", qubit, parameters=[parameter]))
        return self

    def __repr__(self) -> str:
        return f"QuantumCircuit({self.num_qubits} qubits, {len(self.gates)} gates)"

    def __str__(self) -> str:
        gate_strs = [str(gate) for gate in self.gates]
        return f"QuantumCircuit({self.num_qubits} qubits):\n" + "\n".join(f"  {gate}" for gate in gate_strs)
