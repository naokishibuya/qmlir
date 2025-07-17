"""Custom circuit example - shows various quantum gates."""

from qmlir import Circuit


def main():
    """Create a custom quantum circuit with various gates."""
    # Create a more complex circuit
    circuit = Circuit()

    # Create superposition on qubit 0
    circuit.h(0)

    # Add some X gates (these will be optimized away)
    circuit.x(1)
    circuit.x(1)

    # Create entanglement
    circuit.cx(0, 1)

    # Add another Hadamard
    circuit.h(2)

    # Another CNOT
    circuit.cx(1, 2)

    print("Custom Circuit:")
    print(f"Gates: {circuit.gates}")
    print()
    print("MLIR representation:")
    print(circuit.to_mlir())
    print()
    print("Note: The double X gates on qubit 1 can be optimized away")
    print("      using: quantum-opt --quantum-cancel-x")


if __name__ == "__main__":
    main()
