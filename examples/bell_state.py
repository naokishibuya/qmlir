"""Bell state circuit - basic example."""

from qmlir import Circuit


def main():
    """Create a Bell state circuit."""
    # Create a Bell state |00⟩ + |11⟩
    circuit = Circuit()
    circuit.h(0)  # Hadamard on qubit 0
    circuit.cx(0, 1)  # CNOT from qubit 0 to qubit 1

    print("Bell State Circuit:")
    print(f"Gates: {circuit.gates}")
    print()
    print("MLIR representation:")
    print(circuit)


if __name__ == "__main__":
    main()
