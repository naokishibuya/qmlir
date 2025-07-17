"""Bell state circuit - basic example."""

from qmlir import QuantumCircuit, circuit_to_mlir


def main():
    """Create a Bell state circuit."""
    # Create a Bell state |00⟩ + |11⟩
    circuit = QuantumCircuit(2)
    circuit.h(0)  # Hadamard on qubit 0
    circuit.cx(0, 1)  # CNOT from qubit 0 to qubit 1

    print("Bell State Circuit:")
    print(circuit)
    print()

    mlir_code = circuit_to_mlir(circuit, "bell_state")

    print("MLIR representation:")
    print(mlir_code)


if __name__ == "__main__":
    main()
