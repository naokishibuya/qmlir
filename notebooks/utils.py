from qmlir import QuantumCircuit, Observable, JaxSimulator


def evaluate_circuit(circuit: QuantumCircuit, observable: Observable = None, shots: int = 1000):
    """Test basic quantum operators."""
    simulator = JaxSimulator()
    state = simulator.statevector(circuit)
    probs = simulator.probabilities(circuit)
    expval = simulator.expectation(circuit, observable)
    samples = simulator.measure(circuit, shots)

    print(f"Circuit: {circuit}")
    print("\nCompiled MLIR:")
    print(circuit.compiled_mlir)
    print()
    print(f"State vector: {state}")
    print(f"Measurement probabilities: {probs}")
    print(f"Expectation value: {expval}")
    print(f"Samples: {samples} ({shots} shots)")
