from qmlir.circuit import QuantumCircuit
from qmlir.operator import X, RX, H, CX, RZ
from qmlir.mlir.transpiler import apply_passes, circuit_to_mlir


def test_basic_operator_transpilation():
    circuit = QuantumCircuit(2)
    with circuit:
        X(0)
        H(1)
        CX(0, 1)
    mlir_code = circuit_to_mlir(circuit)
    assert "quantum.x" in mlir_code
    assert "quantum.h" in mlir_code
    assert "quantum.cx" in mlir_code


def test_rotation_operator_transpilation():
    with QuantumCircuit(1) as circuit:
        theta = 1.57
        RX(theta)(0)
        RZ(3.14)(0)
    mlir_code = circuit_to_mlir(circuit)
    assert "quantum.rx" in mlir_code
    assert "quantum.rz" in mlir_code
    assert "func.func" in mlir_code
    assert "return" in mlir_code


def test_multiple_qubits_allocated():
    circuit = QuantumCircuit(3)
    with circuit:
        H(0)
        X(1)
        CX(1, 2)
    mlir_code = circuit_to_mlir(circuit)
    assert mlir_code.count("quantum.alloc") == 3
    assert "quantum.h" in mlir_code
    assert "quantum.x" in mlir_code
    assert "quantum.cx" in mlir_code


def test_optimization_passes():
    circuit = QuantumCircuit(3)
    with circuit:
        X(0)
        X(0)
        H(1)
    mlir_code = circuit_to_mlir(circuit)
    print("=== Original MLIR ===")
    print(mlir_code)

    optimized_mlir = apply_passes(mlir_code)
    print("=== Optimized MLIR ===")
    print(optimized_mlir)

    assert "quantum.x" in mlir_code
    assert "quantum.h" in mlir_code
    assert "quantum.x" not in optimized_mlir
    assert "quantum.h" in optimized_mlir
