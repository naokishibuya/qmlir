import pytest
from dataclasses import dataclass
from qmlir.circuit import QuantumCircuit
from qmlir.operator import X, RX, H, CX, RZ
from qmlir.mlir.transpiler import circuit_to_mlir


def test_basic_operator_transpilation():
    qc = QuantumCircuit(2)
    q0, q1 = qc.qubits
    X(q0)
    H(q1)
    CX(q0, q1)
    mlir_code = circuit_to_mlir(qc)
    assert "quantum.x" in mlir_code
    assert "quantum.h" in mlir_code
    assert "quantum.cx" in mlir_code


def test_rotation_operator_transpilation():
    qc = QuantumCircuit(1)
    q0 = qc.qubits[0]
    theta = 1.57
    RX(theta)(q0)
    RZ(3.14)(q0)
    mlir_code = circuit_to_mlir(qc)
    assert "quantum.rx" in mlir_code
    assert "quantum.rz" in mlir_code
    assert "func.func" in mlir_code
    assert "return" in mlir_code


def test_multiple_qubits_allocated():
    qc = QuantumCircuit(3)
    q0, q1, q2 = qc.qubits
    H(q0)
    X(q1)
    CX(q1, q2)
    mlir_code = circuit_to_mlir(qc)
    assert mlir_code.count("quantum.alloc") == 3
    assert "quantum.h" in mlir_code
    assert "quantum.x" in mlir_code
    assert "quantum.cx" in mlir_code


@dataclass(frozen=True)
class DummyMetadata:
    name: str
    kind: str
    long_name: str
    hermitian: bool
    self_inverse: bool
    unitary: bool = True


def test_unknown_operator_raises():
    qc = QuantumCircuit(1)
    q0 = qc.qubits[0]

    class DummyOperator:
        metadata = DummyMetadata(
            name="Dummy", kind="dummy", long_name="Dummy Operator", hermitian=False, self_inverse=False
        )
        name = "dummy"
        parameters = []
        qubits = (q0,)

    qc.operators.append(DummyOperator())

    with pytest.raises(AssertionError, match="Unknown operator: DummyOperator"):
        circuit_to_mlir(qc)
